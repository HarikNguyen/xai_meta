import torch
import torch.func as tf
from .base import BaseAlgorithm
from .utils import get_loss_n_preds, put_on_device, calc_accuracy


class MAML(BaseAlgorithm):
    def __init__(
        self,
        train_base_lr,
        base_lr,
        second_order,
        meta_batch_size=1,
        grad_clip=None,
        vmap_chunk_size=None,
        grad_accum_tasks=False,
        **kwargs,
    ):
        """Initialization of MAML

        Parameters
        ----------
        train_base_lr: float
            Inner level learning rate for meta-training
        base_lr: float
            Inner level learning rate
        second_order: bool
            Whether to use second-order gradient information
        meta_batch_size: int
            Number of tasks to compute outer-update
        grad_accum_tasks: bool
            If True, MAML.train() loops over the meta_batch_size tasks one at
            a time (calling backward()+accumulating after each), instead of
            batching all tasks through a single vmap call + one backward().
            Mathematically identical result (mean of per-task gradients is
            linear, so accumulating it one task at a time gives the same
            total gradient as computing it all at once) -- the only thing
            that changes is that at most 1 task's full second-order inner-
            loop graph is ever resident in memory at a time instead of
            meta_batch_size of them simultaneously. Trades some wall-clock
            speed (no longer vectorized across tasks) for peak VRAM, useful
            for backbones too large to fit meta_batch_size tasks at once
            (e.g. Res12) without lowering meta_batch_size/k_query/T/second_order.
        **kwargs: dict
            Keyword arguments that are ignored
        """
        super().__init__(**kwargs)
        # hyperparameters
        self.train_base_lr = train_base_lr
        self.base_lr = base_lr
        self.second_order = second_order
        self.meta_batch_size = meta_batch_size
        self.grad_clip = grad_clip
        self.vmap_chunk_size = vmap_chunk_size
        self.grad_accum_tasks = grad_accum_tasks

        # get random initialization point for baselearner (theta_0)
        self.baselearner = self.baselearner_fn(**self.baselearner_args)
        self.theta_0 = [
            p.clone().to(self.device).detach().requires_grad_(True) for p in self.baselearner.parameters()
        ]

        # define outer-level optimizer
        self.outer_optim = self.optim_fn(self.theta_0, lr=self.lr)

    def _fast_weights(self, params, gradients, train_mode=False):
        """Compute task-specific weights using the gradients (theta_t)
        """
        lr = self.base_lr if not train_mode else self.train_base_lr

        # Clip gradient values between (-10,+10)
        if not self.grad_clip is None:
            gradients = [
                torch.clamp(p, -1 * self.grad_clip, self.grad_clip) for p in gradients
            ]

        # Return task-specific weights
        fast_weights = [(params[i] - lr * gradients[i])for i in range(len(gradients))]
        return fast_weights

    def _deploy(
        self,
        theta_0,
        sup_x,
        sup_y,
        que_x,
        que_y,
        train_mode,
        T,
        full_trajectory=True,
    ):
        """Deploy on single task
        1. Fast adaptation on support set (sup_x, sup_y) for T steps
        2. Eval on query set (que_x, que_y) after each update step
        3. Return losses and preds at each step

        full_trajectory: bool
            If True (val/test), record support+query loss/pred/acc at every one
            of the T+1 steps, as before. If False (train), only the LAST step's
            query loss is ever used by the caller (see MAML.train), so every
            intermediate query forward pass + accuracy computation is skipped.
            The support-side forward/grad is never skipped: it feeds the next
            _fast_weights update and is required by the inner-loop recursion
            regardless of full_trajectory.
        """
        # init fast_weights with theta_0 (phi)
        fast_weights = [p.clone() for p in theta_0]
        learner = self.baselearner

        # init results list
        sup_losses, que_losses, sup_preds_list, que_preds_list, sup_accs, que_accs = [], [], [], [], [], []

        # get pre-update (theta_0) loss and predictions
        values_n_grad_fn = tf.grad_and_value(get_loss_n_preds, has_aux=True)
        grads, (pre_sup_loss, pre_sup_pred) = values_n_grad_fn(fast_weights, learner, sup_x, sup_y)

        if full_trajectory:
            pre_que_loss, pre_que_pred = get_loss_n_preds(fast_weights, learner, que_x, que_y)

            sup_losses.append(pre_sup_loss)
            que_losses.append(pre_que_loss)
            sup_preds_list.append(pre_sup_pred)
            que_preds_list.append(pre_que_pred)
            sup_accs.append(calc_accuracy(pre_sup_pred, sup_y))
            que_accs.append(calc_accuracy(pre_que_pred, que_y))

        for step in range(T):
            # get fast_weights
            fast_weights = self._fast_weights(
                params=fast_weights,
                gradients=grads,
                train_mode=train_mode,
            )
            # get loss and predictions
            grads, (sup_loss, sup_pred) = values_n_grad_fn(fast_weights, learner, sup_x, sup_y)

            is_last_step = step == T - 1
            if full_trajectory or is_last_step:
                que_loss, que_pred = get_loss_n_preds(fast_weights, learner, que_x, que_y)

                sup_losses.append(sup_loss)
                que_losses.append(que_loss)
                sup_preds_list.append(sup_pred)
                que_preds_list.append(que_pred)
                sup_accs.append(calc_accuracy(sup_pred, sup_y))
                que_accs.append(calc_accuracy(que_pred, que_y))

        return sup_losses, que_losses, sup_preds_list, que_preds_list, sup_accs, que_accs

    def train(self, sup_x, sup_y, que_x, que_y):
        sup_x, sup_y, que_x, que_y = put_on_device(self.device, [sup_x, sup_y, que_x, que_y])
        self.outer_optim.zero_grad()

        if self.grad_accum_tasks:
            return self._train_grad_accum(sup_x, sup_y, que_x, que_y)

        vmap_deploy = tf.vmap(
            self._deploy,
            in_dims=(None, 0, 0, 0, 0),
            chunk_size=self.vmap_chunk_size
        )

        _, que_losses, _, _, _, _ = vmap_deploy(
            self.theta_0,
            sup_x,
            sup_y,
            que_x,
            que_y,
            train_mode=True,
            T=self.T,
            full_trajectory=False,
        )

        meta_loss = que_losses[-1].mean()

        meta_loss.backward()

        self.outer_optim.step()

        return meta_loss.item()

    def _train_grad_accum(self, sup_x, sup_y, que_x, que_y):
        """Same math as train()'s vmap path (mean of per-task query losses,
        gradient w.r.t. theta_0), but processes one task at a time so only
        one task's second-order inner-loop graph is resident at once.
        """
        num_tasks = sup_x.shape[0]
        meta_loss = 0.0

        for i in range(num_tasks):
            _, que_losses, _, _, _, _ = self._deploy(
                self.theta_0,
                sup_x[i],
                sup_y[i],
                que_x[i],
                que_y[i],
                train_mode=True,
                T=self.T,
                full_trajectory=False,
            )
            task_loss = que_losses[-1] / num_tasks
            task_loss.backward()
            meta_loss += task_loss.item()

        self.outer_optim.step()

        return meta_loss

    def val(self, sup_x, sup_y, que_x, que_y):
        sup_losses, que_losses, sup_accs, que_accs = self._validate(sup_x, sup_y, que_x, que_y, T=self.T_val)

        pre_results = {
            "sup_loss": sup_losses[0].mean().item(),
            "que_loss": que_losses[0].mean().item(),
            "sup_acc": sup_accs[0].mean().item(),
            "que_acc": que_accs[0].mean().item(),
        }
        post_results = {
            "sup_loss": sup_losses[-1].mean().item(),
            "que_loss": que_losses[-1].mean().item(),
            "sup_acc": sup_accs[-1].mean().item(),
            "que_acc": que_accs[-1].mean().item(),
        }

        return pre_results, post_results

    def test(self, sup_x, sup_y, que_x, que_y):
        return self._validate(sup_x, sup_y, que_x, que_y, T=self.T_test)

    def _validate(self, sup_x, sup_y, que_x, que_y, T):
        sup_x, sup_y, que_x, que_y = put_on_device(self.device, [sup_x, sup_y, que_x, que_y])

        if self.grad_accum_tasks:
            return self._validate_looped(sup_x, sup_y, que_x, que_y, T)

        vmap_deploy = tf.vmap(
            self._deploy,
            in_dims=(None, 0, 0, 0, 0),
            chunk_size=self.vmap_chunk_size
        )

        sup_losses, que_losses, _, _, sup_accs, que_accs = vmap_deploy(
            self.theta_0,
            sup_x,
            sup_y,
            que_x,
            que_y,
            train_mode=False,
            T=T,
        )
        return sup_losses, que_losses, sup_accs, que_accs

    def _validate_looped(self, sup_x, sup_y, que_x, que_y, T):
        """Same output as _validate()'s vmap path (per-step tensors of shape
        (num_tasks,)), but processes one task at a time so only one task's
        inner-loop graph is resident at once -- val()/test() never call
        backward(), but torch.func.grad_and_value still builds a
        differentiable graph internally for every _deploy call regardless
        (train or not), so batching num_tasks of them via vmap costs the same
        peak memory as train() did before grad_accum_tasks. Detaching each
        task's results immediately lets that graph be freed before the next
        task starts.
        """
        num_tasks = sup_x.shape[0]
        per_task_sup_losses, per_task_que_losses = [], []
        per_task_sup_accs, per_task_que_accs = [], []

        for i in range(num_tasks):
            sup_losses, que_losses, _, _, sup_accs, que_accs = self._deploy(
                self.theta_0,
                sup_x[i],
                sup_y[i],
                que_x[i],
                que_y[i],
                train_mode=False,
                T=T,
            )
            per_task_sup_losses.append([t.detach() for t in sup_losses])
            per_task_que_losses.append([t.detach() for t in que_losses])
            per_task_sup_accs.append(sup_accs)
            per_task_que_accs.append(que_accs)

        num_steps = len(per_task_sup_losses[0])
        sup_losses = [torch.stack([per_task_sup_losses[i][s] for i in range(num_tasks)]) for s in range(num_steps)]
        que_losses = [torch.stack([per_task_que_losses[i][s] for i in range(num_tasks)]) for s in range(num_steps)]
        sup_accs = [torch.stack([per_task_sup_accs[i][s] for i in range(num_tasks)]) for s in range(num_steps)]
        que_accs = [torch.stack([per_task_que_accs[i][s] for i in range(num_tasks)]) for s in range(num_steps)]
        return sup_losses, que_losses, sup_accs, que_accs
        
    def dump_state(self):
        """Return the state of the meta-learner
        """
        return [p.clone().detach() for p in self.theta_0]

    def load_state(self, state):
        """Load the given state into the meta-learner
        """
        with torch.no_grad():
            for p_current, p_loaded in zip(self.theta_0, state):
                p_current.copy_(p_loaded)
