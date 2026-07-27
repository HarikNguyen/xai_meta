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
        grad_accum_chunk_size=1,
        **kwargs,
    ):
        """MAML: train_base_lr/base_lr are the inner-loop LR for train vs val/test.
        grad_accum_tasks trades vectorized speed for lower peak VRAM by looping
        tasks in grad_accum_chunk_size groups with backward() after each."""
        super().__init__(**kwargs)
        # hyperparameters
        self.train_base_lr = train_base_lr
        self.base_lr = base_lr
        self.second_order = second_order
        self.meta_batch_size = meta_batch_size
        self.grad_clip = grad_clip
        self.vmap_chunk_size = vmap_chunk_size
        self.grad_accum_tasks = grad_accum_tasks
        self.grad_accum_chunk_size = grad_accum_chunk_size

        # get random initialization point for baselearner (theta_0)
        self.baselearner = self.baselearner_fn(**self.baselearner_args)
        self.theta_0 = [
            p.clone().to(self.device).detach().requires_grad_(True) for p in self.baselearner.parameters()
        ]

        # fused=True updates theta_0 in one CUDA kernel instead of per-tensor; falls back if unsupported
        try:
            self.outer_optim = self.optim_fn(self.theta_0, lr=self.lr, fused=(self.device == "cuda"))
        except TypeError:
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
        """Fast-adapt on the support set for T steps, evaluating on the query set at
        each step. full_trajectory=False (train) skips intermediate query
        forward passes since only the last step's query loss is used."""
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

    def _chunk_ranges(self, num_tasks):
        chunk_size = max(1, min(self.grad_accum_chunk_size, num_tasks))
        return [(start, min(start + chunk_size, num_tasks)) for start in range(0, num_tasks, chunk_size)]

    def _deploy_chunk(self, start, end, sup_x, sup_y, que_x, que_y, train_mode, T, full_trajectory):
        """Run _deploy on tasks [start, end): vmap-batched if >1 task, direct call if just 1."""
        if end - start == 1:
            return self._deploy(
                self.theta_0, sup_x[start], sup_y[start], que_x[start], que_y[start],
                train_mode=train_mode, T=T, full_trajectory=full_trajectory,
            )
        vmap_deploy = tf.vmap(self._deploy, in_dims=(None, 0, 0, 0, 0))
        return vmap_deploy(
            self.theta_0, sup_x[start:end], sup_y[start:end], que_x[start:end], que_y[start:end],
            train_mode=train_mode, T=T, full_trajectory=full_trajectory,
        )

    def _train_grad_accum(self, sup_x, sup_y, que_x, que_y):
        """Same math as train()'s vmap path, but processed in grad_accum_chunk_size
        groups so at most that many tasks' graphs are resident at once."""
        num_tasks = sup_x.shape[0]
        meta_loss = 0.0

        for start, end in self._chunk_ranges(num_tasks):
            _, que_losses, _, _, _, _ = self._deploy_chunk(
                start, end, sup_x, sup_y, que_x, que_y,
                train_mode=True, T=self.T, full_trajectory=False,
            )
            # .sum() handles scalar or vector shapes; dividing by num_tasks keeps this the mean over all tasks
            chunk_loss = que_losses[-1].sum() / num_tasks
            chunk_loss.backward()
            meta_loss += chunk_loss.item()

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
        """Same output as _validate()'s vmap path, but chunked like _train_grad_accum
        so grad_and_value's graphs (built even without backward()) don't all
        stay resident at once; each chunk is detached before the next runs."""
        num_tasks = sup_x.shape[0]
        chunk_sup_losses, chunk_que_losses = [], []
        chunk_sup_accs, chunk_que_accs = [], []

        for start, end in self._chunk_ranges(num_tasks):
            sup_losses, que_losses, _, _, sup_accs, que_accs = self._deploy_chunk(
                start, end, sup_x, sup_y, que_x, que_y,
                train_mode=False, T=T, full_trajectory=True,
            )
            is_single = end - start == 1
            # unsqueeze size-1 chunks (0-dim scalars) so every chunk has a leading task-dim for torch.cat below
            to_group = lambda t: t.detach().unsqueeze(0) if is_single else t.detach()
            to_group_acc = lambda t: t.unsqueeze(0) if is_single else t

            chunk_sup_losses.append([to_group(t) for t in sup_losses])
            chunk_que_losses.append([to_group(t) for t in que_losses])
            chunk_sup_accs.append([to_group_acc(t) for t in sup_accs])
            chunk_que_accs.append([to_group_acc(t) for t in que_accs])

        num_steps = len(chunk_sup_losses[0])
        num_chunks = len(chunk_sup_losses)
        sup_losses = [torch.cat([chunk_sup_losses[c][s] for c in range(num_chunks)]) for s in range(num_steps)]
        que_losses = [torch.cat([chunk_que_losses[c][s] for c in range(num_chunks)]) for s in range(num_steps)]
        sup_accs = [torch.cat([chunk_sup_accs[c][s] for c in range(num_chunks)]) for s in range(num_steps)]
        que_accs = [torch.cat([chunk_que_accs[c][s] for c in range(num_chunks)]) for s in range(num_steps)]
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
