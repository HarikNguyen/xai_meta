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
    ):
        """Deploy on single task
        1. Fast adaptation on support set (sup_x, sup_y) for T steps
        2. Eval on query set (que_x, que_y) after each update step
        3. Return losses and preds at each step
        """
        # init fast_weights with theta_0 (phi)
        fast_weights = [p.clone() for p in theta_0]
        learner = self.baselearner

        # init results list
        sup_losses, que_losses, sup_preds_list, que_preds_list, sup_accs, que_accs = [], [], [], [], [], []
        
        # get pre-update (theta_0) loss and predictions
        values_n_grad_fn = tf.grad_and_value(get_loss_n_preds, has_aux=True)
        grads, (pre_sup_loss, pre_sup_pred) = values_n_grad_fn(fast_weights, learner, sup_x, sup_y)
        pre_que_loss, pre_que_pred = get_loss_n_preds(fast_weights, learner, que_x, que_y)

        sup_losses.append(pre_sup_loss)
        que_losses.append(pre_que_loss)
        sup_preds_list.append(pre_sup_pred)
        que_preds_list.append(pre_que_pred)
        sup_accs.append(calc_accuracy(pre_sup_pred, sup_y))
        que_accs.append(calc_accuracy(pre_que_pred, que_y))

        for _ in range(T):
            # get fast_weights
            fast_weights = self._fast_weights(
                params=fast_weights,
                gradients=grads,
                train_mode=train_mode,
            )
            # get loss and predictions
            grads, (sup_loss, sup_pred) = values_n_grad_fn(fast_weights, learner, sup_x, sup_y)
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
        )

        meta_loss = que_losses[-1].mean()

        meta_loss.backward()

        self.outer_optim.step()

        return meta_loss.item()

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
