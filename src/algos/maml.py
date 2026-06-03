import torch
import torch.func as tf
from .base import BaseAlgorithm
from .utils import get_loss_with_grad, put_on_device


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
        self.baselearner = self.baselearner_fn(**self.baselearner_args).to(self.device)
        self.theta_0 = [
            p.clone().detach().requires_grad_(True) for p in self.baselearner.parameters()
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
        fast_weights = theta_0
        learner = self.baselearner

        # init results list
        sup_losses, que_losses, sup_preds_list, que_preds_list = [], [], [], []
        
        # get pre-update (theta_0) loss and predictions
        pre_sup_loss, pre_sup_pred, grads = get_loss_with_grad(learner, sup_x, sup_y, fast_weights, return_grad=True)
        pre_que_loss, pre_que_pred = get_loss_with_grad(learner, que_x, que_y, fast_weights)

        sup_losses.append(pre_sup_loss)
        que_losses.append(pre_que_loss)
        sup_preds_list.append(pre_sup_pred)
        que_preds_list.append(pre_que_pred)

        for _ in range(T):
            # get fast_weights
            fast_weights = self._fast_weights(
                params=fast_weights,
                gradients=grads,
                train_mode=train_mode,
            )
            # get loss and predictions
            sup_loss, sup_pred, grads = get_loss_with_grad(learner, sup_x, sup_y, fast_weights, return_grad=True)
            que_loss, que_pred = get_loss_with_grad(learner, que_x, que_y, fast_weights)

            sup_losses.append(sup_loss)
            que_losses.append(que_loss)
            sup_preds_list.append(sup_pred)
            que_preds_list.append(que_pred)

        return sup_losses, que_losses, sup_preds_list, que_preds_list

    def set_train_mode(self):
        self.baselearner.train()

    def set_val_mode(self):
        self.baselearner.eval()

    def train(self, sup_x, sup_y, que_x, que_y):
        sup_x, sup_y, que_x, que_y = put_on_device(self.device, [sup_x, sup_y, que_x, que_y])
        self.outer_optim.zero_grad()
        vmap_deploy = tf.vmap(
            self._deploy, 
            in_dims=(None, 0, 0, 0, 0), 
            chunk_size=self.vmap_chunk_size
        )

        sup_losses, que_losses, sup_preds_list, que_preds_list = vmap_deploy(
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

    def val(self, train_x, train_y, test_x, test_y):
        pass

    def dump_state(self):
        """Return the state of the meta-learner

        Returns
        ----------
        initialization
            Initialization parameters
        """
        return [p.clone().detach().to(self.device) for p in self.theta_0]

    def load_state(self, state):
        """Load the given state into the meta-learner

        Parameters
        ----------
        state : initialization
            Initialization parameters
        """

        self.initialization = [p.clone() for p in state]
        for p in self.theta_0:
            p.requires_grad = True

    def to(self, device):
        """to device"""
        self.baselearner = self.baselearner.to(device)
        self.theta_0 = [p.to(device) for p in self.theta_0]
