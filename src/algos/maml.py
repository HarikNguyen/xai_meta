import torch
from torch.distributed.optim import DistributedOptimizer
from .base import BaseAlgorithm
from .utils import *


class MAML(BaseAlgorithm):
    def __init__(
        self,
        train_base_lr,
        base_lr,
        second_order,
        meta_batch_size=1,
        grad_clip=None,
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
        self.train_base_lr = train_base_lr
        self.base_lr = base_lr
        self.second_order = second_order
        self.meta_batch_size = meta_batch_size
        self.grad_clip = grad_clip

        # Increment after every train step on a single task, and update
        # init when task_counter % meta_batch_size == 0
        self.task_counter = 0

        self.test_loss_sum = 0.0
        # Maintain train loss history
        self.train_losses = []

        # Get random initialization point for baselearner
        self.baselearner = self.baselearner_fn(**self.baselearner_args).to(self.device)
        self.initialization = [
            p.clone().detach().to(self.device) for p in self.baselearner.parameters()
        ]

        # Store gradients across tasks
        self.grad_buffer = [
            torch.zeros(p.size(), device=self.device) for p in self.initialization
        ]

        # Enable gradient tracking for the initialization parameters
        for p in self.initialization:
            p.requires_grad = True

        # Initialize the meta-optimizer
        self.optimizer = self.optim_fn(self.initialization, lr=self.lr)

        # Maintain test history
        self.test_losses = []
        self.test_perfs = []

    def _get_params(self):
        return [p.clone().detach() for p in self.initialization]

    def _fast_weights(self, params, gradients, train_mode=False):
        """Compute task-specific weights using the gradients (theta*)

        Apply a single step of gradient descent using the provided gradients
        to compute task-specific, or equivalently, fast, weights.

        Parameters
        ----------
        params : list
            List of parameter tensors
        gradients : list
            List of torch.Tensor variables containing the gradients per layer
        """
        lr = self.base_lr if not train_mode else self.train_base_lr

        # Clip gradient values between (-10,+10)
        if not self.grad_clip is None:
            gradients = [
                torch.clamp(p, -1 * self.grad_clip, self.grad_clip) for p in gradients
            ]

        # Return task-specific weights
        fast_weights = [(params[i] - lr * gradients[i]) for i in range(len(gradients))]
        return fast_weights

    def _deploy(
        self,
        train_x,
        train_y,
        test_x,
        test_y,
        train_mode,
        T,
        rpc_mode=False,
    ):
        """Run DOSO on a single task to get the loss on the query set

        1. Compute the base-learner loss and gradients on the support set
        (train_x, train_y) using our initialization point.
        2. Make a few weight update based on this information.
        3. Evaluate and return the loss of the fast weights on the query set
        (test_x, test_y).

        Parameters
        ----------
        train_x: torch.Tensor
            Inputs of the support set
        train_y: torch.Tensor
            Outputs of the support set
        test_x: torch.Tensor
            Inputs of the query set
        test_y: torch.Tensor
            Outputs of the query set
        train_mode: boolean
            Whether we are in training mode or test mode

        Returns
        ----------
        test_loss
            Loss of the base-learner on the query set after the proposed
            one-step update
        """
        learner = self.baselearner
        # Copy initialization parameters to fast_weights parameters
        if rpc_mode:
            fast_weights = self.initialization
        else:
            fast_weights = [p.clone() for p in self.initialization]

        # ----- Pre-update (theta_0) -----
        with torch.no_grad():
            pre_preds = learner.forward_weights(train_x, fast_weights)
            pre_loss = learner.criterion(pre_preds, train_y)

        # ----- Inner loop (T updates) -----
        # Train on support set (train_x, train_y)
        for _ in range(T):
            # Get loss and grads
            _, grads = get_loss_and_grads(
                learner,
                train_x,
                train_y,
                weights=fast_weights,
                create_graph=self.second_order,
                retain_graph=T > 1 or self.second_order,
                flat=False,
            )

            # Get fast_weights
            fast_weights = self._fast_weights(
                params=fast_weights,
                gradients=grads,
                train_mode=train_mode,
            )

        # ----- Post-update (theta_T) -----
        # Eval and return performance on query set (test_x, test_y)
        post_preds = learner.forward_weights(test_x, fast_weights)
        post_loss = learner.criterion(post_preds, test_y)

        return pre_loss, post_loss, pre_preds, post_preds

    def set_train_mode(self):
        self.baselearner.train()

    def set_val_mode(self):
        self.baselearner.eval()

    def inner_train(self, train_x, train_y, test_x, test_y, rpc_mode=False):
        return self._deploy(train_x, train_y, test_x, test_y, True, self.T)

    def outer_train(self, pre_losses, post_losses):
        mean_pre_losses = pre_losses.mean()
        mean_post_losses = post_losses.mean()

        # Meta-update
        self.optimizer.zero_grad()
        mean_post_losses.backward()

        # Clip gradient (optional)
        if self.grad_clip is not None:
            for p in self.initialization:
                if p.grad is not None:
                    p.grad.data.clamp_(-self.grad_clip, self.grad_clip)

        self.optimizer.step()
        # opt = DistributedOptimizer(self.optimizer, self.initialization, lr=self.lr)
        # opt.step()

        return mean_pre_losses.item(), mean_post_losses.item()

    def evaluate(self, train_x, train_y, test_x, test_y, val_mode=True):
        T = self.T_val if val_mode else self.T_test

        # Compute the test loss after a single gradient update on the support set
        _, _, pre_preds, post_preds = self._deploy(
            train_x,
            train_y,
            test_x,
            test_y,
            False,
            T,
        )

        # Turn one-hot predictions into class preds
        pre_train_y_hat = torch.argmax(pre_preds, dim=1)
        pre_acc = accuracy(pre_train_y_hat, train_y)
        post_test_y_hat = torch.argmax(post_preds, dim=1)
        post_acc = accuracy(post_test_y_hat, test_y)
        return pre_acc, post_acc

    def train(self, train_x, train_y, test_x, test_y):
        pass

    def dump_state(self):
        """Return the state of the meta-learner

        Returns
        ----------
        initialization
            Initialization parameters
        """
        return [p.clone().detach().to(device) for p in self.initialization]

    def load_state(self, state):
        """Load the given state into the meta-learner

        Parameters
        ----------
        state : initialization
            Initialization parameters
        """

        self.initialization = [p.clone() for p in state]
        for p in self.initialization:
            p.requires_grad = True

    def to(self, device):
        """to device"""
        self.baselearner = self.baselearner.to(device)
        self.initialization = [p.to(device) for p in self.initialization]
