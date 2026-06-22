import torch
from algos.utils import get_loss_n_preds, put_on_device

class MAMLPostHocExplainer:
    def __init__(self, maml_algo):
        """
        Initialize the Post-hoc explainer.
        Takes an already trained MAML instance as input.
        """
        self.algo = maml_algo 
        self.device = maml_algo.device
        self.base_lr = maml_algo.base_lr
        self.baselearner = maml_algo.baselearner
        
        # Extract the current meta-initialization weights (theta_0)
        self.theta_0 = [p.clone().detach().requires_grad_(True) for p in maml_model.theta_0]

    def get_feature_saliency(self, sup_x, sup_y, que_x, que_y, T_test=None):
        """
        Computes the Feature Saliency Map for the support set features (x_j) post-hoc.
        This function uses standard autograd to backpropagate through the unrolled optimization steps.
        """
        T = T_test if T_test is not None else self.algo.T_test
        sup_x, sup_y, que_x, que_y = put_on_device(self.device, [sup_x, sup_y, que_x, que_y])

        # Enable gradient tracking for the support set inputs. 
        # This makes sup_x a leaf node in the computation graph.
        sup_x = sup_x.clone().detach().requires_grad_(True)

        # Initialize fast weights with meta-parameters
        fast_weights = [p.clone() for p in self.theta_0]

        # 1. Unroll the inner-loop adaptation (Forward Trajectory)
        for _ in range(T):
            # Calculate support loss
            loss, _ = get_loss_n_preds(fast_weights, self.baselearner, sup_x, sup_y)
            
            # Compute gradients w.r.t fast_weights.
            # create_graph=True is CRITICAL here to allow backpropagation through the update step later,
            # effectively capturing the mixed derivatives (d2L / dW dx).
            grads = torch.autograd.grad(loss, fast_weights, create_graph=True)
            
            # Update fast weights
            fast_weights = [(w - self.base_lr * g) for w, g in zip(fast_weights, grads)]

        # 2. Calculate query loss using the final adapted weights
        que_loss, _ = get_loss_n_preds(fast_weights, self.baselearner, que_x, que_y)
        
        # 3. Compute the gradient of the query loss with respect to the support set inputs.
        # This triggers the double-backward pass (Vector-Jacobian Product) automatically.
        saliency_map = torch.autograd.grad(que_loss, sup_x)[0]
        
        # Return the negative gradient.
        # Since Delta M = L_Q(pre) - L_Q(post), d(Delta M)/dx = -d(L_Q(post))/dx
        return -saliency_map

    def extract_adjoint_trajectory(self, sup_x, sup_y, que_x, que_y, T):
        """
        Explicitly computes and returns the parameter trajectory phi^(m), 
        the forward gradients g^(m), and the Adjoint states lambda^(m) at EACH STEP.
        
        Note: This expects a SINGLE task as input.
        """
        sup_x, sup_y, que_x, que_y = put_on_device(self.device, [sup_x, sup_y, que_x, que_y])
        
        phis = [ [p.clone().requires_grad_(True) for p in self.theta_0] ]
        forward_grads = []
        
        # 1. FORWARD PASS: Unroll and save the entire trajectory
        for m in range(T):
            loss, _ = get_loss_n_preds(phis[-1], self.baselearner, sup_x, sup_y)
            
            # create_graph=True is needed to track higher-order derivatives
            grads = torch.autograd.grad(loss, phis[-1], create_graph=True)
            forward_grads.append(grads)
            
            phi_next = [(w - self.base_lr * g) for w, g in zip(phis[-1], grads)]
            phis.append(phi_next)
            
        # 2. BASELINE LAMBDA AT STEP K
        que_loss, _ = get_loss_n_preds(phis[-1], self.baselearner, que_x, que_y)
        
        # lambda^(K) = grad(L_Q, phi^(K))
        # retain_graph=True so we can compute HVP going backwards
        lam_K = torch.autograd.grad(que_loss, phis[-1], retain_graph=True)
        
        lambdas = [lam_K]
        
        # 3. BACKWARD ADJOINT LOOP: Compute lambda^(m) backwards to 1
        # Using Pearlmutter's trick (Hessian-Vector Product) to avoid constructing the full Hessian matrix
        current_lam = lam_K
        
        for m in reversed(range(T)):
            # We need to compute: HVP = H^(m) * lambda^(m+1)
            # In PyTorch: H * v = grad( dot(grad_L_S, v) )
            
            # grad_S is the gradient at step m (saved during forward pass)
            grad_S = forward_grads[m] 
            
            # Detach lambda so it acts as a constant vector 'v' in the HVP computation
            v = [l.detach() for l in current_lam]
            
            # Compute dot product: sum_i (g_i * v_i)
            dot_product = sum(torch.sum(g * v_i) for g, v_i in zip(grad_S, v))
            
            # The gradient of this dot product w.r.t the parameters is exactly the HVP
            hvp = torch.autograd.grad(dot_product, phis[m], retain_graph=True)
            
            # lambda^(m) = lambda^(m+1) - alpha * HVP
            lam_prev = [(l - self.base_lr * h) for l, h in zip(current_lam, hvp)]
            
            # Insert at the beginning of the list (since we are traversing backwards)
            lambdas.insert(0, lam_prev)
            current_lam = lam_prev
            
        return {
            "trajectory_phi": phis,           # [phi^0, phi^1, ..., phi^K]
            "forward_grads": forward_grads,   # [g^0, g^1, ..., g^{K-1}]
            "adjoint_lambdas": lambdas        # [lambda^0, lambda^1, ..., lambda^K]
        }
