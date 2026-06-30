import math
from collections import defaultdict
import torch
from torch.optim.optimizer import Optimizer
import warnings

# FSDP support (instead of DeepSpeed)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

warnings.filterwarnings('ignore')




class AdamW_null_space(Optimizer):
    r"""Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, svd=True, thres=1e-3,num_eigen=100, weight_decay=0, amsgrad=False, block_update = False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, svd=svd,
                        num_eigen=num_eigen,
                        thres=thres)
        super(AdamW_null_space, self).__init__(params, defaults)

        self.eigens = {}
        self.transforms = {}
        self.param_to_name = {}
        self.param_to_shape = {}
        self.param_to_feature_indices = {}
        self.soft_projection_alpha = 0.0

        self.should_calculate_null_space_projection = False
        self.fea_in = defaultdict(dict)

        # print('self.param_groups = ')
        # print(self.param_groups)

    def __setstate__(self, state):
        super(AdamW_null_space, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('svd', False)


    # 只有在step里面，gradient才是可见的
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            svd = group['svd']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead')

                update = self.get_update(group, grad, p)
                
                # 这是adamw优化器和adam的不同之处，计算出来梯度的更新之后再加上对于参数的正则化，这样对于梯度的放缩不会影响正则化
                # 由于我们需要对Delta_W做正交投影，所以这里将正则化加到参数更新量之后，随后再做投影，保证整个Delta_W位于input covariance的零空间中
                # 注意这里的weight_decay需要跟lr相乘
                if group['weight_decay'] != 0:
                    update.data.add_( - group['lr'] * group['weight_decay'] * p.data)
                
                update_ = self.Null_space_core(update, svd, p)
                
                p.data.add_(update_)
        
        return loss
    
    def Null_space_core(self, update, svd, p):
        return self._Null_space_core_single_block(update, svd, p)

    # TODO: Precompute grad -> then optimizer_step
    def _Null_space_core_single_block(self, update, svd, p):
        if svd and len(self.transforms) > 0 and p in self.transforms:
            transform_obj = self.transforms[p]
            if isinstance(transform_obj, dict):
                transform = transform_obj["transform"]
                feature_indices = transform_obj.get("feature_indices")
                input_dim = int(transform_obj.get("input_dim", transform.size(0)))
            else:
                transform = transform_obj
                feature_indices = None
                input_dim = int(transform.size(0))
            proj_dtype = transform.dtype

            def _project_rows(flat_tensor):
                if feature_indices is None:
                    projected = torch.mm(flat_tensor.to(proj_dtype), transform)
                    return projected.to(update.dtype)
                idx = feature_indices.to(device=flat_tensor.device, dtype=torch.long)
                if idx.numel() == 0:
                    return flat_tensor
                selected = flat_tensor.index_select(1, idx).to(proj_dtype)
                projected_selected = torch.mm(selected, transform).to(update.dtype)
                projected_full = flat_tensor.clone()
                projected_full.index_copy_(1, idx, projected_selected)
                return projected_full

            if update.ndim == 4:
                # Preserve the existing conv-style fallback.
                flat_update = update.view(update.size(0), -1)
                if flat_update.size(1) == input_dim:
                    _update = _project_rows(flat_update).view_as(update)
                else:
                    _update = update
            elif update.ndim >= 2:
                if update.size(-1) == input_dim:
                    flat_update = update.reshape(-1, input_dim)
                    _update = _project_rows(flat_update).view_as(update)
                else:
                    _update = update
            elif update.ndim == 1:
                # FSDP may expose a flattened local shard at optimizer step time.
                # If the shard boundary still aligns with the input dimension, project row-wise.
                if update.numel() % input_dim == 0:
                    flat_update = update.view(-1, input_dim)
                    _update = _project_rows(flat_update).reshape_as(update)
                else:
                    param_name = self.param_to_name.get(p, "<unknown>")
                    logical_shape = self.param_to_shape.get(p, None)
                    transform_shape = tuple(transform.shape)
                    remainder = int(update.numel() % input_dim)
                    _update = update
            else:
                _update = update
        else:
            _update = update

        alpha = float(getattr(self, "soft_projection_alpha", 0.0) or 0.0)
        alpha = min(max(alpha, 0.0), 1.0)
        if alpha > 0.0 and not torch.equal(_update, update):
            _update = alpha * update + (1.0 - alpha) * _update
        
        return _update
    
    def get_transforms(self):
        for group in self.param_groups:
            svd = group['svd']
            if svd is False:
                continue

            thres = group['thres']
            for p in self.eigens.keys():
                if p not in self.eigens:
                    continue

                fsdp_module = None
                if hasattr(p, '_fsdp_wrapped_module'):
                    fsdp_module = p._fsdp_wrapped_module
                elif hasattr(p, '_handle') and hasattr(p._handle, '_fsdp_wrapped_module'):
                    fsdp_module = p._handle._fsdp_wrapped_module

                if fsdp_module is not None and isinstance(fsdp_module, FSDP):
                    with FSDP.summon_full_params(fsdp_module, writeback=False, recurse=False):
                        self._compute_transforms_for_param(p, thres)
                else:
                    self._compute_transforms_for_param(p, thres)
        # Projection matrices are kept for future optimizer steps.
        # Eigen buffers are only needed during transform construction and can be released now.
        self.eigens = {}

    def _compute_transforms_for_param(self, p, thres):
        """Compute a single null-space projection matrix for a parameter."""
        eig_res = self.eigens.get(p)
        if eig_res is None:
            return

        ev = eig_res["eigen_value"]
        evec = eig_res["eigen_vector"]

        ind = ev <= ev.max() * thres
        basis = evec[:, ind]
        transform = torch.mm(basis, basis.t())

        print('reserving basis {}/{}; cond: {}, radio:{}'.format(
            ind.sum(), ev.shape[0],
            ev[-1] / (ev[0] + 1e-12),
            ev[ind].sum() / ev.sum()
        ))

        norm = torch.norm(transform)
        if norm > 0:
            transform = transform / norm
        feature_indices = eig_res.get("feature_indices")
        input_dim = int(eig_res.get("input_dim", transform.size(0)))
        if feature_indices is not None and int(feature_indices.numel()) < input_dim:
            self.transforms[p] = {
                "transform": transform.detach(),
                "feature_indices": feature_indices.detach().cpu(),
                "input_dim": input_dim,
            }
        else:
            self.transforms[p] = transform.detach()


    def get_eigens(self, fea_in, to_be_updated_modules, feature_indices_by_module=None):
        self.eigens = {}
        self.param_to_name = {}
        self.param_to_shape = {}
        self.param_to_feature_indices = {}
        feature_indices_by_module = feature_indices_by_module or {}
        for group in self.param_groups:
            svd = group['svd']
            if svd is False:
                continue

            for name, cov in fea_in.items():
                module = to_be_updated_modules.get(name)
                if module is None:
                    continue
                module_weight = getattr(module, "weight", None)
                if module_weight is None:
                    continue

                param = None
                for p in group['params']:
                    if p is module_weight or (hasattr(p, 'data') and p.data.data_ptr() == module_weight.data.data_ptr()):
                        param = p
                        break
                
                if param is None:
                    print(f"[WARN] Cannot find param in param_groups for module {name}")
                    continue
                self.param_to_shape[param] = tuple(module_weight.shape)

                fsdp_module = None
                if hasattr(param, '_fsdp_wrapped_module'):
                    fsdp_module = param._fsdp_wrapped_module
                elif hasattr(param, '_handle') and hasattr(param._handle, '_fsdp_wrapped_module'):
                    fsdp_module = param._handle._fsdp_wrapped_module

                if fsdp_module is not None and isinstance(fsdp_module, FSDP):
                    with FSDP.summon_full_params(fsdp_module, writeback=False, recurse=False):
                        self._compute_eigens_for_param(param, name, cov, feature_indices_by_module.get(name))
                else:
                    self._compute_eigens_for_param(param, name, cov, feature_indices_by_module.get(name))
    
    def _compute_eigens_for_param(self, param, name, cov, feature_indices=None):
        """Compute eigenvalues/eigenvectors for a single parameter covariance."""
        self.param_to_name[param] = name
        cov_device = cov.device
        cov = cov.detach().float()
        input_dim = int(cov.size(0))
        selected_feature_indices = None
        if feature_indices is not None:
            if not torch.is_tensor(feature_indices):
                feature_indices = torch.tensor(feature_indices, dtype=torch.long)
            else:
                feature_indices = feature_indices.detach().cpu().to(dtype=torch.long)
            if feature_indices.numel() > 0 and int(feature_indices.numel()) < input_dim:
                selected_feature_indices = feature_indices.unique(sorted=True)
                print(
                    f"[feature] module={name} selecting {selected_feature_indices.numel()}/{input_dim} input dims for projection"
                )
        self.param_to_feature_indices[param] = selected_feature_indices
        cov_cpu = cov.cpu()
        if selected_feature_indices is not None:
            cov_cpu = cov_cpu.index_select(0, selected_feature_indices).index_select(1, selected_feature_indices)
        cov_cpu = 0.5 * (cov_cpu + cov_cpu.T)
        eps = 1e-6 * torch.eye(cov_cpu.size(0), device=cov_cpu.device, dtype=cov_cpu.dtype)
        cov_cpu = cov_cpu + eps
        eigval, eigvec = torch.linalg.eigh(cov_cpu)
        eigval = torch.clamp(eigval, min=0.0)
        eigval, eigvec = eigval.to(cov_device), eigvec.to(cov_device)

        print(f"[eig] min={eigval.min():.3e}, max={eigval.max():.3e}, dtype={eigval.dtype}")
        self.eigens[param] = {
            "eigen_value": eigval.detach().to(cov.dtype),
            "eigen_vector": eigvec.detach().to(cov.dtype),
            "feature_indices": None if selected_feature_indices is None else selected_feature_indices.detach().cpu(),
            "input_dim": input_dim,
        }
                    

    # 更新之后的梯度值
    def get_update(self, group, grad, p):
        amsgrad = group['amsgrad']
        state = self.state[p]

        # State initialization
        if len(state) == 0:
            state['step'] = 0
            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(p.data)
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(p.data)
            if amsgrad:
                # Maintains max of all exp. moving avg. of sq. grad. values
                state['max_exp_avg_sq'] = torch.zeros_like(p.data)

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        if amsgrad:
            max_exp_avg_sq = state['max_exp_avg_sq']
        beta1, beta2 = group['betas']

        state['step'] += 1

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(1 - beta1, grad)
        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            # Use the max. for normalizing running avg. of gradient
            denom = max_exp_avg_sq.sqrt().add_(group['eps'])
        else:
            denom = exp_avg_sq.sqrt().add_(group['eps'])

        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        step_size = group['lr'] * \
            math.sqrt(bias_correction2) / bias_correction1
        update = - step_size * exp_avg / denom

        return update.to(p.data.dtype)
