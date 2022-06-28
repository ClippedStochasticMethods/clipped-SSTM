import math

import torch
from torch import Tensor
from torch.optim import Optimizer

from typing import List, Optional

        
def clipped_gradent_descent_step(
        params: List[Tensor], 
        d_p_list: List[Tensor], 
        momentum_buffer_list: List[Optional[Tensor]],
        lr: float,
        momentum: float,
        clipping_type: str,
        clipping_level: float):
    r"""Functional API that performs clipped step for slipped-SGD and clipped-SSTM algorithm 
        computation.
    See :class:`clipped_SGD` or class:`clipped_SSTM` for details.
    """
    grad_norm = 0.0
    if clipping_type == 'norm':
        for i in range(len(params)):
            grad_norm += d_p_list[i].norm() ** 2
        grad_norm = grad_norm ** 0.5
    
    for i, param in enumerate(params):
        d_p = d_p_list[i]

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1) # no dampening

            d_p = buf
                
        if clipping_type == 'no_clip':
            param.add_(d_p, alpha=-lr)
        elif clipping_type == 'norm':
            alpha = min(1, clipping_level / grad_norm)
            param.add_(d_p, alpha=-lr*alpha)
        elif clipping_type == 'layer_wise':
            alpha = min(1, clipping_level / d_p.norm())
            param.add_(d_p, alpha=-lr*alpha)
        elif clipping_type == 'coordinate_wise':
            eps = 1e-8
            alpha = torch.clip(clipping_level / (torch.abs(d_p) + eps), min=0, max=1)
            param.add_(-lr * alpha * d_p)


class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""
    def __repr__(self):
        return "<required parameter>"

required = _RequiredParameter()


class _DependingParameter(object):
    """Singleton class representing a parameter that depends on other for an Optimizer."""
    def __init__(self, other_parameter_name):
        self.other_parameter_name = other_parameter_name

    def __repr__(self):
        return "<depends on {}>".format(self.other_parameter_name)

depending = _DependingParameter


class clipped_SGD(Optimizer):
    r"""Implements clipped version of stochastic gradient descent
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        clipping_type (string, optional): type of clipping to use: 'norm'|'layer_wise'|'coordinate_wise'.
            'no_clip': no clipping, standart sgd;
            'norm': standard clipping, \min\{1,\lambda / \|\nabla f(x)\|\} \nabla f(x);
            'layer_wise': standard clipping but it is calculated for each layer independently;
            'coordinate_wise': coordinate wise clipping, \min\{1_n, \lambda / \nabla f(x)\} \nabla f(x) 
            where all operations are coordinate wise. (Default: 'norm')
        clipping_level (float, optional): level of clipping \lambda (see clipping_type for more information). 
            Default value depends on clipping_type: 
            for clipping_type='norm' default clipping_level=1
            for clipping_type='layer_wise' default clipping_level=1
            for clipping_type='coordinate_wise' default clipping_level=0.1
    Example:
        >>> optimizer = torch.optim.clipped_SGD(model.parameters(), lr=0.01, 
                                                clipping_type='layer_wise', clipping_level=10)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(
        self, params, 
        lr=required, 
        momentum=0, 
        clipping_type='norm', clipping_level=depending('clipping_type')
    ):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        type_to_default_level = {
            'no_clip': 0.0,
            'norm': 1.0,
            'layer_wise': 0.3,
            'coordinate_wise': 0.1
        }
        if clipping_type not in type_to_default_level:
            raise ValueError("Invalid clipping type: {}, possible types are {}".\
                             format(lr, list(type_to_default_level.keys())))
        if not isinstance(clipping_level, depending) and clipping_level < 0.0:
            raise ValueError("Invalid clipping level: {}".format(clipping_level))
        if isinstance(clipping_level, depending):
            clipping_level = type_to_default_level[clipping_type]
        defaults = dict(
            lr=lr, 
            momentum=momentum, 
            clipping_type=clipping_type, clipping_level=clipping_level
        )
        super(clipped_SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(clipped_SGD, self).__setstate__(state)

    @torch.no_grad() # sets all requires_grad flags to False
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            lr = group['lr']
            momentum = group['momentum']
            clipping_type = group['clipping_type']
            clipping_level = group['clipping_level']

            for p in group['params']:
                if p.grad is not None:
                    d_p_list.append(p.grad)
                    params_with_grad.append(p)
                    
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            # update parameters
            clipped_gradent_descent_step(
                params_with_grad, 
                d_p_list, 
                momentum_buffer_list,
                lr, 
                momentum,
                clipping_type, 
                clipping_level
            )

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss


class clipped_SSTM(Optimizer):
    r"""Implements Clipped Stochastic Similar Triangles Method
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate (stepsize parameter a in paper, inverse to real lr)
        L (float): Lipschitz constant
        clipping_type (string, optional): type of clipping to use: 'norm'|'layer_wise'|'coordinate_wise'.
            'no_clip': no clipping, standart sgd;
            'norm': standard clipping, \min\{1,\lambda / \|\nabla f(x)\|\} \nabla f(x);
            'layer_wise': standard clipping but it is calculated for each layer independently;
            'coordinate_wise': coordinate wise clipping, \min\{1_n, \lambda / \nabla f(x)\} \nabla f(x) 
            where all operations are coordinate wise. (Default: 'norm')
        clipping_level (float, optional): level of clipping \lambda (see clipping_type for more information). 
            In this variant the clipping level changes, and clipping_level is used to calculate it
            Default value depends on clipping_type: 
            for clipping_type='norm' default clipping_level=1.0
            for clipping_type='layer_wise' default clipping_level=0.3
            for clipping_type='coordinate_wise' default clipping_level=0.1
        nu (int, optional): smoothness, must be in [0,1]
        a_k_ratio_upper_bound (float, optional): maximum A_k / A_{k+1} ratio, must be in [0,1] 
            At big steps, A_k / A_{k+1} ratio tends to 1, thus the method becomes too conservative. 
            If a_k_ratio_upper_bound < A_k / A_{k+1} we manually set 
            A_k = a_k_ratio_upper_bound * A_{k+1}, see code for details 
        clipping_iter_start (int, optional): must be > 0. If specified, \nu > 0 and clipping_type=='norm' then 
        	clipping_level will be chosen to ensure that clipping starts at this iteration of method
        	(we can find clipping_level B from B / (k^{2\nu/(1+\nu)}\alpha_0) = 1 when \nu > 0) 
    Example:
        >>> optimizer = clipped_SSTM(model.parameters(), lr=0.01, L=10,
                                     clipping_type='norm', clipping_level=10)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(
        self, params, 
        lr=required, L=required, 
        clipping_type='norm', clipping_level=depending('clipping_type'), 
        nu=1, a_k_ratio_upper_bound=1.0, clipping_iter_start=None
    ):
        if lr is not required and lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if L is not required and L < 0.0:
            raise ValueError("Invalid Lipschitz constant: {}".format(lr))
        
        type_to_default_level = {
            'no_clip': 0.0,
            'norm': 1.0,
            'layer_wise': 0.3,
            'coordinate_wise': 0.1
        }
        if clipping_type not in type_to_default_level:
            raise ValueError("Invalid clipping type: {}, possible types are {}".\
                             format(clipping_type, list(type_to_default_level.keys())))
        if not isinstance(clipping_level, depending) and clipping_level < 0.0:
            raise ValueError("Invalid clipping level: {}".format(clipping_level))
        if isinstance(clipping_level, depending):
            clipping_level = type_to_default_level[clipping_type]
        if nu < 0.0 or nu > 1.0:
            raise ValueError("Invalid nu: {}".format(nu))
        if a_k_ratio_upper_bound <= 0.0 or a_k_ratio_upper_bound > 1.0:
            raise ValueError("Invalid a_k_ratio_upper_bound: {}".format(a_k_ratio_upper_bound))
        if clipping_iter_start is not None:
            if not isinstance(clipping_iter_start, int) or clipping_iter_start <= 0:
	            raise ValueError("Invalid clipping_iter_start: {}, should be positive integer")
            if (nu > 0 and clipping_type == 'norm'):
                a = 1 / lr
                # clipping_level / ( 1 / (2 * a * L) * (k + 1) ** (2 * nu / (1 + nu))) = 1
                clipping_level = 1 / (2 * a * L) * (clipping_iter_start + 1) ** (2 * nu / (1 + nu))
                # print(clipping_level)
            elif (nu < 1e-4):
                a = 1 / lr
                clipping_level = clipping_level / (2 * a * L)

        defaults = dict(
            lr=lr, L=L, 
            clipping_type=clipping_type, clipping_level=clipping_level, 
            nu=nu, a_k_ratio_upper_bound=a_k_ratio_upper_bound,
            state=dict()
        )
        super(clipped_SSTM, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(clipped_SSTM, self).__setstate__(state)

    @torch.no_grad() # sets all requires_grad flags to False
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            a = 1 / group['lr']
            L = group['L']
            clipping_type = group['clipping_type']
            clipping_level = group['clipping_level']
            nu = group["nu"]
            a_k_ratio_upper_bound = group["a_k_ratio_upper_bound"]

            state = group['state']
            # lazy state initialization
            if len(state) == 0:
                state['k'] = 0
                state['alpha_k_1'] = 0
                state['lambda_k_1'] = 0
                state['A_k'] = 0
                state['A_k_1'] = 0

                state['y_k'] = []
                state['z_k'] = []
                for p in group['params']:
                    if p.grad is not None:
                        state['y_k'].append(p.detach().clone())
                        state['z_k'].append(p.detach().clone())
                
            k = state['k']
            alpha_k_1 = state['alpha_k_1']
            lambda_k_1 = state['lambda_k_1']
            A_k = state['A_k']
            A_k_1 = state['A_k_1']
            y_k = [y.detach().clone() for y in state['y_k']]
            z_k = [z.detach().clone() for z in state['z_k']]

            if k > 0:
                for p in group['params']:
                    if p.grad is not None:
                        d_p_list.append(p.grad.data)

                # update z_{k+1}
                clipped_gradent_descent_step(
                    z_k, 
                    d_p_list, 
                    None, # no momentum history
                    alpha_k_1, 
                    0, # no momentum, thus 0
                    clipping_type, 
                    lambda_k_1
                )

                # update y_{k+1}
                i = 0
                for p in group['params']:
                    if p.grad is not None:
                        y_k[i].data = (A_k * y_k[i].data + alpha_k_1 * z_k[i].data) / A_k_1
                        i += 1

            # k_1 means "k + 1", so alpha_k_1 means \alpha_{k+1}
            alpha_k_1 = 1 / (2 * a * L) * (k + 1) ** (2 * nu / (1 + nu))

            A_k = state['A_k_1']
            A_k_1 = A_k + alpha_k_1

            # apply upper bound on A_k / A_{k+1} ratio
            if a_k_ratio_upper_bound < 1.0:
                ratio_mul_factor = 1.0 / (1.0 - a_k_ratio_upper_bound)
                if A_k > ratio_mul_factor * alpha_k_1:
                    A_k = (ratio_mul_factor - 1.0) * alpha_k_1
                    A_k_1 = ratio_mul_factor * alpha_k_1

            lambda_k_1 = clipping_level / alpha_k_1
            # lambda_k_1 = clipping_level
            
            state['y_k'] = y_k
            state['z_k'] = z_k

            # update x_{k+1}
            i = 0
            for p in group['params']:
                if p.grad is not None:
                    p.data = (A_k * state['y_k'][i].data + alpha_k_1 * state['z_k'][i].data) / A_k_1
                    i += 1

            state['k'] += 1
            state['alpha_k_1'] = alpha_k_1
            state['lambda_k_1'] = lambda_k_1
            state['A_k'] = A_k
            state['A_k_1'] = A_k_1

        return loss
