import casadi
import torch
from torch import nn
import numpy as np
from lib import utils


class GradWrapper(nn.Module):
    def __init__(self, _model, n_in):
        super().__init__()
        self.model = _model
        self.n_in = n_in

    def forward(self, *args):
        n_in = self.n_in
        torch_out = self.model(*args[-n_in:])
        ret = []
        for ind, arg in enumerate(args[-n_in:]):
            # if ind == len(args):
            #     ret.append(torch.autograd.grad(torch_out, arg, _adj_seed,
            #                                    retain_graph=False)[0])
            # else:
                ret.append(torch.autograd.grad(torch_out, arg, args[:-n_in],
                                               retain_graph=True,
                                               create_graph=True)[0])
        return ret


class JacWrapper(nn.Module):
    def __init__(self, _model, input_shapes, output_shapes):
        super().__init__()
        self.model = _model
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes
        self.input_size = np.sum(
            [np.product(input_shape) for input_shape in self.input_shapes])
        self.output_size = np.sum(
            [np.product(output_shapes) for output_shapes in self.output_shapes])

    def get_jacs(self, *args):
        return [torch.autograd.functional.jacobian(
            lambda *mini_args: self.model(*mini_args)[i],
            args[:len(self.input_shapes)], vectorize=True)
            for i in range(len(self.output_shapes))]

    # @profile
    def forward(self, *args):
        jacs = self.get_jacs(*args)

        jacs = [utils.reshape_fortran(jacs[i][j],
                                      (np.product(self.output_shapes[i]),
                                       np.product(self.input_shapes[j])))
                for j in range(len(self.input_shapes))
                for i in range(len(self.output_shapes))]
        jacs = [torch.concat(jac_list, dim=0) for jac_list in
                [jacs[k:k + len(self.output_shapes)]
                 for k in range(0, len(jacs), len(self.output_shapes))]]
        jacs = torch.concat(jacs, dim=1)
        return [jacs]


class JacWrapperV2Helper(nn.Module):
    def __init__(self, _model):
        super().__init__()
        self.model = _model

    def forward(self, state_k, input_k):
        return self.model(state_k, input_k)[0]


class JacWrapperV2(nn.Module):
    def __init__(self, _model, input_shapes, output_shapes):
        super().__init__()
        self.model = JacWrapperV2Helper(_model)
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes
        self.input_size = np.sum(
            [np.product(input_shape) for input_shape in self.input_shapes])
        self.output_size = np.sum(
            [np.product(output_shapes) for output_shapes in self.output_shapes])

    # @profile
    def forward(self, state_k, input_k, state_kp1):
        jacs = [
            torch.autograd.functional.jacobian(self.model, (state_k, input_k), strategy="reverse-mode")]

        jacs = [utils.reshape_fortran(jacs[0][j],
                                      (np.product(self.output_shapes[0]),
                                       np.product(self.input_shapes[j])))
                for j in range(len(self.input_shapes))]
        # jacs = [torch.concat(jacs, dim=0)]
        jacs = torch.concat(jacs, dim=1)
        return [jacs]


class TorchEvaluator(casadi.Callback):
    def __init__(self, _model, input_shapes, output_shapes, device, opts=None):
        """
      t_in: list of inputs (tensorflow placeholders)
      t_out: list of outputs (tensors dependeant on those placeholders)
    """
        casadi.Callback.__init__(self)
        self.model = _model
        self.input_shapes = input_shapes
        self.grad_model = GradWrapper(self.model, len(self.input_shapes))
        self.output_shapes = output_shapes
        self.device = device
        if opts is None:
            opts = {}
        self.construct("TorchEvaluator", opts)
        self.refs = []
        self.args = None

    def get_n_in(self): return len(self.input_shapes)

    def get_n_out(self): return len(self.output_shapes)

    def get_sparsity_in(self, i):
        return casadi.Sparsity.dense(*self.input_shapes[i])

    def get_sparsity_out(self, i):
        return casadi.Sparsity.dense(*self.output_shapes[i])

    def eval(self, args):
        # Associate each tensorflow input with the
        # numerical argument passed by CasADi
        # Evaluate the tensorflow expressions
        # if self.args is None:
        self.args = list(
            torch.tensor(arg.toarray(), requires_grad=True, device=self.device)
            for arg in args)
        # else:
        #     for i, arg in enumerate(args):
        #         self.args[i][:] = torch.from_numpy(arg.toarray())
        rets = self.model(*tuple(self.args))
        rets = [casadi.DM(ret.cpu().detach().numpy()) for ret in rets]
        return rets

    # Vanilla tensorflow offers just the reverse mode AD
    def has_reverse(self, nadj): return nadj == 1

    def get_reverse(self, nadj, name, inames, onames, opts):
        # Construct the reverse tensorflow graph through 'gradients'
        grad_model = self.grad_model
        callback = TorchEvaluator(grad_model,
                                  self.output_shapes + self.input_shapes,
                                  self.input_shapes, self.device)
        # Make sure you keep a reference to it
        self.refs.append(callback)

        # Package it in the nominal_in+nominal_out+adj_seed form
        # that CasADi expects
        nominal_in = self.mx_in()
        nominal_out = self.mx_out()
        adj_seed = self.mx_out()
        return casadi.Function(name, nominal_in + nominal_out + adj_seed,
                               callback.call(adj_seed + nominal_in), inames,
                               onames)


# TODO: (torch_casadi) add hessian
class TorchEvaluatorV2(casadi.Callback):
    def __init__(self, _model, input_shapes, output_shapes, device, opts=None):
        """
      t_in: list of inputs (tensorflow placeholders)
      t_out: list of outputs (tensors dependeant on those placeholders)
    """
        casadi.Callback.__init__(self)
        self.model = _model
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes
        self.jac_model = JacWrapper(self.model, input_shapes, output_shapes)
        self.grad_model = GradWrapper(self.model, len(self.input_shapes))
        self.jac_callback = None
        self.device = device
        if opts is None:
            opts = {}
        self.construct("TorchEvaluator", opts)
        self.refs = []
        self.args = None

    def get_n_in(self): return len(self.input_shapes)

    def get_n_out(self): return len(self.output_shapes)

    # TODO: get actual sparsity
    def get_sparsity_in(self, i):
        return casadi.Sparsity.dense(*self.input_shapes[i])

    def get_sparsity_out(self, i):
        return casadi.Sparsity.dense(*self.output_shapes[i])

    def eval(self, args):
        # Associate each tensorflow input with the numerical
        # argument passed by CasADi
        # Evaluate the tensorflow expressions
        self.args = list(
            torch.tensor(arg.toarray(), requires_grad=True, device=self.device)
            for arg in args)
        rets = self.model(*tuple(self.args))
        rets = [casadi.DM(ret.cpu().detach().numpy()) for ret in rets]
        return rets

    # Vanilla tensorflow offers just the reverse mode AD
    def has_jacobian(self): return True

    # @profile
    def get_jacobian(self, name, inames, onames, opts):
        # Construct the reverse tensorflow graph through 'gradients'
        jac_model = self.jac_model
        input_size = np.sum(
            [np.product(input_shape) for input_shape in self.input_shapes])
        output_size = np.sum(
            [np.product(output_shapes) for output_shapes in self.output_shapes])
        self.jac_callback = TorchEvaluator(jac_model,
                                           self.input_shapes + self.output_shapes,
                                           [[output_size, input_size]],
                                           self.device)
        return self.jac_callback

        # Vanilla tensorflow offers just the reverse mode AD

    def has_reverse(self, nadj): return nadj == 1

    def get_reverse(self, nadj, name, inames, onames, opts):
        # Construct the reverse tensorflow graph through 'gradients'
        grad_model = self.grad_model
        callback = TorchEvaluator(grad_model,
                                  self.output_shapes + self.input_shapes,
                                  self.input_shapes,
                                  self.device)
        # Make sure you keep a reference to it
        self.refs.append(callback)

        # Package it in the nominal_in+nominal_out+adj_seed form
        # that CasADi expects
        nominal_in = self.mx_in()
        nominal_out = self.mx_out()
        adj_seed = self.mx_out()
        return casadi.Function(name, nominal_in + nominal_out + adj_seed,
                               callback.call(adj_seed + nominal_in),
                               inames,
                               onames)


class TorchEvaluatorV3(casadi.Callback):
    def __init__(self, _model, input_shapes, output_shapes, device, opts=None):
        """
      t_in: list of inputs (tensorflow placeholders)
      t_out: list of outputs (tensors dependeant on those placeholders)
    """
        casadi.Callback.__init__(self)
        self.model = _model
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes
        self.jac_model = JacWrapper(self.model, input_shapes, output_shapes)
        self.grad_model = GradWrapper(self.model, len(self.input_shapes))

        # needed to keep the callback alive
        self.jac_callback = None
        self.adj_callback = None
        self.hess_callback = None

        self.device = device
        if opts is None:
            opts = {}
        self.construct("TorchEvaluator", opts)
        self.refs = []
        self.args = None

    def get_n_in(self):
        return len(self.input_shapes)

    def get_n_out(self):
        return len(self.output_shapes)

    # TODO: get actual sparsity
    def get_sparsity_in(self, i):
        return casadi.Sparsity.dense(*self.input_shapes[i])

    def get_sparsity_out(self, i):
        return casadi.Sparsity.dense(*self.output_shapes[i])

    def eval(self, args):
        # Associate each tensorflow input with the numerical
        # argument passed by CasADi
        # Evaluate the tensorflow expressions
        self.args = list(
            torch.tensor(arg.toarray(), requires_grad=True, device=self.device)
            for arg in args)
        rets = self.model(*tuple(self.args))
        rets = [casadi.DM(ret.cpu().detach().numpy()) for ret in rets]
        return rets

    # Vanilla tensorflow offers just the reverse mode AD
    def has_jacobian(self):
        return True

    # @profile
    def get_jacobian(self, name, inames, onames, opts):
        # Construct the reverse tensorflow graph through 'gradients'
        jac_model = self.jac_model
        input_size = np.sum(
            [np.product(input_shape) for input_shape in self.input_shapes])
        output_size = np.sum(
            [np.product(output_shapes) for output_shapes in self.output_shapes])
        jac_callback = TorchJacEvaluator(
            self,
            jac_model,
            self.input_shapes + self.output_shapes,
            [[output_size, input_size]],
            self.device)
        return jac_callback

        # Vanilla tensorflow offers just the reverse mode AD

    def has_reverse(self, nadj) -> bool:
        if nadj == 1:
            return True
        else:
            return False

    def get_reverse(self, nadj, name, inames, onames, opts):
        # Construct the reverse tensorflow graph through 'gradients'
        grad_model = self.grad_model
        self.adj_callback = TorchRevEvaluator(
            self,
            grad_model,
            self.output_shapes + self.input_shapes,
            self.input_shapes,
            self.device)

        # Package it in the nominal_in+nominal_out+adj_seed form
        # that CasADi expects
        nominal_in = self.mx_in()
        nominal_out = self.mx_out()
        adj_seed = self.mx_out()
        return casadi.Function(name, nominal_in + nominal_out + adj_seed,
                               self.adj_callback.call(adj_seed + nominal_in),
                               inames,
                               onames)


class TorchJacEvaluator(casadi.Callback):
    def __init__(self,
                 torch_evaluator,
                 _model,
                 input_shapes,
                 output_shapes,
                 device,
                 opts=None):
        """
      t_in: list of inputs (tensorflow placeholders)
      t_out: list of outputs (tensors dependeant on those placeholders)
    """
        casadi.Callback.__init__(self)
        self.torch_evaluator = torch_evaluator
        self.model = _model
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes
        self.device = device
        if opts is None:
            opts = {}
        self.construct("TorchJacEvaluator", opts)
        torch_evaluator.jac_callback = self
        self.args = None

    def get_n_in(self): return len(self.input_shapes)

    def get_n_out(self): return len(self.output_shapes)

    def get_sparsity_in(self, i):
        return casadi.Sparsity.dense(*self.input_shapes[i])

    def get_sparsity_out(self, i):
        return casadi.Sparsity.dense(*self.output_shapes[i])

    def eval(self, args):
        # Associate each tensorflow input with the
        # numerical argument passed by CasADi
        # Evaluate the tensorflow expressions
        # if self.args is None:
        self.args = list(
            torch.tensor(arg.toarray(), requires_grad=True, device=self.device)
            for arg in args)
        # else:
        #     for i, arg in enumerate(args):
        #         self.args[i][:] = torch.from_numpy(arg.toarray())
        rets = self.model(*tuple(self.args))
        rets = [casadi.DM(ret.cpu().detach().numpy()) for ret in rets]
        return rets

    def has_jacobian(self, *args) -> bool:
        return False

    def has_reverse(self, nadj) -> bool:
        return False


class TorchRevEvaluator(casadi.Callback):
    def __init__(self, torch_evaluator: TorchEvaluatorV3,
                 _model, input_shapes, output_shapes, device, opts=None):
        """
      t_in: list of inputs (tensorflow placeholders)
      t_out: list of outputs (tensors dependeant on those placeholders)
    """
        casadi.Callback.__init__(self)
        self.torch_evaluator = torch_evaluator
        self.model = _model
        self.hess_model = JacWrapper(self.model, input_shapes, output_shapes)
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes
        self.device = device
        if opts is None:
            opts = {}
        self.construct("TorchRevEvaluator", opts)
        self.refs = []
        self.args = None

    def get_n_in(self): return len(self.input_shapes)

    def get_n_out(self): return len(self.output_shapes)

    def get_sparsity_in(self, i):
        return casadi.Sparsity.dense(*self.input_shapes[i])

    def get_sparsity_out(self, i):
        return casadi.Sparsity.dense(*self.output_shapes[i])

    def eval(self, args):
        # Associate each tensorflow input with the
        # numerical argument passed by CasADi
        # Evaluate the tensorflow expressions
        # if self.args is None:
        self.args = list(
            torch.tensor(arg.toarray(), requires_grad=True, device=self.device)
            for arg in args)
        # else:
        #     for i, arg in enumerate(args):
        #         self.args[i][:] = torch.from_numpy(arg.toarray())
        rets = self.model(*tuple(self.args))
        rets = [casadi.DM(ret.cpu().detach().numpy()) for ret in rets]
        return rets

    def has_jacobian(self, *args) -> bool:
        return True

    def get_jacobian(self, name, inames, onames, opts):
        # Construct the reverse tensorflow graph through 'gradients'
        hess_model = self.hess_model
        input_size = np.sum(
            [np.product(input_shape) for input_shape in self.input_shapes])
        output_size = np.sum(
            [np.product(output_shapes) for output_shapes in self.output_shapes])
        self.torch_evaluator.hess_callback = TorchHessianEvaluator(
            self.torch_evaluator,
            hess_model,
            self.input_shapes + self.output_shapes,
            [[output_size, input_size]],
            self.device)
        return self.torch_evaluator.hess_callback


class TorchHessianEvaluator(casadi.Callback):
    def __init__(self,
                 torch_evaluator: TorchEvaluator,
                 _model,
                 input_shapes,
                 output_shapes,
                 device,
                 opts=None):
        """
      t_in: list of inputs (tensorflow placeholders)
      t_out: list of outputs (tensors dependeant on those placeholders)
    """
        casadi.Callback.__init__(self)
        self.model = _model
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes
        self.device = device
        if opts is None:
            opts = {}
        self.construct("TorchHessianEvaluator", opts)
        self.args = None

    def get_n_in(self): return len(self.input_shapes)

    def get_n_out(self): return len(self.output_shapes)

    def get_sparsity_in(self, i):
        return casadi.Sparsity.dense(*self.input_shapes[i])

    def get_sparsity_out(self, i):
        return casadi.Sparsity.dense(*self.output_shapes[i])

    def eval(self, args):
        # Associate each tensorflow input with the
        # numerical argument passed by CasADi
        # Evaluate the tensorflow expressions
        # if self.args is None:
        self.args = list(
            torch.tensor(arg.toarray(), requires_grad=True, device=self.device)
            for arg in args)
        # else:
        #     for i, arg in enumerate(args):
        #         self.args[i][:] = torch.from_numpy(arg.toarray())
        rets = self.model(*tuple(self.args))
        rets = [casadi.DM(ret.cpu().detach().numpy()) for ret in rets]
        return rets

    def has_jacobian(self, *args) -> bool:
        return False
    

