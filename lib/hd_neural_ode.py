import torch
from torch import nn
from torch.utils.data import DataLoader
from lib.hd_simulator import Simulator
from lib.hd_dataset import MyDataset
import numpy as np
import random
from lib import utils
from lib import hd_utils
from time import time
import os


random.seed(6)
np.random.seed(3)
torch.manual_seed(5)


class MLP(nn.Module):
    def __init__(self, device, sim_obj: Simulator):
        """

        :param device:
        :param sim_obj:
        """
        super().__init__()
        self.device = device
        self.sim_obj = sim_obj
        nn_dims = self.sim_obj.nn_dims
        self.mlps = nn.ModuleList()
        for _ in range(self.sim_obj.n_time_pts):
            layers = []
            for ind in range(len(nn_dims) - 1):
                if ind != 0:
                    layers.append(nn.Tanh())
                layers.append(nn.Linear(nn_dims[ind], nn_dims[ind + 1],
                                        dtype=torch.float64))
            self.mlps.append(nn.Sequential(*layers))
        self.apply(self.init_weights)

    def forward(self, x, z, u, test=None):
        if x is not None:
            base_shape = list(x.shape)
        elif z is not None:
            base_shape = list(z.shape)
        elif u is not None:
            base_shape = list(u.shape)
        else:
            raise ValueError('All variables (x, z, u) cannot be None'
                             'simultaneously. '
                             'Assign a valid value to at least one of them.')
        base_shape[-2] = 0

        if x is None:
            x = torch.empty(base_shape, device=self.device)
        if z is None:
            z = torch.empty(base_shape, device=self.device)
        if u is None:
            u = torch.empty(base_shape, device=self.device)
        nn_input = torch.cat((x, z, u), dim=-2).squeeze(-1)
        if test is not None:
            np_output = test
            output = torch.asarray(np_output,
                                   dtype=torch.float64).to(self.device)

        else:
            output_list = [mlp(nn_input[:, i, :]) for i, mlp in
                           enumerate(self.mlps)]
            return torch.stack(output_list, dim=1).unsqueeze(-1)
        return output.unsqueeze(-1)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.uniform_(m.weight, -1e-3, 1e-3).double()


class SingleStep(nn.Module):
    def __init__(self, xuyscales, device, sim_obj: Simulator) -> None:
        super(SingleStep, self).__init__()
        self.device = device
        self.sim_obj = sim_obj
        self.mlp = MLP(device, sim_obj)
        self.xuyscales = xuyscales

        ymean, ystd = self.xuyscales['yscale']

        self.xmeans = torch.as_tensor(ymean).unsqueeze(1).to(self.device)
        self.xstds = torch.as_tensor(ystd).unsqueeze(1).to(self.device)

        self.grad_mat = torch.as_tensor(self.sim_obj.grad_mat).to(self.device)
        self.time_dz = torch.as_tensor(self.sim_obj.time_dz,
                                       dtype=torch.complex128).to(self.device)

        self.vec_xmeans = self.xmeans.reshape((-1, self.sim_obj.n_y, 1))
        self.vec_xstds = self.xstds.reshape((-1, self.sim_obj.n_y, 1))
        n = self.sim_obj.n_state_pts
        n_modes = self.sim_obj.n_state_modes
        toeplitz = utils.generate_toeplitz_tensor(
            n, dtype=torch.complex128).to(self.device)
        self.toeplitz = toeplitz[n_modes:-n_modes]

    def _aug_ode_mat(self, xd, z, u, xa, test=None):
        n = self.sim_obj.n_state_pts
        toeplitz = self.toeplitz
        grad_mat = self.grad_mat
        # Get the states before scaling.
        # Compute NN toytion rates.
        phi = hd_utils.torch_stacked_to_cplx(
            self.mlp(None, z, u, test).squeeze(-1))
        adv_term = - 1/n * utils.toeplitz_last_dim(phi, self.toeplitz)
        op = grad_mat @ (grad_mat / self.sim_obj.ld2 + adv_term)
        return op, phi

    def irk(self, xz, u, test):
        n_time_pts = self.sim_obj.n_time_pts
        n_g = self.sim_obj.n_g
        n_p = self.sim_obj.n_p
        n_u = self.sim_obj.n_u
        n_y = self.sim_obj.n_y
        # split into current state and past
        x, z = torch.split(xz, [n_g,
                                n_p * (n_g
                                       + n_u)], dim=1)

        # Construct current state combined with history tensors
        xz_x, xz_u = torch.split(xz, [(n_p + 1) * n_g,
                                      n_p * n_u], dim=1)
        # Stack the past at collocation times into auxiliary batches to
        # handle time-varying state matrix
        stacked_xz_u = xz_u.unsqueeze(1).repeat(1, n_time_pts, 1, 1)

        stacked_xz_x = torch.reshape(
            utils.reshape_fortran(
                torch.reshape(
                    xz_x, (-1, n_time_pts * (n_p + 1), n_y, 1)),
                (xz_x.shape[0], n_time_pts, (n_p + 1), n_y)),
            (-1, n_time_pts, (n_p + 1) * n_y, 1))
        mod_xz = torch.concat((stacked_xz_x,
                               stacked_xz_u), dim=2)

        stacked_u = u.unsqueeze(1).repeat(1, n_time_pts, 1, 1)

        aug_mat, phi = self._aug_ode_mat(
            x, mod_xz, stacked_u, xa=None, test=test)
        x_ = (x[:, :n_y] * self.xstds[:n_y]) + self.xmeans[:n_y]
        mat = (utils.batched_block_diag(aug_mat)
               - torch.kron(torch.flip(self.time_dz[1:, 1:], [0, 1]),
                            torch.eye(n_y, device=self.device)))
        cplx_x = hd_utils.torch_stacked_to_cplx(
            x_.squeeze(-1), True).unsqueeze(-1)
        rhs = cplx_x @ torch.flip(self.time_dz[1:, 0:1], [0]).T
        rhs = utils.reshape_fortran(rhs, (rhs.shape[0],
                                          rhs.shape[1] * rhs.shape[2]))
        sol = torch.linalg.solve(mat, rhs)
        vec_sol = torch.reshape(sol, (sol.shape[0],
                                      n_time_pts,
                                      n_y))
        xplus_ = hd_utils.torch_cplx_to_stacked(
            vec_sol, True).reshape(-1, n_g, 1)
        xplus = (xplus_ - self.xmeans) / self.xstds
        if n_p > 0:
            zp = torch.concat((xz_x[:, :-n_g],
                               u,
                               xz_u[:, :-n_u]), dim=1)
            xzplus = torch.concat((xplus, zp), dim=1)
        else:
            xzplus = xplus
        return xplus, xzplus, phi.unsqueeze(-1)

    def irk_v2(self, xz, u, test):
        raise NotImplementedError

    def resnet(self, xz, u, test):
        n_time_pts = 1
        n_g = self.sim_obj.n_g
        n_p = self.sim_obj.n_p
        n_u = self.sim_obj.n_u
        n_y = self.sim_obj.n_y

        # split into current state and past
        x, z = torch.split(xz, [n_g,
                                n_p * (n_g
                                       + n_u)], dim=1)

        # Construct current state combined with history tensors
        xz_x, xz_u = torch.split(xz, [(n_p + 1) * n_g,
                                      n_p * n_u], dim=1)
        # Stack the past at collocation times into auxiliary batches to
        # handle time-varying state matrix
        stacked_xz_u = xz_u.unsqueeze(1).repeat(1, n_time_pts, 1, 1)

        stacked_xz_x = torch.reshape(
            utils.reshape_fortran(
                torch.reshape(
                    xz_x, (-1, n_time_pts * (n_p + 1), n_y, 1)),
                (xz_x.shape[0], n_time_pts, (n_p + 1), n_y)),
            (-1, n_time_pts, (n_p + 1) * n_y, 1))
        mod_xz = torch.concat((stacked_xz_x,
                               stacked_xz_u), dim=2)

        stacked_u = u.unsqueeze(1).repeat(1, n_time_pts, 1, 1)
        phi = self.mlp(None, mod_xz, stacked_u, test).squeeze(1)
        xplus = x + phi
        if n_p > 0:
            zp = torch.concat((xz_x[:, :-n_g],
                               u,
                               xz_u[:, :-n_u]), dim=1)
            xzplus = torch.concat((xplus, zp), dim=1)
        else:
            xzplus = xplus
        info = (xplus, xzplus, phi)
        return info

    def step(self, xz, u, test):
        return {
            "irk": self.irk,
            'resnet': self.resnet
        }[self.sim_obj.model](xz, u, test)

    def forward(self, xz, u, test=None):
        """ Call function of the hybrid RNN cell.
            Dimension of states: (None, Nx)
            Dimension of input: (None, Nu)
        """
        xplus, xzplus, phi = self.step(xz, u, test)
        return xplus, xzplus, phi


class MultipleStepWrapper(nn.Module):
    def __init__(self, xuyscales, device, sim_obj: Simulator) -> None:
        super(MultipleStepWrapper, self).__init__()
        self.device = device
        self.sim_obj = sim_obj
        self.single_stepper = SingleStep(xuyscales, device, sim_obj)
        self.x0 = torch.nn.Parameter
        self.xuyscales = xuyscales

    def forward(self, xz0, useq, test=None):
        xz = xz0
        hat_xseq = []
        phi_seq = []
        for i in range(useq.shape[-1]):
            xp, xzp, phi = self.single_stepper(xz, useq[:, :, i:i + 1], test)
            hat_xseq += [xp]
            phi_seq += [phi]
            xz = xzp
        hat_xseq = torch.cat(hat_xseq, dim=2)
        phi_seq = torch.cat(phi_seq, dim=-1)
        return hat_xseq, xz, phi_seq


class MeanSquaredFunctionLoss(nn.Module):
    def __init__(self, device, sim_obj: Simulator):
        super(MeanSquaredFunctionLoss, self).__init__()

    def forward(self, output, target):
        error = output - target
        return torch.mean(torch.sum(error**2, axis=1))


class MeanSquaredFunctionLossWithGrad(nn.Module):
    def __init__(self, device, sim_obj: Simulator):
        super(MeanSquaredFunctionLossWithGrad, self).__init__()
        self.oa_iz = torch.as_tensor(
            np.hstack((sim_obj.state_iz[1:-1].flatten(),
                       sim_obj.state_iz[[0, -1]].flatten()))
        ).to(device)
        self.sim_obj = sim_obj
        self.oa_dz = torch.as_tensor(self.sim_obj.oa_state_dz).to(device)
        self.oa_dz2 = torch.as_tensor(self.sim_obj.oa_state_dz2).to(device)

    def forward(self, output, target):
        shape = output.shape
        n_y = self.sim_obj.n_y

        stacked_output = output.reshape(shape[0],
                                        self.sim_obj.n_time_pts,
                                        -1,
                                        shape[-1])
        stacked_target = target.reshape(shape[0],
                                        self.sim_obj.n_time_pts,
                                        -1,
                                        shape[-1])

        se = (stacked_output - stacked_target) ** 2
        ie1 = torch.tensordot(self.oa_iz, se, dims=([0], [2]))

        se = (self.oa_dz.view(1, 1, n_y, -1)
              @ (stacked_output - stacked_target)) ** 2
        ie2 = torch.tensordot(self.oa_iz, se, dims=([0], [2]))

        se = (self.oa_dz2.view(1, 1, n_y, -1)
              @ (stacked_output - stacked_target)) ** 2
        ie3 = torch.tensordot(self.oa_iz, se, dims=([0], [2]))

        return (torch.sum(ie1) + torch.sum(ie2) + torch.sum(ie3)) / shape[-1]


class MeanSquaredFunctionLossWithGradAndTime(nn.Module):
    def __init__(self, device, sim_obj: Simulator):
        super(MeanSquaredFunctionLossWithGradAndTime, self).__init__()
        self.oa_iz = torch.as_tensor(
            np.hstack((sim_obj.state_iz[1:-1].flatten(),
                       sim_obj.state_iz[[0, -1]].flatten()))
        ).to(device)
        self.sim_obj = sim_obj
        self.oa_dz = torch.as_tensor(self.sim_obj.oa_state_dz).to(device)
        self.oa_dz2 = torch.as_tensor(self.sim_obj.oa_state_dz2).to(device)
        self.time_dz = torch.as_tensor(self.sim_obj.time_dz).to(device)
        self.time_dz2 = torch.as_tensor(self.sim_obj.time_dz2).to(device)
        self.ext_time_iz = torch.as_tensor(self.sim_obj.time_iz).to(device)

    def integrate(self, vals):
        se3 = torch.tensordot(self.oa_iz, vals, dims=([0], [2]))
        se4 = torch.tensordot(self.ext_time_iz[1:], se3, dims=([0], [1]))
        return torch.sum(se4)

    def forward(self, output, target):
        shape = output.shape
        n_y = self.sim_obj.n_y

        stacked_output = output.reshape(shape[0],
                                        self.sim_obj.n_time_pts,
                                        -1,
                                        shape[-1])
        stacked_target = target.reshape(shape[0],
                                        self.sim_obj.n_time_pts,
                                        -1,
                                        shape[-1])
        error = torch.flip(stacked_output - stacked_target, dims=[1])
        extra_error_row = torch.concat((torch.zeros_like(error[:, :1, :, :1]),
                                        error[:, -1:, :, :-1]),
                                       dim=3)
        ext_error = torch.concat((extra_error_row, error),
                                 dim=1)
        ext_shape = ext_error.shape
        dedz = self.oa_dz.view(1, 1, n_y, -1) @ ext_error
        dedt = torch.matmul(
            self.time_dz,
            ext_error.view(ext_shape[0],
                           ext_shape[1], -1)).view(ext_shape)
        de2dz2 = self.oa_dz2.view(1, 1, n_y, -1) @ error
        de2dzdt = torch.matmul(
            self.time_dz,
            dedz.view(ext_shape[0],
                      ext_shape[1], -1)).view(ext_shape)
        de2dt2 = torch.matmul(
            self.time_dz2,
            ext_error.view(ext_shape[0],
                           ext_shape[1], -1)).view(ext_shape)

        loss_term_list = [error,
                          dedz[:, 1:],
                          dedt[:, 1:],
                          de2dz2,
                          de2dzdt[:, 1:],
                          de2dt2[:, 1:]
                          ]

        loss = 0
        for loss_term in loss_term_list:
            loss += self.integrate(loss_term ** 2)
        return loss / shape[-1]


class ControlWrapper(nn.Module):
    def __init__(self, ctrl_dt,
                 device,
                 model: MultipleStepWrapper,
                 ) -> None:
        super(ControlWrapper, self).__init__()
        self.device = device
        self.model = model
        self.ctrl_dt = ctrl_dt
        y_mean, y_std = self.model.xuyscales['yscale']
        u_mean, u_std = self.model.xuyscales['uscale']
        y_mean = np.expand_dims(y_mean, axis=1)
        y_std = np.expand_dims(y_std, axis=1)
        u_mean = np.expand_dims(u_mean, axis=1)
        u_std = np.expand_dims(u_std, axis=1)
        xz_mean_list = [y_mean for _ in range(model.sim_obj.n_p + 1)]
        xz_mean_list.extend([u_mean for _ in range(model.sim_obj.n_p)])
        xz_std_list = [y_std for _ in range(model.sim_obj.n_p + 1)]
        xz_std_list.extend([u_std for _ in range(model.sim_obj.n_p)])
        self.xz_mean = torch.as_tensor(np.vstack(xz_mean_list)).to(device)
        self.xz_std = torch.as_tensor(np.vstack(xz_std_list)).to(device)
        self.u_mean = torch.as_tensor(u_mean).to(device)
        self.u_std = torch.as_tensor(u_std).to(device)

    def forward(self, xz, u):
        """ Call function of the hybrid RNN cell.
            Dimension of states: (None, Nx)
            Dimension of input: (None, Nu)
        """

        xz = torch.unsqueeze(xz, 0)
        u = torch.unsqueeze(u, 0)
        norm_xz = utils.norm(xz, self.xz_mean, self.xz_std)
        norm_u = utils.norm(u, self.u_mean, self.u_std)
        norm_x_plus, norm_xz_plus, _ = self.model(norm_xz, norm_u)
        xz_plus = utils.un_norm(norm_xz_plus, self.xz_mean, self.xz_std)

        xz_plus = xz_plus[0, :, :]
        # Return state at the next time-step.
        return [xz_plus]
    
class ForceWrapper(nn.Module):
    def __init__(self, ctrl_dt,
                 device,
                 model: MultipleStepWrapper,
                 ) -> None:
        super(ForceWrapper, self).__init__()
        self.device = device
        self.model = model
        self.ctrl_dt = ctrl_dt
        y_mean, y_std = self.model.xuyscales['yscale']
        u_mean, u_std = self.model.xuyscales['uscale']
        y_mean = np.expand_dims(y_mean, axis=1)
        y_std = np.expand_dims(y_std, axis=1)
        u_mean = np.expand_dims(u_mean, axis=1)
        u_std = np.expand_dims(u_std, axis=1)
        xz_mean_list = [y_mean for _ in range(model.sim_obj.n_p + 1)]
        xz_mean_list.extend([u_mean for _ in range(model.sim_obj.n_p)])
        xz_std_list = [y_std for _ in range(model.sim_obj.n_p + 1)]
        xz_std_list.extend([u_std for _ in range(model.sim_obj.n_p)])
        self.xz_mean = torch.as_tensor(np.vstack(xz_mean_list)).to(device)
        self.xz_std = torch.as_tensor(np.vstack(xz_std_list)).to(device)
        self.u_mean = torch.as_tensor(u_mean).to(device)
        self.u_std = torch.as_tensor(u_std).to(device)

    def forward(self, xz, u):
        """ Call function of the hybrid RNN cell.
            Dimension of states: (None, Nx)
            Dimension of input: (None, Nu)
        """

        xz = torch.unsqueeze(xz, 0)
        u = torch.unsqueeze(u, 0)
        norm_xz = utils.norm(xz, self.xz_mean, self.xz_std)
        norm_u = utils.norm(u, self.u_mean, self.u_std)

        _, _, phi = self.model(norm_xz, norm_u)
        phi0 = torch.real(phi[0, 0, self.model.sim_obj.n_force_modes,:])
        return [phi0[None,:]/self.model.sim_obj.n_state_pts]

class FluxWrapper(nn.Module):
    def __init__(self, ctrl_dt,
                 device,
                 model: MultipleStepWrapper,
                 M:int=None,
                 ) -> None:
        super(FluxWrapper, self).__init__()
        self.device = device
        self.model = model
        self.ctrl_dt = ctrl_dt
        y_mean, y_std = self.model.xuyscales['yscale']
        u_mean, u_std = self.model.xuyscales['uscale']
        y_mean = np.expand_dims(y_mean, axis=1)
        y_std = np.expand_dims(y_std, axis=1)
        u_mean = np.expand_dims(u_mean, axis=1)
        u_std = np.expand_dims(u_std, axis=1)
        xz_mean_list = [y_mean for _ in range(model.sim_obj.n_p + 1)]
        xz_mean_list.extend([u_mean for _ in range(model.sim_obj.n_p)])
        xz_std_list = [y_std for _ in range(model.sim_obj.n_p + 1)]
        xz_std_list.extend([u_std for _ in range(model.sim_obj.n_p)])
        self.y_mean = torch.as_tensor(y_mean).to(device)
        self.y_std = torch.as_tensor(y_std).to(device)
        self.xz_mean = torch.as_tensor(np.vstack(xz_mean_list)).to(device)
        self.xz_std = torch.as_tensor(np.vstack(xz_std_list)).to(device)
        self.u_mean = torch.as_tensor(u_mean).to(device)
        self.u_std = torch.as_tensor(u_std).to(device)
        self.N = int(np.max([self.model.sim_obj.n_state_pts, self.model.sim_obj.n_force_pts]))
        if M is not None:
            self.M = M
        else:
            self.M = int(np.ceil(self.N * 3/2))
            if self.M % 2 == 0:
                self.M += 1

    def forward(self, xz, u):
        """ Call function of the hybrid RNN cell.
            Dimension of states: (None, Nx)
            Dimension of input: (None, Nu)
        """

        xz = torch.unsqueeze(xz, 0)
        u = torch.unsqueeze(u, 0)
        norm_xz = utils.norm(xz, self.xz_mean, self.xz_std)
        norm_u = utils.norm(u, self.u_mean, self.u_std)



        norm_x_plus, _, phi = self.model(norm_xz, norm_u)
        xhat = hd_utils.torch_stacked_to_cplx(
            utils.un_norm(norm_x_plus, self.y_mean, self.y_std)[0, :, 0], scale_k=True)
        # x = hd_utils
        vhat = phi[0,0,:,0]/self.model.sim_obj.n_state_pts
        nx_pad = (self.M - self.model.sim_obj.n_state_pts)//2
        nv_pad = (self.M - self.model.sim_obj.n_force_pts)//2
        padded_xhat = torch.fft.ifftshift(nn.functional.pad(xhat, (nx_pad, nx_pad), 'constant', 0))
        padded_vhat = torch.fft.ifftshift(nn.functional.pad(vhat, (nv_pad, nv_pad), 'constant', 0))
        x = torch.fft.ifft(padded_xhat, norm='forward').real
        v = torch.fft.ifft(padded_vhat, norm='forward').real
        return [(x*v)[:,None]]

def unnorm(x, x_mean, x_std):
    return (x * x_std.view(1, -1, 1)) + x_mean.view(1, -1, 1)


def train(script_name,
          env: Simulator,
          training_info,
          to_backups=False,
          loss_fn=None,):
    """ Get the parameters, training, and validation data."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dirname, basename = utils.get_dir_base(script_name)
    if to_backups:
        dest_name = os.path.join(dirname,
                                 f'backups/'
                                 f'{basename}.pth')
    else:
        dest_name = os.path.join(dirname,
                                 f'build/'
                                 f'{basename}.pth')

    xuyscales = training_info['xuyscales']
    train_data = MyDataset(training_info, env, 'train',
                           oldest_first=False, device=device)
    val_data = MyDataset(training_info, env, 'val',
                         oldest_first=False, device=device)
    test_data = MyDataset(training_info, env, 'test',
                          oldest_first=False, device=device)

    train_dataloader = DataLoader(train_data, batch_size=env.train_batch_size,
                                  num_workers=0)
    val_dataloader = DataLoader(val_data, batch_size=env.val_batch_size,
                                num_workers=0)
    test_dataloader = DataLoader(test_data, batch_size=env.test_batch_size,
                                 num_workers=0)

    model = MultipleStepWrapper(xuyscales, device, env).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-3)
    if loss_fn is None:
        loss_fn = MeanSquaredFunctionLoss(device, env)
    else:
        loss_fn = loss_fn(device, env)

    def train_loop(_dataloader, _model, _loss_fn, _optimizer, _ymean, _ystd):
        for X, y, _ in _dataloader:
            # Compute prediction and loss
            yz0, useq = X
            yz0, useq = yz0.to(device), useq.to(device)
            gpu_y = y.to(device)
            pred, xz, phis = _model(yz0, useq)
            loss = _loss_fn(unnorm(pred, _ymean, _ystd)[:, 1:, :],
                            unnorm(gpu_y, _ymean, _ystd)[:, 1:, :])
            _optimizer.zero_grad()
            loss.backward()
            _optimizer.step()

    def val_loop(_dataloader, _model, _loss_fn, _ymean, _ystd):
        num_batches = len(_dataloader)
        val_loss = 0
        with torch.no_grad():
            for X, y, _ in _dataloader:
                yz0, useq = X
                yz0, useq = yz0.to(device), useq.to(device)
                gpu_y = y.to(device)
                pred, xz, phis = _model(yz0, useq)
                val_loss += _loss_fn(unnorm(pred, _ymean, _ystd)[:, 1:, :],
                                     unnorm(gpu_y, _ymean, _ystd)[:, 1:, :])

        val_loss /= num_batches
        print(f"val_loss: {val_loss:.3e}")
        return val_loss

    def test_loop(_dataloaders, _model, _loss_fn, _ymean, _ystd):
        with torch.no_grad():
            test_losses = []
            preds = []
            phis = []
            for _dataloader in _dataloaders:
                test_loss = 0
                num_batches = len(_dataloader)
                for X, y, _ in _dataloader:
                    yz0, useq = X
                    yz0, useq = yz0.to(device), useq.to(device)
                    gpu_y = y.to(device)
                    pred, xz, phi = _model(yz0, useq)
                    preds.append(pred)
                    phis.append(phi)
                    test_loss += _loss_fn(unnorm(pred, _ymean, _ystd),
                                          unnorm(gpu_y, _ymean, _ystd))
                test_loss /= num_batches
                test_losses.append(test_loss)

        return test_losses, preds, phis

    epochs = 4000
    patience = 200
    counter = 0
    best_loss = np.inf
    toc = time()
    ymean, ystd = xuyscales['yscale']
    ymean, ystd = (torch.from_numpy(ymean).to(device),
                   torch.from_numpy(ystd).to(device))

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, ymean, ystd)
        val_loss = val_loop(val_dataloader, model, loss_fn, ymean, ystd)
        counter += 1
        if val_loss < best_loss:
            torch.save(model.state_dict(),
                       dest_name)
            best_loss = val_loss
            counter = 0
        print(f'best val_loss: {best_loss}')
        if counter > patience:
            "ran out of patience"
            break
    tic = time()
    model.load_state_dict(torch.load(dest_name))
    model.eval()
    test_losses, test_predictions, test_phis = test_loop([test_dataloader],
                                                         model, loss_fn, ymean,
                                                         ystd)
    train_time = tic - toc
    torch.save(dict(
        env_input_dict=env.input_dict,
        xuyscales=xuyscales,
        state_dict=model.state_dict(),
        test_losses=test_losses,
        test_predictions=test_predictions,
        test_phis=test_phis,
        train_time=train_time
    ),
        dest_name)


def train2(script_name,
           env: Simulator,
           training_info,
           to_backups=False,
           loss_fn=None,):
    """ Get the parameters, training, and validation data."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dirname, basename = utils.get_dir_base(script_name)
    if to_backups:
        dest_name = os.path.join(dirname,
                                 f'backups/'
                                 f'{basename}.pth')
    else:
        dest_name = os.path.join(dirname,
                                 f'build/'
                                 f'{basename}.pth')

    xuyscales = training_info['xuyscales']
    train_data = MyDataset(training_info, env, 'train',
                           oldest_first=False, device=device)
    val_data = MyDataset(training_info, env, 'val',
                         oldest_first=False, device=device)
    test_data = MyDataset(training_info, env, 'test',
                          oldest_first=False, device=device)

    train_dataloader = DataLoader(train_data, batch_size=env.train_batch_size,
                                  num_workers=0)
    val_dataloader = DataLoader(val_data, batch_size=env.val_batch_size,
                                num_workers=0)
    test_dataloader = DataLoader(test_data, batch_size=env.test_batch_size,
                                 num_workers=0)

    model = MultipleStepWrapper(xuyscales, device, env).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-3)
    if loss_fn is None:
        loss_fn = MeanSquaredFunctionLoss(device, env)
    else:
        loss_fn = loss_fn(device, env)

    def train_loop(_dataloader, _model, _loss_fn, _optimizer, _ymean, _ystd):
        for X, y, _ in _dataloader:
            # Compute prediction and loss
            yz0, useq = X
            yz0, useq = yz0.to(device), useq.to(device)
            gpu_y = y.to(device)
            pred, xz, phis = _model(yz0, useq)
            loss = _loss_fn(unnorm(pred, _ymean, _ystd)[:, 1:, :],
                            unnorm(gpu_y, _ymean, _ystd)[:, 1:, :])
            _optimizer.zero_grad()
            loss.backward()
            _optimizer.step()

    def val_loop(_dataloader, _model, _loss_fn, _ymean, _ystd):
        num_batches = len(_dataloader)
        val_loss = 0
        with torch.no_grad():
            for X, y, _ in _dataloader:
                yz0, useq = X
                yz0, useq = yz0.to(device), useq.to(device)
                gpu_y = y.to(device)
                pred, xz, phis = _model(yz0, useq)
                val_loss += _loss_fn(unnorm(pred, _ymean, _ystd)[:, 1:, :],
                                     unnorm(gpu_y, _ymean, _ystd)[:, 1:, :])

        val_loss /= num_batches
        print(f"val_loss: {val_loss:.3e}")
        return val_loss

    def test_loop(_dataloaders, _model, _loss_fn, _ymean, _ystd):
        with torch.no_grad():
            test_losses = []
            preds = []
            phis = []
            for _dataloader in _dataloaders:
                test_loss = 0
                num_batches = len(_dataloader)
                for X, y, _ in _dataloader:
                    yz0, useq = X
                    yz0, useq = yz0.to(device), useq.to(device)
                    gpu_y = y.to(device)
                    pred, xz, phi = _model(yz0, useq)
                    preds.append(pred)
                    phis.append(phi)
                    test_loss += _loss_fn(unnorm(pred, _ymean, _ystd),
                                          unnorm(gpu_y, _ymean, _ystd))
                test_loss /= num_batches
                test_losses.append(test_loss)

        return test_losses, preds, phis

    epochs = 4000
    patience = 20
    counter = 0
    best_loss = np.inf
    toc = time()
    ymean, ystd = xuyscales['yscale']
    ymean, ystd = (torch.from_numpy(ymean).to(device),
                   torch.from_numpy(ystd).to(device))

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, ymean, ystd)
        val_loss = val_loop(val_dataloader, model, loss_fn, ymean, ystd)
        counter += 1
        if val_loss < best_loss:
            torch.save(model.state_dict(),
                       dest_name)
            best_loss = val_loss
            counter = 0
        print(f'best val_loss: {best_loss}')
        if counter > patience:
            "ran out of patience"
            break
    tic = time()
    model.load_state_dict(torch.load(dest_name))
    model.eval()
    test_losses, test_predictions, test_phis = test_loop([test_dataloader],
                                                         model, loss_fn, ymean,
                                                         ystd)
    train_time = tic - toc
    torch.save(dict(
        env_input_dict=env.input_dict,
        xuyscales=xuyscales,
        state_dict=model.state_dict(),
        test_losses=test_losses,
        test_predictions=test_predictions,
        test_phis=test_phis,
        train_time=train_time
    ),
        dest_name)


def test(script_name,
         env: Simulator,
         training_info,
         pth_info,
         to_backups=False,
         loss_fn=None,):
    """ Get the parameters, training, and validation data."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dirname, basename = utils.get_dir_base(script_name)
    if to_backups:
        dest_name = os.path.join(dirname,
                                 f'backups/'
                                 f'{basename}.pth')
    else:
        dest_name = os.path.join(dirname,
                                 f'build/'
                                 f'{basename}.pth')

    xuyscales = training_info['xuyscales']
    test_data = MyDataset(training_info, env, 'test',
                          oldest_first=False, device=device)

    test_dataloader = DataLoader(test_data, batch_size=env.test_batch_size,
                                 num_workers=0)

    model = MultipleStepWrapper(xuyscales, device, env).to(device)
    model.load_state_dict(pth_info['state_dict'])
    if loss_fn is None:
        loss_fn = MeanSquaredFunctionLoss(device, env)
    else:
        loss_fn = loss_fn(device, env)

    def test_loop(_dataloaders, _model, _loss_fn, _ymean, _ystd):
        with torch.no_grad():
            test_losses = []
            preds = []
            phis = []
            for _dataloader in _dataloaders:
                test_loss = 0
                num_batches = len(_dataloader)
                for X, y, _ in _dataloader:
                    yz0, useq = X
                    yz0, useq = yz0.to(device), useq.to(device)
                    gpu_y = y.to(device)
                    pred, xz, phi = _model(yz0, useq)
                    preds.append(pred)
                    phis.append(phi)
                    test_loss += _loss_fn(unnorm(pred, _ymean, _ystd),
                                          unnorm(gpu_y, _ymean, _ystd))
                test_loss /= num_batches
                test_losses.append(test_loss)

        return test_losses, preds, phis
    ymean, ystd = xuyscales['yscale']
    ymean, ystd = (torch.from_numpy(ymean).to(device),
                   torch.from_numpy(ystd).to(device))
    model.eval()
    test_losses, test_predictions, test_phis = test_loop([test_dataloader],
                                                         model, loss_fn, ymean,
                                                         ystd)
    torch.save(dict(
        env_input_dict=pth_info['env_input_dict'],
        xuyscales=xuyscales,
        state_dict=model.state_dict(),
        test_losses=test_losses,
        test_predictions=test_predictions,
        test_phis=test_phis,
        train_time=pth_info['train_time']
    ), dest_name)
