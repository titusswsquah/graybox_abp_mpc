from torch.utils.data import Dataset
from lib import utils
from lib.hd_simulator import Simulator
import torch
import numpy as np


def get_train_val_test_data(*,
                            data_list,
                            env: Simulator,
                            xuyscales,
                            oldest_first=True,
                            _norm=True,
                            vander_mat=None):
    """
    Scale all the data trajectories using the provided scaling dictionary
    for training and validation of the black-box and hybrid models.


    :param env: Simulator object
    :param data_list: List of data trajectories.
    :param xuyscales: Dictionary containing scaling information for state,
            control input, and measurement.
    :param oldest_first: Whether to stack past data from oldest to
            newest. Defaults to True.
    :param _norm: Whether to norm data. Defaults to True.
    :param vander_mat: Vandermonde matrix to map from measurement to state
                        for yz0. Defaults to None.
    :return: tuple: A tuple containing train_data, val_data, and test_data
            dictionaries.
    """
    # Extract the mean and standard deviations of the scaling
    # of the state, control input, and measurement.

    nx = int(env.n_time_pts * env.n_state_pts * (env.meas_order * 2 - 1))
    nu = env.n_u
    ny = int(env.n_g)
    n_p = env.n_p

    n_train_traj = env.n_train_traj
    n_val_traj = env.n_val_traj
    n_test_traj = env.n_test_traj

    xmean, xstd = [scale[:nx] for scale in xuyscales['xscale']]
    umean, ustd = [scale[:nu] for scale in xuyscales['uscale']]
    ymean, ystd = [scale[:ny] for scale in xuyscales['yscale']]

    xp = utils.get_array_module(xmean)

    if vander_mat is None:
        vander_mat = xp.eye(env.n_g)

    # Sizes.
    # Lists to store data.
    # The xseq is collected mainly to check predictions of the unmeasured
    # grey-box states during the training.

    xseq, useq, yseq = [], [], []
    y0, z0, yz0 = [], [], []

    # Loop through all the data trajectories in the data list.
    for ind, data in enumerate(data_list):

        if ind < env.n_train_traj + env.n_val_traj:
            steps_per_traj = int(env.train_sim_time // env.nnam_dt) + 1
        else:
            steps_per_traj = int(env.test_sim_time // env.nnam_dt) + 1
        # Scale data.
        x = data.x[:, :nx]
        u = data.u[:, :nu]
        y = data.y[:, :ny]
        if _norm:
            x = (x - xmean) / xstd
            u = (u - umean) / ustd
            y = (y - ymean) / ystd

        # Get the input and output trajectory.
        x_traj = x[n_p + 1:n_p + steps_per_traj, :][xp.newaxis, ...]
        u_traj = u[n_p:n_p + steps_per_traj - 1, :][xp.newaxis, ...]
        y_traj = y[n_p + 1:n_p + steps_per_traj, :][xp.newaxis, ...]
        # TODO: check to make sure this works for multiple Np>0
        if oldest_first:
            yp0seq = y[:n_p, :]
            up0seq = u[:n_p, :]
        else:
            yp0seq = y[:n_p, :][::-1, :]
            up0seq = u[:n_p, :][::-1, :]
        if n_p > 0:
            yp0seq = (yp0seq.reshape(env.n_p, -1, env.n_g)
                      @ vander_mat).reshape(env.n_p, ny)
        yp0seq = yp0seq.reshape(n_p * ny, )[xp.newaxis, :]
        up0seq = up0seq.reshape(n_p * nu, )[xp.newaxis, :]
        y0_traj = (y[n_p, xp.newaxis, :].reshape(1, -1, env.n_g)
                   @ vander_mat).reshape(1, ny)
        z0_traj = xp.concatenate((yp0seq, up0seq), axis=-1)
        yz0_traj = xp.concatenate((y0_traj, z0_traj), axis=-1)

        # Collect the x, u, and y trajectories, and the initial
        # y0, z0, and yz0 in their respective lists.
        xseq += [x_traj]
        useq += [u_traj]
        yseq += [y_traj]
        y0 += [y0_traj]
        z0 += [z0_traj]
        yz0 += [yz0_traj]

    # Data dictionary for the training trajectory.
    train_data = dict(xseq=xp.concatenate(xseq[:n_train_traj], axis=0),
                      useq=xp.concatenate(useq[:n_train_traj], axis=0),
                      yseq=xp.concatenate(yseq[:n_train_traj], axis=0),
                      y0=xp.concatenate(y0[:n_train_traj], axis=0),
                      z0=xp.concatenate(z0[:n_train_traj], axis=0),
                      yz0=xp.concatenate(yz0[:n_train_traj], axis=0))

    # Data dictionary for the validation trajectories.
    xseqs_val = xseq[n_train_traj:n_train_traj + n_val_traj]
    useqs_val = useq[n_train_traj:n_train_traj + n_val_traj]
    yseqs_val = yseq[n_train_traj:n_train_traj + n_val_traj]
    y0_val = y0[n_train_traj:n_train_traj + n_val_traj]
    z0_val = z0[n_train_traj:n_train_traj + n_val_traj]
    yz0_val = yz0[n_train_traj:n_train_traj + n_val_traj]
    val_data = dict(xseq=xp.concatenate(xseqs_val, axis=0),
                    useq=xp.concatenate(useqs_val, axis=0),
                    yseq=xp.concatenate(yseqs_val, axis=0),
                    y0=xp.concatenate(y0_val, axis=0),
                    z0=xp.concatenate(z0_val, axis=0),
                    yz0=xp.concatenate(yz0_val, axis=0))

    # Data dictionary for the testing dataset trajectory.
    xseqs_test = xseq[-n_test_traj:]
    useqs_test = useq[-n_test_traj:]
    yseqs_test = yseq[-n_test_traj:]
    y0_test = y0[-n_test_traj:]
    z0_test = z0[-n_test_traj:]
    yz_test = yz0[-n_test_traj:]
    test_data = dict(xseq=xp.concatenate(xseqs_test, axis=0),
                     useq=xp.concatenate(useqs_test, axis=0),
                     yseq=xp.concatenate(yseqs_test, axis=0),
                     y0=xp.concatenate(y0_test, axis=0),
                     z0=xp.concatenate(z0_test, axis=0),
                     yz0=xp.concatenate(yz_test, axis=0))

    # Return.
    return train_data, val_data, test_data


class MyDataset(Dataset):
    def __init__(self, info, env: Simulator, set_type, oldest_first=False,
                 device="cpu", norm=True):
        training_data = info['training_data']
        xuyscales = info['xuyscales']

        (train_data, val_data, test_data) = get_train_val_test_data(
            env=env,
            data_list=training_data,
            xuyscales=xuyscales,
            oldest_first=oldest_first,
            _norm=norm
        )

        if 'train' in set_type:
            data = train_data
            n_trajs = (env.n_train_batches * env.train_batch_size)
        elif 'val' in set_type:
            data = val_data
            n_trajs = (env.n_val_batches * env.val_batch_size)
        elif 'test' in set_type:
            data = test_data
            n_trajs = (env.n_test_batches * env.test_batch_size)
        else:
            raise ValueError('set_type must be one of the following: '
                             '["train", "val", "test"')

        self.yz0 = torch.as_tensor(
            np.expand_dims(data['yz0'], axis=-1)).to(device)[:n_trajs, :]
        self.useq = torch.as_tensor(
            np.transpose(data['useq'], axes=(0, 2, 1))).to(device)[:n_trajs, :]
        self.y = torch.as_tensor(
            np.transpose(data['yseq'], axes=(0, 2, 1))).to(device)[:n_trajs, :]
        self.xseq = torch.as_tensor(
            np.transpose(data['xseq'], axes=(0, 2, 3, 4, 1))).to(device)[:n_trajs, :]

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return (self.yz0[idx], self.useq[idx]), self.y[idx], self.xseq[idx]
