import copy
import matplotlib.pyplot as plt
import casadi as cs
from scipy.special import roots_legendre, roots_chebyu, roots_chebyt, \
    roots_jacobi, roots_hermite, eval_legendre
import logging
import numpy as np
import pickle
from matplotlib.backends.backend_pdf import PdfPages
import os
import re
import collections
from functools import partial
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
import sys
import sympy as sp
from scipy.linalg import block_diag, expm, cholesky
from types import SimpleNamespace
import torch
import gsd.hoomd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.transforms import Bbox
import time
import xml.etree.ElementTree as ET
import io
try:
    import cupy as cp
except ImportError:
    cp = None

SPECIALDIRS = {
    "LIB": ["lib/"],
}

simvars = ['t', 'x', 'u', 'y', 'p', 'ysp']
SimData = collections.namedtuple('SimData', simvars,
                                 defaults=[None for _ in simvars])

from lib.xp_utils import array_module, asnumpy, ascupy, get_array_module

xp = array_module('cupy')


def polinterp(xcol, xint):
    ncol = len(xcol)
    nint = len(xint)
    w = np.zeros((nint, ncol))
    for i in range(ncol):
        d = 1.0
        for j in range(ncol):
            if j != i:
                d = d * (xcol[i] - xcol[j])

        for k in range(nint):
            xx = xint[k]
            n = 1.0
            for j in range(ncol):
                if j != i:
                    n = n * (xx - xcol[j])

            w[k, i] = n / d
    return w


def bisection(f, a, b, tol=1e-14):
    assert np.sign(f(a)) != np.sign(f(b))
    m = None
    while b - a > tol:
        m = a + (b - a) / 2
        fm = f(m)
        if np.sign(f(a)) != np.sign(fm):
            b = m
        else:
            a = m

    return m


def eval_legendre_deriv(n, x):
    return (x * eval_legendre(n, x)
            - eval_legendre(n - 1, x)) / ((x ** 2 - 1) / n)


def export_dict(info_dict, filename, xp):
    exp_dict = info_dict.copy()
    exp_dict = recursive_to_numpy(exp_dict, xp)
    with open(filename, 'wb') as handle:
        pickle.dump(exp_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return None

def export_pth(info_dict, filename, xp):
    exp_dict = info_dict.copy()
    exp_dict = recursive_to_numpy(exp_dict, xp)
    torch.save(exp_dict, filename)
    return None

def import_dict(filename):
    with open(filename, 'rb') as handle:
        info_dict = pickle.load(handle)
    return info_dict


def save_pdf(figure_list, filename, dpi=100):
    with PdfPages(filename) as pdf:
        for counter, figure in enumerate(figure_list):
            print(f'PDF {counter + 1} of {len(figure_list)}', end='\r')
            plt.figure(figure.number)
            pdf.savefig(bbox_inches='tight', dpi=dpi)
    return None


def tex_writer(text, tex_file):
    with open(tex_file, 'w') as texfile:
        for i in range(len(text)):
            texfile.write(text[i])
            texfile.write("\n")


def lazy_pdf(figures, script_path, dpi=100, savefig_kwargs=None):
    if savefig_kwargs is None:
        savefig_kwargs = {}
    dirname = os.path.dirname(script_path)
    dirname = os.path.join(dirname, 'build')
    filename = (f"{dirname}/"
                f"{os.path.basename(script_path).replace('.py', '.pdf')}")
    print(f'Saving to {filename}')
    with PdfPages(filename) as pdf:
        for figure in figures:
            plt.figure(figure.number)
            has_3d_axes = any(isinstance(ax, Axes3D) for ax in figure.get_axes())
            if has_3d_axes:
                # plt.tight_layout()
                # renderer = figure.canvas.get_renderer()
                # bbox = figure.get_tightbbox(renderer)
                # new_bbox = Bbox.from_bounds(bbox.x0,
                #                             bbox.y0,
                #                             bbox.width * 10,
                #                             bbox.height)
                pdf.savefig(dpi=dpi, **savefig_kwargs)
            else:
                pdf.savefig(bbox_inches='tight', dpi=dpi, **savefig_kwargs)


def lazy_pickle_export(info, script_path, to_backups=False, xp=np):
    dirname, basename = get_dir_base(script_path)
    if to_backups:
        filename = os.path.join(
            dirname,
            f'backups/{basename}.pickle')
    else:
        filename = os.path.join(
            dirname,
            f'build/{basename}.pickle')
    export_dict(info, filename, xp)

def lazy_npz(info, script_path, to_backups=False):
    dirname, basename = get_dir_base(script_path)
    if to_backups:
        filename = os.path.join(
            dirname,
            f'backups/{basename}')
    else:
        filename = os.path.join(
            dirname,
            f'build/{basename}')
    np.savez_compressed(filename, **info)


def lazy_pickle_import(script_path):
    dirname = os.getcwd()
    in_build_dir = ('build' not in os.listdir())
    suffixes = ['_pdf', '_mp4', '_tex', '_gsd']
    base_name = os.path.splitext(os.path.basename(script_path))[0]
    for suffix in suffixes:
        base_name = base_name.replace(suffix, '')
    if in_build_dir:
        filename = f"{dirname}/{base_name}_pickle.pickle"
    else:
        filename = f"{dirname}/build/{base_name}_pickle.pickle"
    return import_dict(filename)


def lazy_pickle_import_v2(script_path, use_backups=False):
    info_names = []
    with open(script_path) as f:
        lines = f.readlines()
    counter = 0
    line = lines[counter]
    while '# [depends]' in line:
        info_names.extend(re.findall(r"(\S+\.pickle)", line))
        counter += 1
        line = lines[counter]
    info = []

    dirname, _ = get_dir_base(script_path)
    for info_name in info_names:
        if use_backups:
            filename = os.path.join(dirname,
                                    f"backups/{info_name}")
        else:
            filename = os.path.join(dirname,
                                    f"build/{info_name}")
        for key, val in SPECIALDIRS.items():
            filename = filename.replace(f'build/%{key}%/', val[0])
            filename = filename.replace(f'backups/%{key}%/', val[0])
        info.append(import_dict(filename))
    return info


def lazy_pickle_import_v3(script_path):
    info_names = []
    with open(script_path) as f:
        lines = f.readlines()
    counter = 0
    line = lines[counter]
    while '[depends]' in line:
        info_names.extend(re.findall(r"(\S+\.pickle)", line))
        counter += 1
        line = lines[counter]
    info = []
    dirname, basename = get_dir_base(script_path)
    for info_name in info_names:
        filepath = os.path.join(dirname,
                                f'build/{info_name}')
        for key, val in SPECIALDIRS.items():
            filepath = filepath.replace(f'build/%{key}%/', val[0])

        info.append(import_dict(filepath))
    return info

def get_dependencies_list(script_path, use_backups=False):
    info_names = []

    with open(script_path) as f:
        inside_depends_section = False

        for line in f:
            if '[depends]' in line:
                if line.strip():  # Skip empty lines
                    # Extract dependencies from the line and preserve order
                    info_names.extend(re.findall(r"(\S+\.pickle|\S+\.pth|\S+\.gsd|\S+\.npz|\S+\.tex)", line))

        # Construct file paths and preserve order
        dirname, basename = get_dir_base(script_path)
        info = []
        for info_name in info_names:
            if use_backups:
                filepath = os.path.join(dirname, f'backups/{info_name}')
            else:
                filepath = os.path.join(dirname, f'build/{info_name}')
            for key, val in SPECIALDIRS.items():
                filepath = filepath.replace(f'build/%{key}%/', val[0])
            info.append(filepath)

    return info

def lazy_info_import(script_path, use_backups=False, device='cpu'):
    info_names = get_dependencies_list(script_path, use_backups=use_backups)
    info = []
    for filepath in info_names:
        if use_backups:
            for key, val in SPECIALDIRS.items():
                filepath = filepath.replace(f'backups/%{key}%/', val[0])
        else:
            for key, val in SPECIALDIRS.items():
                filepath = filepath.replace(f'build/%{key}%/', val[0])
        if '.pickle' in filepath:
            info.append(import_dict(filepath))
        elif '.pth' in filepath:
            info.append(torch.load(filepath, map_location=device))
        elif '.gsd' in filepath:
            info.append(gsd.hoomd.open(name=filepath, mode='rb'))
        elif '.npz' in filepath:
            info.append(SimpleNamespace(**np.load(filepath)))
        elif '.tex' in filepath:
            with open(filepath, 'r') as f:
                info.append(f.readlines())
        else:
            # warn user their file was not appended
            logging.warning(f'{filepath} was not appended to info')
            pass
    return info


def lazy_tex(text, script_path):
    dirname = os.path.dirname(script_path)
    filename = os.path.join(
        dirname,
        "build/"
        f"{os.path.basename(script_path).replace('.py', '.tex')}")

    with open(filename, 'w') as texfile:
        for i in range(len(text)):
            texfile.write(text[i])
            texfile.write("\n")


def split_data_into_train_val(simData,
                              n_p,
                              n_train_traj,
                              n_val_traj,
                              n_test_traj,
                              n_train_steps_per_traj,
                              n_val_steps_per_traj,
                              n_test_steps_per_traj,
                              ):
    """ Given one full training data trajectory in the
        simData object, split the entire trajectory into
        training and validation trajectories.

        Np: Number of past measurements and control inputs used for the model.

        nTrainTraj: The entire number of training and trainval
        (cross validatoin) trajectories.

        NtVal: The length of the validation trajectory.
    """
    # Empty list to store the training and validation data.
    data_list = []

    # Number of time steps in each partition of the training data set.
    # Loop over the number of training partitions.
    start = 0
    end = n_train_traj * (n_train_steps_per_traj + 1) + n_p
    for i in range(n_train_traj):
        # Extract the time, state, control input, measurement,
        # and disturbance arrays for the current partition.
        i1 = i * n_train_steps_per_traj
        i2 = i1 + n_p + n_train_steps_per_traj + 1
        t = simData.t[start:end][i1:i2]
        x = simData.x[start:end][i1:i2]
        u = simData.u[start:end][i1:i2]
        y = simData.y[start:end][i1:i2]
        if simData.p is not None:
            p = simData.p[start:end][i1:i2]
        else:
            p = None
        if simData.ysp is not None:
            ysp = simData.ysp[start:end][i1:i2]
        else:
            ysp = None

        # Create a simData object for the current partition.
        train_sim_data = SimData(t=t, x=x, u=u, y=y, p=p, ysp=ysp)

        # Save the simData object to the list.
        data_list += [train_sim_data]
    start = end
    end = start + n_val_traj * (n_val_steps_per_traj + 1) + n_p
    for i in range(n_val_traj):
        # Extract the time, state, control input, measurement,
        # and disturbance arrays for the current partition.
        i1 = i * n_val_steps_per_traj
        i2 = i1 + n_p + n_val_steps_per_traj + 1
        t = simData.t[start:end][i1:i2]
        x = simData.x[start:end][i1:i2]
        u = simData.u[start:end][i1:i2]
        y = simData.y[start:end][i1:i2]
        if simData.p is not None:
            p = simData.p[start:end][i1:i2]
        else:
            p = None
        if simData.ysp is not None:
            ysp = simData.ysp[start:end][i1:i2]
        else:
            ysp = None

        # Create a simData object for the current partition.
        val_sim_data = SimData(t=t, x=x, u=u, y=y, p=p, ysp=ysp)

        # Save the simData object to the list.
        data_list += [val_sim_data]
    start = end
    end = start + n_test_traj * (n_test_steps_per_traj + 1) + n_p
    for i in range(n_test_traj):
        # Extract the time, state, control input, measurement,
        # and disturbance arrays for the current partition.
        i1 = i * n_test_steps_per_traj
        i2 = i1 + n_p + n_test_steps_per_traj + 1
        t = simData.t[start:end][i1:i2]
        x = simData.x[start:end][i1:i2]
        u = simData.u[start:end][i1:i2]
        y = simData.y[start:end][i1:i2]
        if simData.p is not None:
            p = simData.p[start:end][i1:i2]
        else:
            p = None
        if simData.ysp is not None:
            ysp = simData.ysp[start:end][i1:i2]
        else:
            ysp = None

        # Create a simData object for the current partition.
        test_sim_data = SimData(t=t, x=x, u=u, y=y, p=p, ysp=ysp)

        # Save the simData object to the list.
        data_list += [test_sim_data]

    # Return.
    return data_list


def get_uyscaling(*, simData):
    """ Get scalings for the input and output
        from the provided simData. """

    # Mean and standard deviation
    # for the control input.
    umean = np.mean(simData.u, axis=0)
    ustd = np.std(simData.u, axis=0)
    ustd = np.clip(ustd, umean * 1e-8, np.inf)

    # Mean and standard deviation
    # for the measurements.
    ymean = np.mean(simData.y, axis=0)
    
    if np.iscomplex(simData.y).any():
        ystd = np.std(simData.y.real, axis=0) + 1j * np.std(simData.y.imag, axis=0)
    else:
        ystd = np.std(simData.y, axis=0)

    # Return.
    return dict(uscale=(umean, ustd),
                yscale=(ymean, ystd))


def col_swap(mat, to_ode_alg=True, n_blocks=None):
    if n_blocks:
        n_tot = len(mat)
        n_pts = int(n_tot // n_blocks)
        n_a = n_blocks * 2
        alg_cols = [0]
        for i in range(n_a - 1):
            if i % 2 == 0:
                alg_cols.append(alg_cols[-1] + n_pts - 2 + 1)
            else:
                alg_cols.append(alg_cols[-1] + 1)
        alg_bools = np.zeros(n_tot)
        alg_bools[alg_cols] = 1
        alg_bools = np.array(alg_bools, dtype=bool)
        if to_ode_alg:
            a1 = mat[:, ~alg_bools]
            a2 = mat[:, alg_bools]

            return np.hstack((a1,
                              a2))
        else:
            return col_end_to_targets(mat, alg_cols)
    else:
        if to_ode_alg:
            return np.hstack((mat[:, 1:-1],
                              mat[:, 0:1],
                              mat[:, -1:]))
        else:
            return np.hstack((mat[:, -2:-1],
                              mat[:, :-2],
                              mat[:, -1:]))


def row_swap(mat, to_ode_alg=True, n_blocks=None):
    if n_blocks:
        n_tot = len(mat)
        n_pts = int(n_tot // n_blocks)
        n_a = n_blocks * 2
        alg_inds = [0]
        for i in range(n_a - 1):
            if i % 2 == 0:
                alg_inds.append(alg_inds[-1] + n_pts - 2 + 1)
            else:
                alg_inds.append(alg_inds[-1] + 1)
        alg_bools = np.zeros(n_tot)
        alg_bools[alg_inds] = 1
        alg_bools = np.array(alg_bools, dtype=bool)
        if to_ode_alg:

            return row_move_to_end(mat, alg_inds)
        else:
            return row_end_to_targets(mat, alg_inds)
    else:
        if to_ode_alg:
            return np.vstack((mat[1:-1, :],
                              mat[0:1, :],
                              mat[-1:, :]))
        else:
            return np.vstack((mat[-2:-1, :],
                              mat[:-2, :],
                              mat[-1:, :]))


def row_col_swap(mat, to_oa, n_blocks=None):
    ret_mat = copy.deepcopy(mat)
    ret_mat = row_swap(ret_mat, to_ode_alg=to_oa, n_blocks=n_blocks)
    ret_mat = col_swap(ret_mat, to_ode_alg=to_oa, n_blocks=n_blocks)
    return ret_mat


def get_train_val_test_data(*,
                            data_list,
                            n_p,
                            xuyscales,
                            n_train_traj,
                            n_val_traj,
                            oldest_first=True,
                            _norm=True,
                            vander_mat=None):
    """
    Scale all the data trajectories using the provided scaling dictionary
    for training and validation of the black-box and hybrid models.

    :param data_list: List of data trajectories.
    :param n_p: number of past data
    :param xuyscales: Dictionary containing scaling information for state,
            control input, and measurement.
    :param n_train_traj: Number of trajectories to use for training.
    :param n_val_traj: Number of trajectories to use for validation.
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
    xmean, xstd = xuyscales['xscale']
    umean, ustd = xuyscales['uscale']
    ymean, ystd = xuyscales['yscale']

    nx, nu, ny = len(xmean), len(umean), len(ymean)

    if vander_mat is None:
        vander_mat = np.eye(ny)

    # Sizes.
    # Lists to store data.
    # The xseq is collected mainly to check predictions of the unmeasured
    # grey-box states during the training.

    xseq, useq, yseq = [], [], []
    y0, z0, yz0 = [], [], []

    # Loop through all the data trajectories in the data list.
    for data in data_list:
        # Scale data.
        x = data.x
        u = data.u
        y = data.y
        if _norm:
            x = (x - xmean) / xstd
            u = (u - umean) / ustd
            y = (y - ymean) / ystd

        # Get the input and output trajectory.
        x_traj = x[n_p:, :][np.newaxis, ...]
        u_traj = u[n_p:, :][np.newaxis, ...]
        y_traj = y[n_p:, :][np.newaxis, ...]
        # TODO: check to make sure this works for multiple Np>0
        # Get initial y0, z0, and yz0 for the current trajectory.
        if oldest_first:
            yp0seq = y[:n_p, :]
            up0seq = u[:n_p, :]
        else:
            yp0seq = y[:n_p, :][::-1, :]
            up0seq = u[:n_p, :][::-1, :]
        if n_p > 0:
            yp0seq = yp0seq @ vander_mat
        yp0seq = yp0seq.reshape(n_p * ny, )[np.newaxis, :]
        up0seq = up0seq.reshape(n_p * nu, )[np.newaxis, :]
        y0_traj = y[n_p, np.newaxis, :] @ vander_mat
        z0_traj = np.concatenate((yp0seq, up0seq), axis=-1)
        yz0_traj = np.concatenate((y0_traj, z0_traj), axis=-1)

        # Collect the x, u, and y trajectories, and the initial
        # y0, z0, and yz0 in their respective lists.
        xseq += [x_traj]
        useq += [u_traj]
        yseq += [y_traj]
        y0 += [y0_traj]
        z0 += [z0_traj]
        yz0 += [yz0_traj]

    # Data dictionary for the training trajectory.
    train_data = dict(xseq=np.concatenate(xseq[:n_train_traj], axis=0),
                      useq=np.concatenate(useq[:n_train_traj], axis=0),
                      yseq=np.concatenate(yseq[:n_train_traj], axis=0),
                      y0=np.concatenate(y0[:n_train_traj], axis=0),
                      z0=np.concatenate(z0[:n_train_traj], axis=0),
                      yz0=np.concatenate(yz0[:n_train_traj], axis=0))

    # Data dictionary for the validation trajectories.
    xseqs_val = xseq[n_train_traj:n_train_traj + n_val_traj]
    useqs_val = useq[n_train_traj:n_train_traj + n_val_traj]
    yseqs_val = yseq[n_train_traj:n_train_traj + n_val_traj]
    y0_val = y0[n_train_traj:n_train_traj + n_val_traj]
    z0_val = z0[n_train_traj:n_train_traj + n_val_traj]
    yz0_val = yz0[n_train_traj:n_train_traj + n_val_traj]
    val_data = dict(xseq=np.concatenate(xseqs_val, axis=0),
                    useq=np.concatenate(useqs_val, axis=0),
                    yseq=np.concatenate(yseqs_val, axis=0),
                    y0=np.concatenate(y0_val, axis=0),
                    z0=np.concatenate(z0_val, axis=0),
                    yz0=np.concatenate(yz0_val, axis=0))

    # Data dictionary for the testing dataset trajectory.
    xseqs_test = xseq[n_train_traj + n_val_traj:]
    useqs_test = useq[n_train_traj + n_val_traj:]
    yseqs_test = yseq[n_train_traj + n_val_traj:]
    y0_test = y0[n_train_traj + n_val_traj:]
    z0_test = z0[n_train_traj + n_val_traj:]
    yz_test = yz0[n_train_traj + n_val_traj:]
    test_data = dict(xseq=np.concatenate(xseqs_test, axis=0),
                     useq=np.concatenate(useqs_test, axis=0),
                     yseq=np.concatenate(yseqs_test, axis=0),
                     y0=np.concatenate(y0_test, axis=0),
                     z0=np.concatenate(z0_test, axis=0),
                     yz0=np.concatenate(yz_test, axis=0))

    # Return.
    return train_data, val_data, test_data


def reformat_xuyscales(env, xuyscales):
    nx = int(env.n_time_pts * env.n_sim_ode_states)
    nu = env.n_u
    ny = int(env.n_time_pts * env.n_g)
    xmean, xstd = [scale[:nx] for scale in xuyscales['xscale']]
    umean, ustd = [scale[:nu] for scale in xuyscales['uscale']]
    ymean, ystd = [scale[:ny] for scale in xuyscales['yscale']]
    new_xuyscales = dict(
        xscale=(xmean, xstd),
        uscale=(umean, ustd),
        yscale=(ymean, ystd),
    )
    return new_xuyscales


def reformat_xuyscales_v2(xuyscales, n_u, n_y, n_time_pts):
    nu = n_u
    ng = int(n_time_pts * n_y)
    umean, ustd = [scale[:nu] for scale in xuyscales['uscale']]
    ymean, ystd = [scale[:ng] for scale in xuyscales['yscale']]
    new_xuyscales = dict(
        xscale=xuyscales['xscale'],
        uscale=(umean, ustd),
        yscale=(ymean, ystd),
    )
    return new_xuyscales


def get_train_val_test_data_v2(*,
                               data_list,
                               env,
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

    nx = int(env.n_time_pts * env.n_sim_ode_states)
    nu = env.n_u
    ny = int(env.n_time_pts * env.n_g)
    n_p = env.n_p

    n_train_traj = env.n_train_traj
    n_val_traj = env.n_val_traj
    n_test_traj = env.n_test_traj

    xmean, xstd = [scale[:nx] for scale in xuyscales['xscale']]
    umean, ustd = [scale[:nu] for scale in xuyscales['uscale']]
    ymean, ystd = [scale[:ny] for scale in xuyscales['yscale']]

    xp = get_array_module(xmean)

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
            steps_per_traj = int(env.train_sim_time // env.dt) + 1
        else:
            steps_per_traj = int(env.test_sim_time // env.dt) + 1
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


def domovie(fig, filename, frame_func, N, axes=None, beautify=False,
            display=True, incremental=True, fps=15, dpi=100, ffmpeg_args=None):
    # Sort out some arguments.
    if axes is None:
        axes = []

    # Create save object object.
    if ffmpeg_args is None:
        ffmpeg_args = []
    ffmpeg = animation.writers["ffmpeg"]
    mp4 = ffmpeg(fps=fps, extra_args=ffmpeg_args)
    mp4.setup(fig, filename, dpi)

    def saveframe():
        """Saves 1 frame to mp4."""
        mp4.grab_frame()

    def finish():
        """Finishes mp4 file creation."""
        mp4.finish()

    # Then make the frames.
    plt.ioff()
    if display:
        print("\n")
    for i in range(N):
        if display:
            print("\rFrame %4d of %d" % (i + 1, N), end="")
            sys.stdout.flush()
        frame_func(i)
        saveframe()
    print("\n")
    plt.ion()
    finish()


def lazy_movie_pdf(script_path, fig, update, n_frames, savefig_kwargs=None):
    if savefig_kwargs is None:
        savefig_kwargs = {}
    dirname, basename = get_dir_base(script_path)
    filename = (f"{dirname}/"
                f"build/{os.path.basename(script_path)}".replace('.py',
                                                                 '.pdf'))
    with PdfPages(filename) as pdf:
        for i in range(n_frames):
            update(i)  # Update the y-data of the line
            pdf.savefig(**savefig_kwargs)
            print(f'Frame {i + 1} of {n_frames}', end='\r', )


def movie_pdf(save_path, fig, update, n_frames, savefig_kwargs=None):
    with PdfPages(save_path) as pdf:
        for i in range(n_frames):
            update(i)  # Update the y-data of the line
            pdf.savefig(**savefig_kwargs)
            print(f'Frame {i + 1} of {n_frames}', end='\r', )


def norm(_x, _x_mean, _x_std):
    return (_x - _x_mean.reshape(-1, 1)) / _x_std.reshape(-1, 1)


def un_norm(_x, _x_mean, _x_std):
    return _x_std.reshape(-1, 1) * _x + _x_mean.reshape(-1, 1)


def sympy_lagrange_poly(x, order, i, xi=None):
    if xi is None:
        xi = sp.symbols('x:%d' % (order + 1))
    index = [i for i in range(order + 1)]
    index.pop(i)
    return sp.prod([(x - xi[j]) / (xi[i] - xi[j]) for j in index])


def sympy_butcher_tableu(c_col):
    a_mat = []
    b_row = []
    for j_ind in range(len(c_col)):
        x = sp.symbols('x')
        ell_j = sympy_lagrange_poly(x, len(c_col) - 1, j_ind, c_col)
        ell_j_ig = sp.integrate(ell_j)
        b_row.append(ell_j_ig.evalf(subs={x: 1}) - ell_j_ig.evalf(subs={x: 0}))
        a_row = []
        for i_ind, c in enumerate(c_col):
            a_row.append(ell_j_ig.evalf(subs={x: c})
                         - ell_j_ig.evalf(subs={x: 0}))
        a_mat.append(a_row)
    return np.array(a_mat).T, b_row


def butcher_tableu(s, h=1, method='gauss'):
    b_weights = None
    if 'radau' in method:
        c_roots = roots_jacobi(s - 1, 1, 0)[0].flatten()
        c_roots = ((c_roots + 1) / 2)
        if '_l' in method:
            c_roots = np.hstack((np.zeros(1), c_roots))
        elif '_r' in method:
            c_roots = np.hstack((c_roots, np.ones(1)))
        else:
            raise ValueError(f'Method {method} not found, must be in list:'
                             f' ["gauss", "radau_l", "radau_r", "lobatto"]')
    else:
        if 'lobatto' == method:
            c_roots = colloc_v2(s, method='lobatto')[0]
        elif 'gauss' == method:
            c_roots, b_weights = roots_legendre(s)
            c_roots = ((c_roots + 1) / 2)
            c_roots = c_roots.flatten()
            b_weights = b_weights.flatten()
            b_weights = (b_weights / 2)
        elif 'gauss_w_endpoints' == method:
            c_roots = colloc(s - 2, left=True, right=True)[0]
        elif 'gauss_w_right' == method:
            c_roots = colloc(s - 1, right=True)[0]
        else:
            raise ValueError(f'Method {method} not found, must be in list:'
                             f' ["gauss", "radau_l", "radau_r", "lobatto"]')
    if b_weights is None:
        a_mat, b_weights = sympy_butcher_tableu(c_roots)
    else:
        a_mat, _ = sympy_butcher_tableu(c_roots)
    return (np.array(a_mat, dtype='float64'),
            np.array(b_weights, dtype='float64'),
            np.array(c_roots, dtype='float64'))


def get_lobatto(n_pts, epsilon=1e-15, via_recursion=True):
    if via_recursion:
        if n_pts < 2:
            raise ValueError('Error: n must be larger than 1')
        else:
            lg_p = eval_legendre

            def dLgP(n_, xi_):
                """
              Evaluates the first derivative of P_{n}(xi)
              """
                return n_ * (lg_p(n_ - 1, xi_) - xi_
                             * lg_p(n_, xi_)) / (1 - xi_ ** 2)

            def d2LgP(n_, xi_):
                """
              Evaluates the second derivative of P_{n}(xi)
              """
                return (2 * xi_ * dLgP(n_, xi_) - n_
                        * (n_ + 1) * lg_p(n_, xi_)) / (1 - xi_ ** 2)

            def d3LgP(n_, xi_):
                """
              Evaluates the third derivative of P_{n}(xi)
              """
                return (4 * xi_ * d2LgP(n_, xi_) - (n_ * (n_ + 1) - 2)
                        * dLgP(n_, xi_)) / (1 - xi_ ** 2)

            nodes = np.empty(n_pts)
            weights = np.empty(n_pts)

            nodes[0] = -1
            nodes[n_pts - 1] = 1
            weights[0] = weights[0] = 2.0 / (n_pts * (n_pts - 1))
            weights[n_pts - 1] = weights[0]

            n_2 = n_pts // 2

            for i in range(1, n_2):

                xi = (1 - (3 * (n_pts - 2)) / (8 * (n_pts - 1) ** 3)) * \
                     np.cos((4 * i + 1) * np.pi / (4 * (n_pts - 1) + 1))

                error = 1.0

                while error > epsilon:
                    y = dLgP(n_pts - 1, xi)
                    y1 = d2LgP(n_pts - 1, xi)
                    y2 = d3LgP(n_pts - 1, xi)
                    dx = 2 * y * y1 / (2 * y1 ** 2 - y * y2)

                    xi -= dx
                    error = abs(dx)

                nodes[i] = -xi
                nodes[n_pts - i - 1] = xi

                weights[i] = 2 / (n_pts * (n_pts - 1)
                                  * lg_p(n_pts - 1, nodes[i]) ** 2)
                weights[n_pts - i - 1] = weights[i]

            if n_pts % 2 != 0:
                nodes[n_2] = 0
                weights[n_2] = (2.0 /
                                ((n_pts * (n_pts - 1))
                                 * lg_p(n_pts - 1, np.array(nodes[n_2])) ** 2))
    else:
        brackets = roots_legendre(n_pts - 1)[0]
        nodes = np.zeros(n_pts)
        nodes[0] = -1
        nodes[-1] = 1
        for i in range(n_pts - 2):
            nodes[i + 1] = bisection(
                partial(eval_legendre_deriv, n_pts - 1),
                brackets[i], brackets[i + 1])
        max_degree = len(nodes) - 1
        powers = np.arange(max_degree + 1)

        vt = nodes ** powers.reshape(-1, 1)

        a, b = -1, 1
        rhs = 1 / (powers + 1) * (b ** (powers + 1) - a ** (powers + 1))

        weights = np.linalg.solve(vt, rhs)
    r = ((nodes + 1) / 2).reshape(-1, 1)
    q = (weights / 2).reshape(-1, 1)
    return r, q


def get_radau(n_pts):
    r = roots_jacobi(n_pts - 1, 1, 0)[0].flatten()
    r = ((r + 1) / 2)
    r = np.hstack((r, np.ones(1)))
    _, q = sympy_butcher_tableu(r)
    return r, np.array(q)


def get_radau_w_left(n_pts):
    if n_pts <= 2:
        r = np.array([0, 1])
    else:
        r = roots_jacobi(n_pts - 2, 1, 0)[0].flatten()
        r = ((r + 1) / 2)
        r = np.hstack((np.zeros(1), r, np.ones(1)))
    _, q = sympy_butcher_tableu(r)
    return r, np.array(q)


def get_gauss(n_pts):
    r, q = roots_legendre(n_pts)
    r = ((r + 1) / 2).reshape(-1, 1)
    q = (q / 2).reshape(-1, 1)
    return r, q


def colloc_v2(n_pts, interval_size=1, shifted=True, method='gauss'):
    r, q = {'gauss': get_gauss,
            'radau_r': get_radau,
            'lobatto': get_lobatto,
            'radau_w_left': get_radau_w_left}[method](n_pts)
    roots = r.flatten()
    tvar = cs.SX.sym("t")
    n = len(roots)
    polys = [None] * n
    for j in range(n):
        p = cs.DM(1)
        for i in range(n):
            if i != j:
                p *= (tvar - roots[i]) / (roots[j] - roots[i])
        polys[j] = p
    eval_a_at = roots
    n_i = len(eval_a_at)
    n_j = len(polys)
    a = cs.DM.zeros((n_i, n_j))
    b = cs.DM.zeros((n_i, n_j))
    for j, p in enumerate(polys):
        pder = cs.Function("pder", [tvar], [cs.jacobian(p, tvar)])
        pder2 = cs.Function("pder2", [tvar], [cs.hessian(p, tvar)[0]])
        for i in range(n_i):
            a[i, j] = pder(eval_a_at[i])
            b[i, j] = pder2(eval_a_at[i])
    r = r * interval_size
    a = np.array(a) / interval_size
    b = np.array(b) / interval_size ** 2
    q = q * interval_size
    if not shifted:
        r -= interval_size / 2
    return (np.float64(r.flatten()),
            np.float64(a),
            np.float64(b),
            np.float64(q))


def colloc(n_pts, left=False, right=False, plate_gap=1, roots='legendre',
           alpha=0, beta=0, shifted=True, lobatto=False):
    roots_to_fcn = {'legendre': roots_legendre,
                    'chebyu': roots_chebyu,
                    'chebyt': roots_chebyt,
                    'jacobi': roots_jacobi,
                    'hermite': roots_hermite}
    if lobatto:
        brackets = roots_legendre(n_pts - 1)[0]

        nodes = np.zeros(n_pts)
        nodes[0] = -1
        nodes[-1] = 1
        for i in range(n_pts - 2):
            nodes[i + 1] = bisection(
                partial(eval_legendre_deriv, n_pts - 1),
                brackets[i], brackets[i + 1])
        max_degree = len(nodes) - 1
        powers = np.arange(max_degree + 1)

        vt = nodes ** powers.reshape(-1, 1)

        a, b = -1, 1
        rhs = 1 / (powers + 1) * (b ** (powers + 1) - a ** (powers + 1))

        weights = np.linalg.solve(vt, rhs)
        r = ((nodes + 1) / 2).reshape(-1, 1)
        q = (weights / 2).reshape(-1, 1)
    else:
        if n_pts == 0:
            r = np.empty((1, 1))
            q = np.empty((1, 1))
        elif 'jacobi' in roots:
            r, q = roots_to_fcn[roots](n_pts, alpha, beta)
        else:
            r, q = roots_to_fcn[roots](n_pts)
        r = ((r + 1) / 2).reshape(-1, 1)
        q = (q / 2).reshape(-1, 1)
        if left:
            r = np.vstack((np.zeros((1, 1)),
                           r))
            q = np.vstack((np.zeros((1, 1)),
                           q))
        if right:
            r = np.vstack((r,
                           np.ones((1, 1))))
            q = np.vstack((q,
                           np.zeros((1, 1))))

    roots = r.flatten()
    tvar = cs.SX.sym("t")
    n = len(roots)
    polys = [None] * n
    for j in range(n):
        p = cs.DM(1)
        for i in range(n):
            if i != j:
                p *= (tvar - roots[i]) / (roots[j] - roots[i])
        polys[j] = p
    eval_a_at = roots
    n_i = len(eval_a_at)
    n_j = len(polys)
    a = cs.DM.zeros((n_i, n_j))
    b = cs.DM.zeros((n_i, n_j))
    for j, p in enumerate(polys):
        pder = cs.Function("pder", [tvar], [cs.jacobian(p, tvar)])
        pder2 = cs.Function("pder2", [tvar], [cs.hessian(p, tvar)[0]])
        for i in range(n_i):
            a[i, j] = pder(eval_a_at[i])
            b[i, j] = pder2(eval_a_at[i])
    r = r * plate_gap
    a = np.array(a) / plate_gap
    b = np.array(b) / plate_gap ** 2
    q = q * plate_gap
    if not shifted:
        r -= plate_gap / 2
    return r.flatten(), a, b, q


def square_matrix_generator_from_list(eigenvalues, repeated_index_tracker=None):
    """
    Generate a square matrix with specified eigenvalues.

    Args:
        eigenvalues: List of eigenvalues.
        repeated_index_tracker: List indicating repeated eigenvalues (optional).

    Returns:
        mat: Square matrix with specified eigenvalues.
        nonsingular_mat: Non-singular matrix.
        jordan_blk: Jordan block matrix.

    """
    if repeated_index_tracker is None:
        repeated_index_tracker = np.zeros(len(eigenvalues))

    jordan_upper_diagonal = np.empty((0, 0))
    blk_diag = np.empty((0, 0))
    evals = []

    for ind1 in range(len(eigenvalues)):
        if type(eigenvalues[ind1]) is tuple:
            real_part = eigenvalues[ind1][0]
            imag_part = eigenvalues[ind1][1]
            complex_block = np.array([[real_part, -imag_part],
                                      [imag_part, real_part]])
            blk_diag = np.array(block_diag(blk_diag, complex_block))

            if ind1 != 0:
                if repeated_index_tracker[ind1] == 1:
                    upper_block = np.eye(2)
                else:
                    upper_block = np.eye(2) * 0
                jordan_upper_diagonal = np.array(block_diag(
                    jordan_upper_diagonal, upper_block))
        else:
            blk_diag = np.array(block_diag(blk_diag, eigenvalues[ind1]))

            if ind1 != 0:
                if repeated_index_tracker[ind1] == 1:
                    upper_block = np.eye(1)
                else:
                    upper_block = np.eye(1) * 0
                jordan_upper_diagonal = np.array(block_diag(
                    jordan_upper_diagonal, upper_block))

    jordan_upper_deficiency = len(blk_diag) - len(jordan_upper_diagonal)
    jordan_upper_diagonal = np.vstack((jordan_upper_diagonal,
                                       np.zeros(
                                           (jordan_upper_deficiency,
                                            jordan_upper_diagonal.shape[1])),
                                       ))
    jordan_upper_diagonal = np.hstack((np.zeros((jordan_upper_diagonal.shape[0],
                                                 jordan_upper_deficiency)),
                                       jordan_upper_diagonal,
                                       ))
    jordan_blk = blk_diag + jordan_upper_diagonal

    evals = np.array(evals)
    nonsingular_mat = np.random.randint(-2, 2, blk_diag.shape)

    # Generate a nonsingular matrix
    while np.linalg.det(nonsingular_mat) <= 1e-2:
        nonsingular_mat = np.random.randint(-2, 2, blk_diag.shape)

    mat = nonsingular_mat @ jordan_blk @ np.linalg.inv(nonsingular_mat)

    return mat, nonsingular_mat, jordan_blk


def sys_gen_from_list(a11_eigvals, a11_repeat_list,
                      a22_eigvals, a22_repeat_list,
                      a33_eigvals, a33_repeat_list,
                      a44_eigvals, a44_repeat_list):
    """
    Generate a system matrix (A), input matrix (B), output matrix (C),
    and weighting matrices (R, Qw)
    from specified eigenvalues and repeat lists.

    Args:
        a11_eigvals: Eigenvalues for the controllable
        and observable subsystem (a11).
        a11_repeat_list: Repeat list for a11 eigenvalues.
        a22_eigvals: Eigenvalues for the controllable,
        unobservable subsystem (a22).
        a22_repeat_list: Repeat list for a22 eigenvalues.
        a33_eigvals: Eigenvalues for the uncontrollable,
        observable subsystem (a33).
        a33_repeat_list: Repeat list for a33 eigenvalues.
        a44_eigvals: Eigenvalues for the uncontrollable
        and unobservable subsystem (a44).
        a44_repeat_list: Repeat list for a44 eigenvalues.

    Returns:
        A: System matrix.
        B: Input matrix.
        C: Output matrix.
        R: Weighting matrix.
        Qw: Weighting matrix.

    """
    a11 = square_matrix_generator_from_list(a11_eigvals, a11_repeat_list)[0]
    a22 = square_matrix_generator_from_list(a22_eigvals, a22_repeat_list)[0]
    a33 = square_matrix_generator_from_list(a33_eigvals, a33_repeat_list)[0]
    a44 = square_matrix_generator_from_list(a44_eigvals, a44_repeat_list)[0]

    n11 = len(a11)
    n22 = len(a22)
    n33 = len(a33)
    n44 = len(a44)

    a12 = np.zeros((n11, n22))
    a13 = np.random.random((n11, n33))
    a14 = np.zeros((n11, n44))

    a21 = np.random.random((n22, n11))
    a23 = np.random.random((n22, n33))
    a24 = np.random.random((n22, n44))

    a31 = np.zeros((n33, n11))
    a32 = np.zeros((n33, n22))
    a34 = np.zeros((n33, n44))

    a41 = np.zeros((n44, n11))
    a42 = np.zeros((n44, n22))
    a43 = np.random.random((n44, n33))

    A = np.block([[a11, a12, a13, a14],
                  [a21, a22, a23, a24],
                  [a31, a32, a33, a34],
                  [a41, a42, a43, a44]])

    n = n11 + n22 + n33 + n44

    B = rect_mat_generator([n11, n22, n33, n44], n, [1, 1, 0, 0], 'eye', 1)
    C = rect_mat_generator([n11, n22, n33, n44], n, [1, 0, 1, 0], 'eye', 2).T

    Q = np.eye(len(C))
    R = np.eye(C.shape[0])
    Qw = C.T @ Q @ C

    return A, B, C, R, Qw


def rect_mat_generator(ns: list,
                       m: int,
                       blocktype_list: list,
                       nonzero_blocks_type: str,
                       seed: int):
    """
        A matrix to make a nonsquare matrix of size sum(ns) x m
        :param ns: list that specify number of rows for each blocktype
        in blocktype_list
        :param m: number of columns
        :param blocktype_list: list containing 1's or 0's: 1 for
        filled, 0 for zeros
        for ctrl B matrix: [1,1,0,0]
        for obs C matrix: [1,0,1,0]
        :param nonzero_blocks_type: string either "random" or
        "eye" to specify nonzero block numbers
        :param seed: numpy seed
        :return: matrix
        """
    blocktype_list = np.array(blocktype_list)
    np.random.seed(seed)
    if nonzero_blocks_type == 'random':
        mat = np.empty((0, m))
        for ind1 in range(len(blocktype_list)):
            if blocktype_list[ind1] == 0:
                mini_mat = np.zeros((ns[ind1], m))
            elif blocktype_list[ind1] == 1:
                mini_mat = np.random.random((ns[ind1], m))
            else:
                mini_mat = np.empty((0, m))
            mat = np.concatenate((mat, mini_mat), axis=0)
    elif nonzero_blocks_type == 'eye':
        n_ones = 0
        for n, blocktype in zip(ns, blocktype_list):
            if blocktype == 1:
                n_ones += n
        mat = np.eye(n_ones)
        current_index = 0
        nonzero_spots = np.argwhere(blocktype_list == 1).flatten()
        if nonzero_spots[0] == 0 and nonzero_spots[1] == 1:
            mat = np.concatenate((mat, np.zeros((ns[2] + ns[3], n_ones))))
        elif nonzero_spots[0] == 0 and nonzero_spots[1] == 2:
            mat = np.insert(mat, ns[0], np.zeros((ns[1], n_ones)), axis=0)
            mat = np.concatenate((mat, np.zeros((ns[3], n_ones))))
        else:
            raise ValueError("blocktype_list must be "
                             "either [1,1,0,0] or [1,0,1,0]"
                             "for nonzero_mat_type 'eye'")
    else:
        raise ValueError("nonzero_blocks_type must be either 'random' or 'eye'")
    return mat


def reshape_fortran(x, shape):
    """
    Reshape a tensor in Fortran order (last index changes fastest).

    Parameters:
    x (torch.Tensor): The input tensor to be reshaped.
    shape (tuple): The target shape.

    Returns:
    torch.Tensor: The reshaped tensor.
    """
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


def get_dir_base(filename):
    """
    Get the directory name and base name (w/o the extension) of a file.

    Parameters:
    filename (str): The file path.

    Returns:
    tuple: The directory name and base name of the file.
    """
    dirname = os.path.dirname(filename)
    basename = os.path.splitext(os.path.basename(filename))[0]
    return dirname, basename


def col_move_to_end(mat, columns_to_move):
    """
    Move specified columns of a matrix to the end.

    Parameters:
    mat (numpy.ndarray): The input matrix.
    columns_to_move (list): The indices of the columns to move.

    Returns:
    numpy.ndarray: The matrix with specified columns moved to the end.
    """
    n, p = mat.shape
    # Define the indices of the columns you want to move

    # Ensure that the column indices are within the valid range
    valid_indices = list(range(p))
    if any(col >= p for col in columns_to_move):
        raise ValueError("Column indices to move are out of range.")

    # Calculate the list of new column indices
    new_indices = [i for i in valid_indices if
                   i not in columns_to_move] + columns_to_move

    # Reorder the columns in the matrix based on the new indices
    mat_reordered = mat[:, new_indices]
    return mat_reordered


def col_end_to_targets(matrix, target_indices):
    """
    Move specified columns of a matrix from the end to target positions.

    Parameters:
    matrix (numpy.ndarray): The input matrix.
    target_indices (list): The target positions for the columns.

    Returns:
    numpy.ndarray: The matrix with specified columns moved to target positions.
    """
    num_columns = matrix.shape[1]
    num_targets = len(target_indices)

    head_indices = np.arange(num_columns - num_targets)
    tail_indices = np.arange(num_columns - num_targets, num_columns)

    reordered_indices = []
    head_count = 0
    for i in range(num_columns):
        if i in target_indices:
            tail_loc = np.argwhere(i == np.array(target_indices)).flatten()[0]
            reordered_indices.append(tail_indices[tail_loc])
        else:
            reordered_indices.append(head_indices[head_count])
            head_count += 1
    reordered_matrix = matrix[:, reordered_indices]

    return reordered_matrix


def row_end_to_targets(matrix, target_indices):
    """
    Move specified rows of a matrix from the end to target positions.

    Parameters:
    matrix (numpy.ndarray): The input matrix.
    target_indices (list): The target positions for the rows.

    Returns:
    numpy.ndarray: The matrix with specified rows moved to target positions.
    """
    num_rows = matrix.shape[0]
    num_targets = len(target_indices)

    head_indices = np.arange(num_rows - num_targets)
    tail_indices = np.arange(num_rows - num_targets, num_rows)

    reordered_indices = []
    head_count = 0
    for i in range(num_rows):
        if i in target_indices:
            tail_loc = np.argwhere(i == np.array(target_indices)).flatten()[0]
            reordered_indices.append(tail_indices[tail_loc])
        else:
            reordered_indices.append(head_indices[head_count])
            head_count += 1
    reordered_matrix = matrix[reordered_indices, :]

    return reordered_matrix


def row_move_to_end(mat, rows_columns_to_move):
    """
    Move specified rows of a matrix to the end.

    Parameters:
    mat (numpy.ndarray): The input matrix.
    rows_columns_to_move (list): The indices of the rows to move.

    Returns:
    numpy.ndarray: The matrix with specified rows moved to the end.
    """
    n, p = mat.shape
    # Define the indices of the columns you want to move

    # Ensure that the column indices are within the valid range
    valid_indices = list(range(p))
    if any(col >= p for col in rows_columns_to_move):
        raise ValueError("Column indices to move are out of range.")

    # Calculate the list of new column indices
    new_indices = [i for i in valid_indices if
                   i not in rows_columns_to_move] + rows_columns_to_move

    # Reorder the columns in the matrix based on the new indices
    mat_reordered = mat[new_indices, :]
    return mat_reordered


def sliding_windows(arr, window_shape):
    """
    Generate a sliding window view of an array.

    Parameters:
    arr (numpy.ndarray): The input array.
    window_shape (tuple): The shape of the sliding window.

    Returns:
    numpy.ndarray: The array of sliding windows.
    """
    window_shape = tuple(window_shape)
    new_shape = tuple(np.subtract(arr.shape, window_shape) + 1) + window_shape
    strides = arr.strides + arr.strides
    return np.lib.stride_tricks.as_strided(arr, shape=new_shape,
                                           strides=strides)


def generate_interior_points(array1, interior_array):
    """
    Generate interior points between points in an array.

    Parameters:
    array1 (numpy.ndarray): The input array.
    interior_array (numpy.ndarray): The array of interior points.

    Returns:
    numpy.ndarray: The array of generated points.
    """
    diffs = np.diff(array1)
    interior_points = (array1[:-1]
                       + interior_array.reshape(-1, 1)
                       @ diffs.reshape(-1, 1).T).flatten('f')
    return np.hstack(interior_points)


def get_constrained_pts(pts, n_interior):
    """
    Generate constrained points between points in an array.

    Parameters:
    pts (numpy.ndarray): The input array.
    n_interior (int): The number of interior points.

    Returns:
    numpy.ndarray: The array of constrained points.
    """
    if n_interior == 0:
        return pts
    interior_pts = colloc(n_interior)[0]
    constrained_pts = generate_interior_points(pts, interior_pts)
    constrained_pts = np.hstack([pts[0], constrained_pts, pts[-1]])
    return constrained_pts


def get_fcc_pts(w, h, a, xp):
    """
    Generate face-centered cubic (FCC) points in a rectangle.

    Parameters:
    w (float): The width of the rectangle.
    h (float): The height of the rectangle.
    a (float): The spacing between points.
    xp (module): The module that defines the linspace and meshgrid functions (e.g., numpy or cupy).

    Returns:
    numpy.ndarray: The array of FCC points.
    """
    x1 = xp.linspace(-w / 2, w / 2, int(w // a) + 1)
    y1 = xp.linspace(-h / 2, h / 2, int(h // a) + 1)
    # Create a meshgrid
    X1, Y1 = xp.meshgrid(x1, y1)
    face_center_x1 = X1[:-1, :-1] + a / 2
    face_center_y1 = Y1[:-1, :-1] + a / 2
    # Flatten the coordinates to get the face-centered points
    face_centers1 = xp.column_stack((face_center_x1.flatten(),
                                     face_center_y1.flatten()))
    return face_centers1


def recursive_to_device(data, device):
    """
    Recursively move tensors in a data structure to a device.

    This function traverses the input data structure. If a value is a tensor, it moves it to the specified device.
    If a value is a list or dictionary, it recursively applies the same operation.

    Parameters:
    data (list/dict/torch.Tensor): The input data structure where the operation should be applied.
    device (torch.device): The target device.

    Returns:
    list/dict/torch.Tensor: The modified data structure with all tensors moved to the target device.
    """
    if isinstance(data, list):
        return [recursive_to_device(item, device) for item in data]
    elif isinstance(data, dict):
        return {key: recursive_to_device(value, device) for key, value in
                data.items()}
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data


def recursive_apply(func, obj):
    """
    Recursively apply a function to each element of a nested structure.

    This function traverses the input structure. If an element is a dictionary, list, or tuple,
    it recursively applies the same function to that element.

    Parameters:
    func (callable): The function to be applied to each element.
    obj (dict, list, tuple): The input structure where the function should be applied.

    Returns:
    dict, list, tuple: The modified structure with the function applied to each element.
    """
    if isinstance(obj, dict):
        return {key: recursive_apply(func, val) for key, val in obj.items()}
    elif isinstance(obj, list) or type(obj) == tuple:
        return type(obj)(recursive_apply(func, item) for item in obj)
    else:
        return func(obj)

def recursive_to_numpy(mod_dict, xp):
    """
    Recursively convert all ndarray values in a dictionary to numpy arrays.

    This function traverses the input dictionary. If a value is an ndarray, it converts it to a numpy array.
    If a value is a dictionary, it recursively applies the same operation.

    Parameters:
    mod_dict (dict): The input dictionary where the conversion should be applied.
    xp (module): The module that defines the ndarray type (e.g., numpy or cupy).

    Returns:
    dict: The modified dictionary with all ndarray values converted to numpy arrays.
    """
    def mini_to_numpy(val, xp):
        if isinstance(val, xp.ndarray):
            return asnumpy(val)
        else:
            return val
    return recursive_apply(partial(mini_to_numpy, xp=xp), mod_dict)


def norm_mat(mat, mat_min, mat_max, min_val=-1, max_val=1):
    """
    Normalize a matrix to the range [-1, 1].

    The normalization is done based on the minimum and maximum values provided.

    Parameters:
    mat (numpy.ndarray): The input matrix to be normalized.
    mat_min (float): The minimum value for normalization.
    mat_max (float): The maximum value for normalization.

    Returns:
    numpy.ndarray: The normalized matrix.
    """
    scale = max_val - min_val
    offset = np.ones_like(mat) * min_val
    return scale * (mat - mat_min) / (mat_max - mat_min) + offset


def get_basis_vector(i, n):
    """Get the ith basis vector of an n-dimensional space."""
    basis_vector = np.zeros(n)
    basis_vector[i] = 1
    return basis_vector


def augknt(knots, k, m=1):
    return np.array([knots[0]] * (k + 1)
                    + list(np.repeat(knots[1:-1], m))
                    + [knots[-1]] * (k + 1))



def legendre_matrix(x, n, bounds):
    """
    Construct the Legendre matrix of degree n for given x values.
    """
    x_ = 2 * (x - bounds[0]) / (bounds[1] - bounds[0]) - 1
    X = np.vstack([np.polynomial.legendre.Legendre.basis(i)(x_) for i in range(n + 1)]).T
    return np.linalg.inv(X)


def generate_toeplitz_tensor(n, dtype=None):
    """
    Return a 3D tensor that can create Toeplitz matrices of size 2n-1 x n x n.
    Usage:
    ```
    toeplitz_matrix = torch.tensordot(T, vec, dims=([2], [-1]))
    toeplitz_matrix = toeplitz_matrix.permute(*range(2, toeplitz_matrix.dim()), 0, 1)
    ```
    """
    # Get dimensions
    T = torch.zeros(2*n -1, n, n, dtype=dtype)

    # Fill the tensor T according to the general pattern
    for j in range(n):
        for i in range(j, j + n):
            if i < 2*n-1:
                T[i, j, i - j] = 1
    return T

def toeplitz_last_dim(v, T):
    """
    Compute the Toeplitz matrix-vector product with the last dimension of the tensor T.
    """
    toeplitz_matrix = torch.tensordot(T, v, dims=([2], [-1]))
    toeplitz_matrix = toeplitz_matrix.permute(*range(2, toeplitz_matrix.dim()), 0, 1)
    return toeplitz_matrix



def batched_block_diag(arg):
    # Define the function to be vectorized using torch.vmap
    def block_diag_func(arg):
        return torch.block_diag(*arg)
    
    # Use torch.vmap to vectorize over batches (in_dim=0 specifies batch dimension)
    return torch.vmap(block_diag_func, in_dims=0)(arg)

def periodic_clip(numbers, lower, upper):
    range_size = upper - lower
    
    # Convert to numpy array if it isn't already
    is_scalar = np.isscalar(numbers)
    numbers_array = np.asarray(numbers)
    
    # Vectorized operations for numpy arrays
    adjusted_numbers = (numbers_array - lower) % range_size + lower
    
    # Handle array case
    if not is_scalar:
        mask = adjusted_numbers > upper
        adjusted_numbers[mask] -= range_size
    # Handle scalar case
    else:
        if adjusted_numbers > upper:
            adjusted_numbers -= range_size
            
    return adjusted_numbers


class AnimationController:
    def __init__(self, fig, update, n_frames, init_frame=0):
        self.fig = fig
        self.n_frames = n_frames
        self.playing = False
        self.update = update
        self.current_frame = init_frame
        self.update(init_frame)

        # Create the slider
        slider_ax = plt.axes([0.25, 0.01, 0.5, 0.03], facecolor='lightgoldenrodyellow')
        self.frame_slider = Slider(slider_ax, 'Frame', 0, self.n_frames - 1, valinit=init_frame, valstep=1)
        self.frame_slider.on_changed(self.update_frame)

        # Create the play button
        button_ax = plt.axes([0.8, 0.025, 0.1, 0.04])  # Position for the play button
        self.button = Button(button_ax, 'Play')
        self.button.on_clicked(self.toggle_play)

        # Connect the keyboard event handler
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # Animation object, initialized to None
        self.anim = None

    def update_frame(self, val):
        """Update function to change the frame based on slider value."""
        self.update(int(val))  # Update the plot based on the current slider value
        self.current_frame = int(val)
        self.fig.canvas.draw_idle()

    def toggle_play(self, event):
        """Toggle play/pause when the button is clicked."""
        if not self.playing:
            self.playing = True
            self.button.label.set_text('Pause')
            self.start_animation()
        else:
            self.playing = False
            self.button.label.set_text('Play')
            if self.anim:
                self.anim.event_source.stop()

    def start_animation(self):
        """Start the animation using FuncAnimation."""
        def update_frame_func(frame):
            if self.current_frame >= self.n_frames:
                self.playing = False
                self.button.label.set_text('Play')
                self.anim.event_source.stop()
                return  # End the animation
            else:
                self.frame_slider.set_val(self.current_frame)
                self.update(self.current_frame)
                self.current_frame += 1

        self.anim = animation.FuncAnimation(self.fig, update_frame_func, frames=range(self.n_frames), interval=100)
        plt.draw()

    def on_key(self, event):
        """Keyboard event handler to adjust slider with arrow keys."""
        current_val = self.frame_slider.val
        step = 5 if event.key in ['shift+left', 'shift+right'] else 1

        if event.key == 'right':
            new_val = min(current_val + step, self.n_frames - 1)
        elif event.key == 'left':
            new_val = max(current_val - step, 0)
        elif event.key == 'shift+right':
            new_val = min(current_val + 5, self.n_frames - 1)
        elif event.key == 'shift+left':
            new_val = max(current_val - 5, 0)
        else:
            return

        self.frame_slider.set_val(new_val)

def find_close_indices_vectorized(ts, target_times, atol=0.005, chunk_size=1000):
                # Pre-allocate results array
                indices = []
                
                # Process target times in chunks to avoid memory issues
                for i in range(0, len(target_times), chunk_size):
                    chunk = target_times[i:i + chunk_size]
                    
                    # Get rough bounds for each target time to reduce search space
                    left_bounds = np.searchsorted(ts, chunk - atol)
                    right_bounds = np.searchsorted(ts, chunk + atol)
                    
                    # For each pair of bounds in this chunk
                    for left, right, target in zip(left_bounds, right_bounds, chunk):
                        # Only check elements within the bounds
                        mask = np.abs(ts[left:right] - target) <= atol
                        found_indices = np.where(mask)[0] + left
                        indices.extend(found_indices)
                
                return np.array(sorted(set(indices)))

def remove_layers_from_svg(svg_file, layer_labels_to_remove):
    """
    Remove specified layers from SVG content and return the modified SVG as a string.
    
    Args:
        svg_content (str): The input SVG file content
        layer_labels_to_remove (list): List of inkscape:label values to remove
    
    Returns:
        str: Modified SVG content as string
    """
    # Register the inkscape namespace
    with open(svg_file, 'r') as f:
        svg_content = f.read()
    ET.register_namespace('', "http://www.w3.org/2000/svg")
    ET.register_namespace('inkscape', "http://www.inkscape.org/namespaces/inkscape")
    
    # Parse the SVG content
    tree = ET.ElementTree(ET.fromstring(svg_content))
    root = tree.getroot()
    
    # Find all group elements
    for g in root.findall('{http://www.w3.org/2000/svg}g'):
        # Check if this is an inkscape layer
        label = g.get('{http://www.inkscape.org/namespaces/inkscape}label')
        if label in layer_labels_to_remove:
            root.remove(g)
    
    # Convert back to string
    return ET.tostring(root, encoding='unicode')

def keep_only_layers(svg_file, layer_labels_to_keep):
    """
    Keep only specified layers in SVG content and remove all others.
    
    Args:
        svg_content (str): The input SVG file content
        layer_labels_to_keep (list): List of inkscape:label values to keep
    
    Returns:
        str: Modified SVG content as string
    """
    with open(svg_file, 'r') as f:
        svg_content = f.read()
    # Register the inkscape namespace
    ET.register_namespace('', "http://www.w3.org/2000/svg")
    ET.register_namespace('inkscape', "http://www.inkscape.org/namespaces/inkscape")
    
    # Parse the SVG content
    tree = ET.ElementTree(ET.fromstring(svg_content))
    root = tree.getroot()
    
    # Find all group elements and remove those not in the keep list
    for g in root.findall('{http://www.w3.org/2000/svg}g'):
        # Check if this is an inkscape layer
        label = g.get('{http://www.inkscape.org/namespaces/inkscape}label')
        if label is not None and label not in layer_labels_to_keep:
            root.remove(g)
    
    # Convert back to string
    return ET.tostring(root, encoding='unicode')

def replace_rectangle_with_pdf(path_line, pdf_info):
    """
    Convert a TikZ rectangle path to a PDF inclusion node
    
    Args:
        path_line (str): The TikZ path line containing the rectangle
        pdf_info (dict): Dictionary containing 'page' and 'file' information for the PDF
    
    Returns:
        str: LaTeX code for including the PDF
    """
    # Extract the shift coordinates
    shift_pattern = r"shift=\{([-\d.,]+)\}"
    shift_match = re.search(shift_pattern, path_line)
    shift_coords = shift_match.group(1).split(',') if shift_match else [0, 0]
    shift_x, shift_y = [float(coord) for coord in shift_coords]
    
    # Extract rectangle coordinates
    rect_pattern = r"\(([-\d.,\s]+)\)\s+rectangle\s+\(([-\d.,\s]+)\)"
    rect_match = re.search(rect_pattern, path_line)
    if not rect_match:
        raise ValueError("Could not find rectangle coordinates in path")
    
    # Parse coordinates
    start_coords = [float(x) for x in rect_match.group(1).replace(' ', '').split(',')]
    end_coords = [float(x) for x in rect_match.group(2).replace(' ', '').split(',')]
    
    # Apply shift to coordinates
    start_x = start_coords[0] + shift_x
    start_y = start_coords[1] + shift_y
    
    # Calculate width and height in cm
    width = abs(end_coords[0] - start_coords[0])
    height = abs(end_coords[1] - start_coords[1])
    
    # Create the new node with PDF inclusion
    new_line = (
        f"\\node[inner sep=0pt, anchor=north west]"
        f"at ({start_x},{start_y})"
        f" {{\\includegraphics[width={width}cm, height={height}cm, page={pdf_info['page']}]"
        f"{{{pdf_info['file']}}}}};"
    )
    
    return new_line

def adaptive_stepping(f, x_start, x_target, initial_state, max_iterations=1000, tolerance=1e-10):
    """
    Adaptively steps from x_start to x_target using binary search when steps fail.
    
    Args:
        f: Function that takes (x, prev_state) and returns (success, new_state)
        x_start: Starting x value (where we know the solution)
        x_target: Target x value we want to reach
        initial_state: Initial state at x_start
        max_iterations: Maximum number of iterations to prevent infinite loops
        tolerance: Minimum distance between points to consider them different
    
    Returns:
        tuple: (success, final_state, x_history, state_history)
    """
    x_history = [x_start]
    state_history = [initial_state]
    current_state = initial_state
    
    # If we can directly reach the target, we're done
    success, final_state = f(x_target, current_state)
    if success:
        return True, final_state, [x_start, x_target], [initial_state, final_state]
    
    x_last_success = x_start
    last_good_state = initial_state
    x_last_failure = x_target
    
    iteration = 0
    while iteration < max_iterations:
        # Try a point halfway between last success and last failure
        x_try = (x_last_success + x_last_failure) / 2
        
        # If we're too close to the last success point, we're done
        if abs(x_try - x_last_success) < tolerance:
            return False, last_good_state, x_history, state_history
        
        # Try the new point
        success, new_state = f(x_try, last_good_state)
        
        x_history.append(x_try)
        state_history.append(new_state if success else None)
        
        if success:
            x_last_success = x_try
            last_good_state = new_state
            
            # Try to reach target from this new successful point
            success, final_state = f(x_target, last_good_state)
            if success:
                x_history.append(x_target)
                state_history.append(final_state)
                return True, final_state, x_history, state_history
        else:
            x_last_failure = x_try
            
        iteration += 1
    
    return False, last_good_state, x_history, state_history