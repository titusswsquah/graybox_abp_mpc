try:
    import hoomd
except ImportError:
    hoomd = None
    print("warning: hoomd is not loaded")    
from abc import abstractmethod
from lib.utils import get_array_module, array_module, asnumpy
from lib.utils import xp as xp_
from lib.utils import periodic_clip
from lib.utils import augknt
import numpy as np
import casadi as ca
import torch
from scipy.interpolate import make_lsq_spline


def fourier_integrate(cns, bounds, xp=xp_):
    scale = (bounds[1] - bounds[0])
    return (cns[0] ** 2 + 2 * xp.sum(xp.abs(cns[1:] / 2) ** 2)) * scale


def fourier_fit(y, bounds, order, xp=xp_):
    # scale = (bounds[1] - bounds[0])
    # delta = scale / ((2 * order) - 1)
    # x = xp.arange(0, ((2 * order) - 1) * delta, delta)
    hat_y = xp.fft.fft(y) / ((2 * order) - 1)
    cns = hat_y[:order]
    cns[1:] *= 2
    return cns


def evaluate_fourier_series(x_, cns_, bounds, xp=xp_):
    """
    Evaluates a Fourier series at given points.

    Args:
        x_ (array-like): Input points at which to evaluate the Fourier series.
        cns_ (array-like): Coefficients of the Fourier series.
        bounds (tuple): Bounds of the input points.
        xp (array-like, optional): Array library to use for computations. Defaults to xp_.

    Returns:
        array-like: The values of the Fourier series evaluated at the input points.

    Note:
        cns_ should be in the form [c0, c1, c2, ..., cN] where c0 is the constant term.
    """
    norm_x = (x_ - bounds[0]) / (bounds[1] - bounds[0])
    return (cns_ @ xp.exp(1j * 2 * xp.pi * norm_x[:, xp.newaxis] * xp.arange(cns_.shape[-1])).T).real


def evaluate_fourier_series_v2(x_, cns_, bounds, xp=xp_):
    """
    Evaluates a Fourier series at given points.

    Args:
        x_ (array-like): Input points at which to evaluate the Fourier series.
        cns_ (array-like): Coefficients of the Fourier series.
        bounds (tuple): Bounds of the input points.
        xp (array-like, optional): Array library to use for computations. Defaults to xp_.

    Returns:
        array-like: The values of the Fourier series evaluated at the input points.

    Note:
        cns_ should be in the form [c0, c1, c2, ..., cN] where c0 is the constant term.
    """
    norm_x = (x_ - bounds[0]) / (bounds[1] - bounds[0])
    return xp.sum(cns_[xp.newaxis, :] * xp.exp(1j * 2 * xp.pi * norm_x[:, xp.newaxis] * xp.arange(len(cns_))),
                  axis=1).real
def evaluate_fourier_series_v3(x_, cns_, bounds, xp=xp_):
    """
    Evaluates a Fourier series at given points.

    Args:
        x_ (array-like): Input points at which to evaluate the Fourier series.
        cns_ (array-like): Coefficients of the Fourier series.
        bounds (tuple): Bounds of the input points.
        xp (array-like, optional): Array library to use for computations. Defaults to xp_.

    Returns:
        array-like: The values of the Fourier series evaluated at the input points.

    Note:
        cns_ should be in the form [c-N, ... c-1, c0, c1, c2, ..., cN] where c0 is the constant term.
    """
    norm_x = (x_ - bounds[0]) / (bounds[1] - bounds[0])
    exps = xp.arange(cns_.shape[-1]) - cns_.shape[-1] // 2
    return (cns_ @ xp.exp(1j * 2 * xp.pi * norm_x[:, xp.newaxis] * exps).T).real

def samp_to_fourier(unnorm_x, thetas, J_, order, bounds, xp=xp_, forces=None):
    scale = (bounds[1] - bounds[0])
    x_ = (unnorm_x - bounds[0]) / scale
    vals = xp.exp(-1j * 2 * xp.pi * xp.arange(J_)[:, xp.newaxis] * x_)
    coeffs = xp.mean(vals, axis=1) / scale
    coeffs[0] = 1 / scale
    coeffs[1:] *= 2
    state = {'pr000r': coeffs.real,
             'pr000i': coeffs.imag}
    for i in range(1, order):
        weight = xp.cos(i * thetas)
        val = xp.mean(vals * weight, axis=1) / scale
        state['pr{0:0=3d}r'.format(i)] = val.real
        state['pr{0:0=3d}i'.format(i)] = val.imag
        weight = xp.sin(i * thetas)
        val = xp.mean(vals * weight, axis=1) / scale
        state['pi{0:0=3d}r'.format(i)] = val.real
        state['pi{0:0=3d}i'.format(i)] = val.imag
    if forces is not None:
        weight = forces
        val = xp.mean(vals * weight, axis=1) / scale
        state['forcer'] = val.real
        state['forcei'] = val.imag
    return state

def fourier_density_estimate(unnorm_x, weights, J_, bounds, xp=xp_):
    scale = (bounds[1] - bounds[0])
    x_ = (unnorm_x - bounds[0]) / scale
    vals = xp.exp(-1j * 2 * xp.pi * xp.arange(J_)[:, xp.newaxis] * x_)
    coeffs = xp.mean(vals * weights, axis=1) / scale
    coeffs[0] = 1 / scale
    coeffs[1:] *= 2
    return coeffs


class BaseController(hoomd.md.force.Custom):
    def __init__(self,
                 env,
                 full_state0
                 ):
        meas_order = env.meas_order
        super().__init__(aniso=False)
        self.n_meas_modes = env.n_meas_modes
        self.meas_order = meas_order
        self.full_state = None
        for key in env.full_state_keys:
            setattr(self, key, full_state0[key])
        self.bounds = env.bounds
        self.dt = env.md_samp_dt
        self.plate_gap = env.plate_gap


    # @profile
    def measure(self,
                evalulate_meas_vars, forces=None, xp=xp_):
        if hoomd.version.gpu_enabled:
            local_snapshot = self._state.gpu_local_snapshot
        else:
            local_snapshot = self._state.cpu_local_snapshot
        with local_snapshot as snapshot:
            pcl_pos = xp.array(snapshot.particles.position[:, 1], copy=False)
            pcl_quarts = xp.array(snapshot.particles.orientation, copy=False)
            pcl_thetas = gpu_quaternion_to_euler_angle_vectorized3(
                pcl_quarts[:, 0],
                pcl_quarts[:, 3], xp)
            if evalulate_meas_vars:
                pcl_rtags = snapshot.particles.rtag
                self.full_state = samp_to_fourier(pcl_pos[pcl_rtags],
                                                  pcl_thetas[pcl_rtags],
                                                  self.n_meas_modes,
                                                  self.meas_order,
                                                  self.bounds,
                                                  forces=forces
                                                  )
                for key in self.full_state.keys():
                    setattr(self, key, asnumpy(self.full_state[key]))
            return pcl_pos, pcl_thetas

    @abstractmethod
    def set_forces(self, timestep):
        pass


def gpu_quaternion_to_euler_angle_vectorized3(w, z, xp=xp_):
    t3 = +2.0 * (w * z)
    t4 = +1.0 - 2.0 * (z * z)
    Z = xp.degrees(xp.arctan2(t3, t4))
    return Z * xp.pi / 180


def keys_to_array(keys, log, cutoff=None, xp=xp_):
    reals = xp.array([log[f'{key}r'][:cutoff] for key in keys])
    imags = xp.array([log[f'{key}i'][:cutoff] for key in keys])
    return xp.concatenate((reals, imags[:, 1:]), axis=1)


def stacked_to_cplx(stacked_real_imag, xp=xp_):
    """
    Takes [Real, Imag] pairs and returns Real + Imag[1:].
    """
    zero_shape = list(stacked_real_imag.shape)
    zero_shape[0] = 1
    n_modes = stacked_real_imag.shape[0]//2
    reals = stacked_real_imag[:n_modes+1]
    imags = xp.concatenate([xp.zeros(zero_shape),
                            stacked_real_imag[n_modes+1:]])
    return reals + 1j * imags


def cplx_to_stacked(cplx, xp=xp_):
    """
    Takes Real + Imag[1:] and returns [Real, Imag].
    """
    n_modes = cplx.shape[0]
    reals = cplx.real
    imags = cplx.imag
    return xp.concatenate((reals, imags[1:]))


def torch_stacked_to_cplx(stacked_real_imag, scale_k=False):
    """
    Takes [Real, Imag] pairs and returns Real[1:]-Imag[1:], Real[0],Real[1:] + Imag[1:].
    """
    n_modes = stacked_real_imag.shape[-1]//2
    reals = stacked_real_imag[..., :n_modes+1]
    imags = stacked_real_imag[..., n_modes+1:]
    if scale_k:
        reals[..., 1:] /= 2
        imags /= 2
    plus = reals[..., 1:] + 1j * imags
    minus = torch.flip(reals[..., 1:] - 1j * imags, dims=[-1])
    cplx = torch.cat((reals[..., 0:1], plus, minus), dim=-1)
    return torch.fft.fftshift(cplx, dim=-1)


def torch_cplx_to_stacked(cplx, scale_k=False):
    """
    Real[1:]-Imag[1:], Real[0],Real[1:] + Imag[1:] to [Real, Imag].
    """
    n_modes = cplx.shape[-1]//2
    shifted = torch.fft.ifftshift(cplx, dim=-1)
    reals = shifted[..., :n_modes+1].real
    imags = shifted[..., :n_modes+1].imag[..., 1:]
    if scale_k:
        reals[..., 1:] *= 2
        imags *= 2
    return torch.cat((reals, imags), dim=-1)


class XFilter(hoomd.filter.CustomFilter):
    def __init__(self, min_x, max_x, xp=xp_):
        self.min_x = min_x
        self.max_x = max_x
        self.xp = xp

    def __hash__(self):
        return hash((self.min_x, self.max_x))

    def __eq__(self, other):
        return (isinstance(other, XFilter)
                and self.min_x == other.min_x
                and self.max_x == other.max_x)

    def __call__(self, state: hoomd.state):
        if hoomd.version.gpu_enabled:
            local_snapshot = state.gpu_local_snapshot
        else:
            local_snapshot = state.cpu_local_snapshot
        with local_snapshot as snap:
            xs = snap.particles.position[:, 0]
            indices = ((self.xp.array(xs) > self.min_x)
                       & (self.xp.array(xs) < self.max_x))
            return np.copy(asnumpy(self.xp.array(snap.particles.tag[indices])))


def ifft_interpolate(y, n_interp_pts, axis=0, xp=np):
    # Compute the FFT along the specified axis
    fft_y = xp.fft.fft(y, axis=axis)

    # Determine the original number of points along the specified axis
    n_orig = fft_y.shape[axis]

    if n_interp_pts > n_orig:
        # Interpolation (make the signal more dense)

        # Determine padding sizes
        n_pad = (n_interp_pts - n_orig) // 2

        # Shift the FFT result
        interp_hat_y = xp.fft.fftshift(fft_y, axes=axis)

        # Create a padding shape that matches the dimensions of y
        pad_width = [(0, 0)] * y.ndim
        pad_width[axis] = (n_pad, n_pad)

        # Pad the FFT result
        interp_hat_y = xp.pad(interp_hat_y, pad_width, mode='constant')

        # Shift back the padded FFT result
        interp_hat_y = xp.fft.ifftshift(interp_hat_y, axes=axis)

        # Compute the inverse FFT to get the interpolated signal
        dense_y = xp.fft.ifft(interp_hat_y, axis=axis)

        # Scale the result to maintain the correct amplitude
        dense_y = xp.real(dense_y) * n_interp_pts / n_orig

    elif n_interp_pts < n_orig:
        # Decimation (make the signal less dense)

        # Determine truncation sizes
        n_trunc = (n_orig - n_interp_pts) // 2

        # Shift the FFT result
        trunc_hat_y = xp.fft.fftshift(fft_y, axes=axis)

        # Create a slicing shape to truncate the dimensions of y
        slice_obj = [slice(None)] * y.ndim
        slice_obj[axis] = slice(n_trunc, n_orig - n_trunc)

        # Truncate the FFT result
        trunc_hat_y = trunc_hat_y[tuple(slice_obj)]

        # Shift back the truncated FFT result
        trunc_hat_y = xp.fft.ifftshift(trunc_hat_y, axes=axis)

        # Compute the inverse FFT to get the decimated signal
        dense_y = xp.fft.ifft(trunc_hat_y, axis=axis)

        # Scale the result to maintain the correct amplitude
        dense_y = xp.real(dense_y) * n_interp_pts / n_orig

    else:
        # No change in the number of points
        dense_y = xp.real(xp.fft.ifft(fft_y, axis=axis))

    return dense_y


def casadi_fourier_series(n, domain_size=2*np.pi, funcname='series'):
    # Define symbolic variable
    x_sym = ca.MX.sym('x')
    c_sym = ca.MX.sym('c', 2*n - 1)
    # Initialize the sum
    sum_result = c_sym[0]
    for i in range(1, n):
        c = (2*np.pi*i) / domain_size
        sum_result += c_sym[i] * ca.cos(c * (x_sym-domain_size/2))
    for i, ii in enumerate(range(n, 2*n-1), start=1):
        c = (2*np.pi*i) / domain_size
        sum_result -= c_sym[ii] * ca.sin(c * (x_sym-domain_size/2))
    # sum_result /= 2 * np.pi * domain_size

    # Create a CasADi function for the sum
    fourier_series_func = ca.Function(funcname, [x_sym, c_sym], [sum_result])

    return fourier_series_func


def casadi_fourier_series_ps(n, domain_size=2*np.pi, funcname='series_w_phase_shift'):
    # Define symbolic variable
    x_sym = ca.MX.sym('x')
    c_sym = ca.MX.sym('c', 2*n - 1)
    ps_sym = ca.MX.sym('ps')
    # Initialize the sum
    sum_result = c_sym[0]
    for i in range(1, n):
        c = (2*np.pi*i) / domain_size
        sum_result += c_sym[i] * ca.cos(c * (x_sym - ps_sym - domain_size/2))
    for i, ii in enumerate(range(n, 2*n-1), start=1):
        c = (2*np.pi*i) / domain_size
        sum_result -= c_sym[ii] * ca.sin(c * (x_sym - ps_sym - domain_size/2))
    # sum_result /= 2 * np.pi * domain_size

    # Create a CasADi function for the sum
    fourier_series_func = ca.Function(
        funcname, [x_sym, c_sym, ps_sym], [sum_result])

    return fourier_series_func


def casadi_fourier_series_v(n, dt, domain_size=2*np.pi, funcname='series_w_vel'):
    # Define symbolic variable
    x = ca.MX.sym('x')
    sp0 = ca.MX.sym('sp0')
    v = ca.MX.sym('v')
    t = ca.MX.sym('t')
    c0 = ca.MX.sym('c0', 2*n - 1)
    c1 = ca.MX.sym('c0', 2*n - 1)

    # Interpolate the coefficients
    c = c0 + t * (c1 - c0)/dt
    sp = sp0 + v * t

    # Initialize the sum
    sum_result = c[0]
    for i in range(1, n):
        a = (2*np.pi*i) / domain_size
        sum_result += c[i] * ca.cos(a * (x - domain_size/2+ sp))
    for i, ii in enumerate(range(n, 2*n-1), start=1):
        a = (2*np.pi*i) / domain_size
        sum_result -= c[ii] * ca.sin(a * (x - domain_size/2+ sp))
    # sum_result /= 2 * np.pi * domain_size

    # Create a CasADi function for the sum
    fourier_series_func = ca.Function(
        funcname, [x, t, c0, c1, sp0, v], [sum_result])

    return fourier_series_func


def quad_x(x_pts, x_weights, n, dt, domain_size=2*np.pi, funcname='quad_x'):
    sp0 = ca.MX.sym('sp0')
    v = ca.MX.sym('v')
    t = ca.MX.sym('t')
    c0 = ca.MX.sym('c0', 2*n - 1)
    c1 = ca.MX.sym('c0', 2*n - 1)
    func = casadi_fourier_series_v(n, dt, domain_size)
    func_map = func.map(x_pts.shape[0])
    result = x_weights.T @ func_map(x_pts, t, c0, c1, sp0, v).T
    return ca.Function(funcname, [t, c0, c1, sp0, v], [result])


def penalty_integrand(min_trapped_frac, x_pts, x_weights, n, dt, domain_size=2*np.pi, funcname='penalty_integrand'):
    sp0 = ca.MX.sym('sp0')
    v = ca.MX.sym('v')
    t = ca.MX.sym('t')
    c0 = ca.MX.sym('c0', 2*n - 1)
    c1 = ca.MX.sym('c0', 2*n - 1)
    quad_x_fcn = quad_x(x_pts, x_weights, n, dt, domain_size)
    penalty = penalty_fcn(0)
    val = penalty(min_trapped_frac - quad_x_fcn(t, c0, c1, sp0, v))
    return ca.Function(funcname, [t, c0, c1, sp0, v], [val])

def quad_t(t_pts, t_weights,min_trapped_frac, x_pts, x_weights, n, dt, domain_size=2*np.pi, funcname='quad_t'):
    sp0 = ca.MX.sym('sp0')
    v = ca.MX.sym('v')
    c0 = ca.MX.sym('c0', 2*n - 1)
    c1 = ca.MX.sym('c0', 2*n - 1)
    func = penalty_integrand(min_trapped_frac, x_pts, x_weights, n, dt, domain_size)
    func_map = func.map(t_pts.shape[0])
    result = t_weights.T @ func_map(t_pts, c0, c1, sp0, v).T
    return ca.Function(funcname, [c0, c1, sp0, v], [result])

def quad_t_v2(t_pts, t_weights, min_trapped_frac, x_pts, x_weights, n, dt, domain_size=2*np.pi, funcname='quad_t'):
    sp0 = ca.SX.sym('sp0')
    v = ca.SX.sym('v')
    c0 = ca.SX.sym('c0', 2*n - 1)
    c1 = ca.SX.sym('c0', 2*n - 1)

    # Define symbolic variables
    x = ca.SX.sym('x')
    t = ca.SX.sym('t')
    c0_ = ca.SX.sym('c0', 2*n - 1)
    c1_ = ca.SX.sym('c0', 2*n - 1)

    # Interpolate the coefficients
    c = c0_ + t * (c1_ - c0_) / dt
    sp = sp0 + v * t

    # Initialize the sum
    sum_result = c[0]
    for i in range(1, n):
        a = (2 * np.pi * i) / domain_size
        sum_result += c[i] * ca.cos(a * (x - domain_size/2 + sp))
    for i, ii in enumerate(range(n, 2 * n - 1), start=1):
        a = (2 * np.pi * i) / domain_size
        sum_result -= c[ii] * ca.sin(a * (x - domain_size/2 + sp))

    # Create a CasADi function for the sum
    fourier_series_func = ca.Function('series_w_vel', [x, t, c0_, c1_, sp0, v], [sum_result])

    # Map the function
    func_map = fourier_series_func.map(x_pts.shape[0])
    quad_x_result = x_weights.T @ func_map(x_pts, t, c0, c1, sp0, v).T

    # Penalty function
    penalty = penalty_fcn(0)  # You need to define the penalty_fcn somewhere

    # Calculate penalty integrand
    penalty_integrand_val = penalty(min_trapped_frac - quad_x_result)

    # Map the penalty function
    penalty_integrand_func = ca.Function('penalty_integrand', [t, c0, c1, sp0, v], [penalty_integrand_val])
    penalty_integrand_map = penalty_integrand_func.map(t_pts.shape[0])

    # Calculate final result
    result = t_weights.T @ penalty_integrand_map(t_pts, c0, c1, sp0, v).T

    return ca.Function(funcname, [c0, c1, sp0, v], [result])


def penalty_fcn(eps, funcname='penalty'):
    x_sym = ca.MX.sym('x')
    # result = ca.sqrt(ca.fmax(x_sym, 0)**2 + eps)
    result = ca.fmax(x_sym, 0)**2
    return ca.Function(funcname, [x_sym], [result])

def quad_prod(x_pts, x_weights, n1, n2, domain_size=2*np.pi, funcname='quad_prod'):
    c1_sym = ca.MX.sym('c1', 2*n1 - 1)
    c2_sym = ca.MX.sym('c2', 2*n2 - 1)
    func1 = casadi_fourier_series(n1, domain_size)
    func2 = casadi_fourier_series(n2, domain_size)
    func_map1 = func1.map(x_pts.shape[0])
    func_map2 = func2.map(x_pts.shape[0])
    result = x_weights.T @ (func_map1(x_pts, c1_sym) * func_map2(x_pts, c2_sym)).T
    return ca.Function(funcname, [c1_sym, c2_sym], [result])

def quad_x_v2(x_pts, x_weights, n, domain_size=2*np.pi, funcname='quad_x'):
    sp0 = ca.MX.sym('sp0')
    c_sym = ca.MX.sym('c', 2*n - 1)
    func = casadi_fourier_series_ps(n, domain_size)
    func_map = func.map(x_pts.shape[0])
    result = x_weights.T @ func_map(x_pts, c_sym, -sp0).T
    return ca.Function(funcname, [c_sym, sp0], [result])

def ca_penalty_v2(min_trapped_frac, x_pts, x_weights, n, domain_size=2*np.pi, funcname='penalty_integrand'):
    sp0 = ca.MX.sym('sp0')
    c_sym = ca.MX.sym('c', 2*n - 1)
    quad_x_fcn = quad_x_v2(x_pts, x_weights, n, domain_size)
    penalty = penalty_fcn(0)
    val = penalty(min_trapped_frac - quad_x_fcn(c_sym, sp0))
    return ca.Function(funcname, [c_sym, sp0], [val])

def ca_state_func(x_pts, n, domain_size=2*np.pi, funcname='max_x'):
    c_sym = ca.MX.sym('c', 2*n - 1)
    func = casadi_fourier_series(n, domain_size)
    func_map = func.map(x_pts.shape[0])
    result = func_map(x_pts, c_sym)
    return ca.Function(funcname, [c_sym], [result])

def ca_state_penalty(max_val, x_pts, n, domain_size=2*np.pi, funcname='state_penalty'):
    c_sym = ca.MX.sym('c', 2*n - 1)
    state_func = ca_state_func(x_pts, n, domain_size)
    state = state_func(c_sym)
    penalty = penalty_fcn(0)
    val = ca.sum2(penalty(state-max_val))
    return ca.Function(funcname, [c_sym], [val])

def integrated_fourier_series(n, domain_size=2*np.pi, funcname='integral'):
    # Define symbolic variable
    x_sym = ca.MX.sym('x')
    c_sym = ca.MX.sym('c', 2*n - 1)
    sum_result = c_sym[0] * x_sym
    for i in range(1, n):
        c = (2*np.pi*i) / domain_size
        sum_result += 1/c * c_sym[i] * ca.sin(c * x_sym)
    for i, ii in enumerate(range(n, 2*n-1), start=1):
        c = (2*np.pi*i) / domain_size
        sum_result -= 1/c * c_sym[ii] * ca.cos(c * x_sym)
    # sum_result /= 2 * np.pi * domain_size

    # Create a CasADi function for the integral
    integrated_series_func = ca.Function(
        funcname, [x_sym, c_sym], [sum_result])

    return integrated_series_func


def step_function(n_steps, funcname='step_func'):
    # Define symbolic variable
    x_sym = ca.MX.sym('x')
    step_locs = ca.MX.sym('p', n_steps)
    step_sizes = ca.MX.sym('s', n_steps)

    # Initialize the step function to zero
    step_func_expr = 0

    # Add steps at specified locations with specified sizes
    for i in range(n_steps):
        step_func_expr += ca.if_else(x_sym >= step_locs[i], step_sizes[i], 0)

    # Create a CasADi function for the step function
    step_func = ca.Function(
        funcname, [x_sym, step_locs, step_sizes], [step_func_expr])

    return step_func


def solve_tridiagonal(A, b):
    """
    Solve a tridiagonal system of equations Ax = b using the Thomas algorithm.
    """
    n = len(b)
    c = A[0, 1:]
    d = A[0, 0]
    x = np.zeros(n)
    c[0] /= d
    b[0] /= d
    for i in range(1, n-1):
        d = A[i, i] - A[i, i-1] * c[i-1]
        c[i] /= d
        b[i] = (b[i] - A[i, i-1] * b[i-1]) / d
    b[-1] = (b[-1] - A[-1, -2] * b[-2]) / (A[-1, -1] - A[-1, -2] * c[-2])
    x[-1] = b[-1]
    for i in range(n-2, -1, -1):
        x[i] = b[i] - c[i] * x[i+1]
    return x

def count_trapped_pcls(sp, trap_width, plate_gap, pcl_pos, xp=xp_):
    m_edge = periodic_clip(sp - trap_width / 2, -plate_gap / 2, plate_gap / 2)
    p_edge = periodic_clip(sp + trap_width / 2, -plate_gap / 2, plate_gap / 2)
    extra = xp.zeros(2)
    # extra[int(True)] captures the extra pcls that are trapped by the periodic boundary
    extra[int((xp.sign(p_edge) != xp.sign(sp)) & (sp > 0))] = xp.sum(pcl_pos < p_edge)
    extra[int((xp.sign(m_edge) != xp.sign(sp)) & (sp < 0))] = xp.sum(pcl_pos > m_edge)
    n_trapped_pcls = (xp.sum(xp.abs(pcl_pos - sp) < trap_width / 2) + extra[1])
    return n_trapped_pcls


def count_trapped_pcls_fre(sp, trap_width, plate_gap, pcl_pos, xp=xp_):
    """Counts trapped particles from right edge""" 
    m_edge = periodic_clip(sp - trap_width, -plate_gap / 2, plate_gap / 2)
    extra = xp.zeros(2)
    # extra[int(True)] captures the extra pcls that are trapped by the periodic boundary
    extra[int((xp.sign(m_edge) != xp.sign(sp)) & (sp < 0))] = xp.sum(pcl_pos > m_edge)
    n_trapped_pcls = (xp.sum(xp.abs(pcl_pos - sp + trap_width/ 2 ) < trap_width / 2) + extra[1])
    return n_trapped_pcls


def my_spline(data, spline_order, knot_dt):
    ts, xs, us = data.ts, data.xs, data.us
    knots = np.arange(ts[0], ts[-1] + knot_dt, knot_dt)
    k = spline_order
    m = 1
    ext_knots = augknt(knots, k, m)
    step_times = ts[np.where((us[1:] != us[:-1]).any(axis=1))[0] + 1]
    ext_knots = np.sort(np.hstack(
        (ext_knots,
            step_times.repeat(k - 1))))
    spline = make_lsq_spline(ts, xs, ext_knots, k=k)
    return spline