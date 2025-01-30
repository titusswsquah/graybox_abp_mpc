import numpy as np
import hoomd
from lib.hd_simulator import Simulator
from lib.hd_neural_ode import MultipleStepWrapper, ControlWrapper, ForceWrapper, FluxWrapper
import mpctools as mpc
import casadi as cs
from time import time
from scipy.interpolate import interp1d
from lib import utils
from lib.torch_casadi import TorchEvaluatorV3
from lib import hd_utils
from lib.hd_utils import BaseController, keys_to_array, casadi_fourier_series, evaluate_fourier_series
from lib.utils import array_module, asnumpy, ascupy
from scipy.interpolate import make_lsq_spline
import copy
xp = array_module('cupy')


class SetPoint:
    def __init__(self, env: Simulator):
        self.center = env.center
        self.center1 = env.center1
        self.center2 = env.center2
        self.trigger_times = env.trigger_times
        self.sim_time = env.ctrl_sim_time
        self.w001_max = env.w001_max
        self.amp = env.amp
        self.sp_step_sizes = env.sp_step_sizes

        self.fc_start = env.fc_start
        self.fc_weight = env.fc_weight
        self.fc_amp = env.fc_amp
        self.fc_period = env.fc_period

    def sp_fcn(self, t):
        trigger_times = self.trigger_times
        sp_step_sizes = self.sp_step_sizes
        size1 = None
        size2 = None
        for i in range(len(trigger_times) - 1):
            if trigger_times[i] <= t < trigger_times[i + 1]:
                size1 = sp_step_sizes[i][0]
                size2 = sp_step_sizes[i][1]
        if size1 is None:
            size1 = sp_step_sizes[-1][0]
            size2 = sp_step_sizes[-1][1]
        if t < self.fc_start:
            force_weight = 0
        else:
            force_weight = self.fc_weight
        force_sp = self.fc_amp * \
            np.sin(2 * np.pi * (t-self.fc_start) / self.fc_period)
        
        force_sp2 = -np.floor(t / 10) * 0.1

        return size1, size2, self.center1, self.center2, force_weight, force_sp, force_sp2


class ControlSimulator(Simulator):
    def __init__(self,
                 pars_dict,
                 ):
        super().__init__(**pars_dict)
        meas_time_pad = self.meas_time_pad
        buffer_time = (self.n_p + 1) * self.nnam_dt + meas_time_pad
        self.n_buffer_steps = round(buffer_time / self.ctrl_samp_dt) + 1
        knots = np.linspace(0, buffer_time, round(
            buffer_time / self.nnam_dt) + 1)
        self.ext_knots = utils.augknt(knots, self.n_time_pts, self.n_time_pts)
        self.mini_ts = np.linspace(0, buffer_time, self.n_buffer_steps)

        self.time_pts = utils.asnumpy(self.time_pts)

        self.nnam_ts = knots[knots > meas_time_pad][::-1]
        self.nnam_aug_ts = (
            self.nnam_ts[:, np.newaxis] + self.time_pts[:0:-1]) - self.nnam_dt
        self.set_point = SetPoint(self)

        self.y_buffer0 = None
        self.u_buffer0 = None

    def init_frames_to_buffers(self,
                               init_state_frames):
        y_buffer = []
        u_buffer = []
        state_keys = ['pr000']
        ctrl_keys = list(self.ctrl_info_dict.keys())
        for i, frame in enumerate(init_state_frames[-self.n_buffer_steps:]):
            if isinstance(self.model, str):
                y_buffer.append(keys_to_array(
                    state_keys, frame.log, cutoff=self.n_state_modes+1, xp=np))
            else:
                pass
                # state = []
                # for key in self.full_state_keys:
                #     state.append(frame.log[key])
                # y_buffer.append(state)
            if i % round(self.nnam_dt / self.ctrl_samp_dt) == 0:
                u_buffer.append(keys_to_array(ctrl_keys, frame.log, xp=np))
        y_buffer = np.array(y_buffer)
        u_buffer = np.array(u_buffer)[-self.n_p - 1:-1]
        return y_buffer, u_buffer

    def get_yz0(self, y_buffer, u_buffer):
        """
        y_buffer and u_buffer run from oldest to newest
        This function is used to get the initial state for the MPC controller. It takes the buffer of measurements and
        control inputs, creates a least squares spline fit for the measurements, evaluates the spline at the required
        points, and then reshapes and concatenates the measurements and control inputs to form the initial state.

        Parameters:
        y_buffer (numpy.ndarray): A buffer that contains the recent measurements. The measurements are stored in a
                                  numpy array where each row corresponds to a measurement at a different time step.
        u_buffer (numpy.ndarray): A buffer that contains the recent control inputs. The control inputs are stored in a
                                  numpy array where each row corresponds to a control input at a different time step.

        Returns:
        yz0 (numpy.ndarray): The initial state for the MPC controller. It is a numpy array that contains the reshaped
                             measurements and control inputs.
        """
        yz0 = None
        if self.use_spline:
            spline_ts = self.mini_ts
            spline_y = y_buffer
        else:
            dt = self.nnam_dt
            ts = self.mini_ts
            sample_ts = (
                (np.arange(0, ts[-1] + dt, dt)[:, None] + self.time_pts[:-1])).flatten()
            t_mask = utils.find_close_indices_vectorized(
                ts, sample_ts, atol=(ts[1]-ts[0])/2)
            spline_ts = ts[t_mask]
            spline_y = y_buffer[t_mask]
        spline = make_lsq_spline(
            spline_ts, spline_y, self.ext_knots, k=self.n_time_pts)
        fit_curves_matrices = spline(self.nnam_aug_ts)
        if isinstance(self.model, str):
            nnam_xs = fit_curves_matrices
            nnam_ys = (nnam_xs[:, :, 0, :]).reshape(nnam_xs.shape[0], -1)
            yz0 = np.hstack((nnam_ys.flatten(), u_buffer[::-1].flatten()))
        else:
            pass
            # nnam_xs = legval(self.norm_oa_pts[:self.n_state_pts - 2],
            #                  fit_curves_matrices.swapaxes(0, -1)).squeeze()
            # yz0 = nnam_xs.flatten()
        return yz0

    def create_ctrl_gsd(self,
                        init_gsd_path,
                        init_state_frames,
                        nnam_pars_dict,
                        dest_gsd,
                        stagecost=None,
                        termcost=None
                        ):
        # Number of time steps in the entire training and validation data.
        if hoomd.version.gpu_enabled:
            device = hoomd.device.GPU()
        else:
            device = hoomd.device.CPU()
        sim = hoomd.Simulation(device=device, seed=self.md_seed)
        sim.timestep = 0
        sim.create_state_from_gsd(filename=init_gsd_path)
        integrator = hoomd.md.Integrator(dt=self.md_dt,
                                         integrate_rotational_dof=True)
        brownian = hoomd.md.methods.Brownian(kT=self.kbt,
                                             filter=hoomd.filter.All())
        brownian.gamma.default = self.zeta_t
        brownian.gamma_r.default = [self.zeta_r, self.zeta_r, self.zeta_r]
        integrator.methods.append(brownian)
        sim.operations.integrator = integrator
        active1 = hoomd.md.force.Active(filter=hoomd.filter.All())
        active1.active_force['A'] = (self.fp, 0, 0)
        integrator.forces.append(active1)
        if self.interact:
            cell = hoomd.md.nlist.Cell(buffer=0.4)
            lj_pcl = hoomd.md.pair.LJ(nlist=cell)
            lj_pcl.params[('A', 'A')] = dict(epsilon=self.epsilon,
                                             sigma=(2 * self.pcl_rad
                                                    / (2 ** (1 / 6))))
            lj_pcl.r_cut[('A', 'A')] = (2 * self.pcl_rad)
            integrator.forces.append(lj_pcl)

        snapshot = sim.state.get_snapshot()
        pcl_pos = xp.clip(
            snapshot.particles.position[:, 1],
            -self.plate_gap / 2,
            self.plate_gap / 2)
        pcl_quarts = xp.asarray(snapshot.particles.orientation)
        pcl_thetas = \
            hd_utils.gpu_quaternion_to_euler_angle_vectorized3(
                pcl_quarts[:, 0],
                pcl_quarts[:, 3], xp)
        if self.track_force:
            forces = xp.zeros(snapshot.particles.N)
        else:
            forces = None

        full_state0 = hd_utils.samp_to_fourier(xp.asarray(pcl_pos),
                                               pcl_thetas,
                                               self.n_state_modes + 1,
                                               self.meas_order,
                                               self.bounds,
                                               forces=forces
                                               )
        if isinstance(self.model, str):
            model = MultipleStepWrapper(nnam_pars_dict['xuyscales'],
                                        'cpu',
                                        self).to('cpu')
            model.load_state_dict(nnam_pars_dict['state_dict'])
        else:
            model = self.model
        (self.y_buffer0,
         self.u_buffer0) = self.init_frames_to_buffers(init_state_frames)
        yz0 = self.get_yz0(self.y_buffer0,
                           self.u_buffer0)
        policy = MPCControlPolicy(self,
                                  model,
                                  yz0,
                                  'cpu',
                                  stagecost=stagecost,
                                  termcost=termcost)
        hoomd_controller = MPCController(self,
                                         full_state0,
                                         policy)
        integrator.forces.append(hoomd_controller)
        logger = hoomd.logging.Logger()
        logger['wr001r'] = (hoomd_controller, 'wr001r', 'sequence')
        logger['wr001i'] = (hoomd_controller, 'wr001i', 'sequence')
        logger['solve_status'] = (hoomd_controller, 'solve_status', 'string')
        logger['solve_time'] = (hoomd_controller, 'solve_time', 'scalar')
        logger['stage_cost'] = (hoomd_controller, 'stage_cost', 'scalar')
        logger['n_pcls_left'] = (hoomd_controller, 'n_pcls_left', 'scalar')
        logger['left_msd'] = (hoomd_controller, 'left_msd', 'scalar')
        logger['right_msd'] = (hoomd_controller, 'right_msd', 'scalar')
        for key in self.full_state_keys:
            logger[key] = (hoomd_controller, key, 'sequence')
        for horizon in range(self.n_horizon + 1):
            logger[f'x{horizon}'] = (
                hoomd_controller, f'x{horizon}', 'sequence')
        for horizon in range(self.n_horizon):
            logger[f'u{horizon}'] = (
                hoomd_controller, f'u{horizon}', 'sequence')

        if 'none' in str(self.xfilter_max).lower() or 'none' in str(self.xfilter_min).lower():
            gsd_writer1 = hoomd.write.GSD(
                filename=dest_gsd,
                trigger=hoomd.trigger.Periodic(
                    int(xp.round(1 / (self.md_dt
                                      / self.md_samp_dt)))),
                dynamic=['property'],
                mode='wb',
                filter=hoomd.filter.Null())
        elif 'all' in str(self.xfilter_max).lower() or 'all' in str(self.xfilter_min).lower():
            gsd_writer1 = hoomd.write.GSD(
                filename=dest_gsd,
                trigger=hoomd.trigger.Periodic(
                    int(xp.round(1 / (self.md_dt
                                      / self.md_samp_dt)))),
                dynamic=['property'],
                mode='wb')
        else:
            filter_ = hd_utils.XFilter(self.xfilter_min,
                                       self.xfilter_max,
                                       xp=xp)
            filter_updater = hoomd.update.FilterUpdater(hoomd.trigger.Periodic(
                int(xp.round(1 / (self.md_dt / self.md_samp_dt)))),
                [filter_])
            sim.operations.updaters.append(filter_updater)
            gsd_writer1 = hoomd.write.GSD(filename=dest_gsd,
                                          trigger=hoomd.trigger.Periodic(
                                              int(xp.round(1 / (self.md_dt
                                                                / self.md_samp_dt)))),
                                          mode='wb',
                                          dynamic=['property'],
                                          filter=filter_)

        gsd_writer1.log = logger
        sim.operations.writers.append(gsd_writer1)

        sim.run(self.ctrl_sim_time * ((xp.round(1 / self.md_dt))))
        return None


class MPCControlPolicy:
    def __init__(self,
                 env: ControlSimulator,
                 model: MultipleStepWrapper or int,
                 yz0,
                 device: str,
                 stagecost=None,
                 termcost=None,
                 ):
        self.env = env
        n_state_modes = env.n_state_modes
        n_ctrl_modes = int((env.n_u + 1)/2)
        plate_gap = env.plate_gap
        self.constrained_pts = np.linspace(-env.plate_gap / 2,
                                           env.plate_gap / 2,
                                           env.n_constrained_pts)
        self.quad_pts, _, _, self.quad_ig = utils.colloc(env.n_quad_pts,
                                                         plate_gap=env.plate_gap,
                                                         shifted=False)
        self.step_func = hd_utils.step_function(2)
        self.half_pts, _, _, self.half_ig = utils.colloc(env.n_quad_pts,
                                                         plate_gap=env.plate_gap/2,
                                                         shifted=True)
        self.state_penalty_func = hd_utils.ca_state_penalty(0.22,
                                                            self.constrained_pts,
                                                            env.n_state_modes+1,
                                                            env.plate_gap
                                                            )
        
        self.quad_prod_func = hd_utils.quad_prod(self.quad_pts,
                                                 self.quad_ig,
                                                 env.n_state_modes+1,
                                                 env.n_force_modes+1,
                                                 env.plate_gap
                                                 )

        if stagecost is None:
            self.state_func = hd_utils.integrated_fourier_series(
                n_state_modes + 1, plate_gap)
            self.stagecost = (lambda x, u, x_sp, u_sp, Deltau:
                              cdf_stagecost(self, x, u, x_sp, u_sp, Deltau))
        else:
            self.state_func = hd_utils.casadi_fourier_series(n=n_state_modes + 1,
                                                             domain_size=env.plate_gap)
            self.stagecost = (lambda x, u, x_sp, u_sp, Deltau:
                              stagecost(self, x, u, x_sp, u_sp, Deltau))
        if termcost is None:
            self.termcost = (lambda x, x_sp: cdf_termcost(self, x, x_sp))
        else:
            self.termcost = (lambda x, x_sp: termcost(self, x, x_sp))

        self.cs_model = None
        if isinstance(env.model, int):
            pass
        else:
            casadi_type = 'MX'
            torch_state_space_model = ControlWrapper(env.nnam_dt,
                                                     device,
                                                     model
                                                     )
            torch_force_model = ForceWrapper(env.nnam_dt,
                                             device,
                                             model
                                             )
            
            torch_flux_model = FluxWrapper(env.nnam_dt,
                                             device,
                                             model
                                             )
            n_vars = {"x": env.n_g + env.n_g * env.n_p + env.n_u * env.n_p,
                      "u": env.n_u, "t": env.n_horizon,
                      "e": env.n_constrained_pts}
            self.cs_model = TorchEvaluatorV3(
                torch_state_space_model,
                [[n_vars['x'], 1],
                 [n_vars['u'], 1]],
                [[n_vars['x'], 1]], device)

            self.force_model = TorchEvaluatorV3(
                torch_force_model,
                [[n_vars['x'], 1],
                    [n_vars['u'], 1]],
                [[1, 1]], device)
            
            self.flux_model = TorchEvaluatorV3(
                torch_flux_model,
                [[n_vars['x'], 1],
                    [n_vars['u'], 1]],
                [[torch_flux_model.M, 1]], device)
            
            # self.force_model = mpc.getCasadiFunc(force_model,
            #                            [n_vars['x'], n_vars['u']],
            #                            ["x", "u"],
            #                            funcname="force",
            #                            casaditype=casadi_type)

            f_step = mpc.getCasadiFunc(self.cs_model,
                                       [n_vars['x'], n_vars['u']],
                                       ["x", "u"],
                                       funcname="f_step",
                                       casaditype=casadi_type)
        e = mpc.getCasadiFunc(
            lambda x, u: self.constraints_fcn(
                x, u, n_ctrl_modes, self.constrained_pts),
            [n_vars['x'], n_vars['u']],
            ["x", "u"],
            "e")
        ell = mpc.getCasadiFunc(self.stagecost,
                                [n_vars['x'],
                                 n_vars['u'],
                                 n_vars['x'],
                                 n_vars['u'],
                                 n_vars['u'], ],
                                ["x", "u", "x_sp", "u_sp", "Du"],
                                funcname="l",
                                casaditype='MX')

        ell_f = mpc.getCasadiFunc(self.termcost,
                                  [
                                      n_vars['x'],
                                      n_vars['x'],
                                  ],
                                  ["x", "x_sp"], funcname="Pf",
                                  casaditype='MX')

        # lb = dict(u=-env.w001_max * np.ones(env.n_u))
        # ub = dict(u=env.w001_max * np.ones(env.n_u))

        funcargs = {"fode": ["x", "u"],
                    "l": ["x", "u", "x_sp", "u_sp", "Du"],
                    "Pf": ["x", "x_sp"],
                    "e": ["x", "u"],
                    }
        sp = dict(x=np.zeros(n_vars['x']), u=np.ones(env.n_u))

        guess_x = [yz0]
        for _ in range(env.n_horizon):
            guess_x.append(self.cs_model(
                guess_x[-1], np.zeros(env.n_u)).full().flatten())
        guess = dict(x=guess_x, u=np.ones((env.n_horizon, env.n_u)))

        if isinstance(env.model, int):
            nmpcargs = {
                "f": f_step,
                "l": ell,
                "Pf": ell_f,
                "N": n_vars,
                "x0": yz0,
                "Delta": env.nnam_dt,
                "verbosity": 3,
                "funcargs": funcargs,
                "sp": sp,
                "timelimit": int(1e6),
                "e": e,
                'uprev': np.ones(env.n_u),
                'casaditype': 'SX'
            }
        else:
            nmpcargs = {
                "f": f_step,
                "l": ell,
                "Pf": ell_f,
                "N": n_vars,
                "x0": yz0,
                "verbosity": 5,
                "funcargs": funcargs,
                "sp": sp,
                "timelimit": int(1e6),
                "e": e,
                'uprev': np.ones(env.n_u),
                'casaditype': 'MX',
                'guess': guess,
            }
        self.policy = mpc.nmpc(**nmpcargs)
        if self.env.force_control:
            solveroptions = dict(max_iter=5000,
                                 max_cpu_time=int(1e6),
                                 hessian_approximation="exact",
                                 acceptable_tol=1e-5,
                                 acceptable_iter=15,
                                 acceptable_dual_inf_tol=1e10,
                                 acceptable_constr_viol_tol=1e-2,
                                 acceptable_compl_inf_tol=1e-2,
                                 acceptable_obj_change_tol=1e-4,
                                #  linear_solver='ma27',
                                #  mu_strategy='adaptive',
                                #  ma27_pivtol=1e-7
                                 )
        else:
            solveroptions = dict(max_iter=5000,
                                 max_cpu_time=int(1e6),
                                 hessian_approximation="exact",
                                 acceptable_tol=1e-5,
                                 acceptable_iter=15,
                                 acceptable_dual_inf_tol=1e10,
                                 acceptable_constr_viol_tol=1e-2,
                                 acceptable_compl_inf_tol=1e-2,
                                 acceptable_obj_change_tol=1e-4,
                                 )
        self.policy.initialize(solveroptions=solveroptions)

    def constraints_fcn(self, x, u, n_ctrl_modes, constrained_pts):
        u_fcn = casadi_fourier_series(
            n_ctrl_modes, domain_size=self.env.plate_gap, funcname='constraints')
        constrained_vals = cs.vertcat(
            *[u_fcn(pt, u)**2 - self.env.w001_max**2 for pt in constrained_pts])
        return constrained_vals

    def odefunc(self, x, u):
        raise (NotImplementedError)


class MPCController(BaseController):
    def __init__(self,
                 env: ControlSimulator,
                 full_state0,
                 control_policy: MPCControlPolicy):
        super().__init__(env, full_state0)
        self.env = env
        self.control_policy = control_policy
        self.controller = control_policy.policy
        self.use_spline = env.use_spline
        self._solve_status = ''
        self._solve_time = 0
        self._stage_cost = 0
        self._ctrl_cost = 0
        self._n_pcls_left = xp.zeros(1)
        self._left_msd = xp.zeros(1)
        self._right_msd = xp.zeros(1)
        self.sim_dt = env.md_dt
        self.sample_dt = env.ctrl_samp_dt
        self.md_ctrl_dt = env.md_ctrl_dt
        self.ctrl_dt = env.nnam_dt
        self.n_time_pts = env.n_time_pts
        self.n_p = env.n_p
        self.force_control = env.force_control
        self.fc_tol = env.fc_tol

        self.track_force = env.track_force
        self.y_forces = None
        self.alpha = self.sim_dt / self.sample_dt

        self.zeta_r = env.zeta_r
        meas_time_pad = env.meas_time_pad
        buffer_time = (env.n_p + 1) * env.nnam_dt + meas_time_pad
        n_buffer_steps = round(buffer_time / env.ctrl_samp_dt) + 1

        self.y_buffer = env.y_buffer0
        self.u_buffer = env.u_buffer0
        self.model = env.model
        self.full_state_keys = env.full_state_keys

        self._wr001_coeffs = np.zeros(env.n_ctrl_modes, dtype=np.complex128)

        knots = np.linspace(0, buffer_time, round(
            buffer_time / env.nnam_dt) + 1)
        self.ext_knots = utils.augknt(knots, env.n_time_pts, env.n_time_pts)
        self.mini_ts = np.linspace(0, buffer_time, n_buffer_steps)
        self.nnam_aug_ts = (knots[:, np.newaxis] +
                            env.time_pts[:0:-1]) - env.nnam_dt

        self.time_pts = utils.asnumpy(env.time_pts)
        self.bounds = (-env.plate_gap / 2, env.plate_gap / 2)

        self.nnam_ts = knots[knots > meas_time_pad]
        self.nnam_aug_ts = (
            self.nnam_ts[:, np.newaxis] + self.time_pts[:0:-1]) - self.ctrl_dt

        self.n_x = self.controller.misc['N']['x']
        self.n_u = self.controller.misc['N']['u']
        self.n_horizon = env.n_horizon
        self.sp_fcn = env.set_point.sp_fcn

        self.get_yz0 = env.get_yz0

        for horizon in range(env.n_horizon + 1):
            setattr(self, f'x{horizon}', np.zeros(self.n_x))
        for horizon in range(env.n_horizon):
            setattr(self, f'u{horizon}', np.ones(self.n_u))

    def w_ctrl_fcn(self, z):
        wr001 = evaluate_fourier_series(z.flatten(),
                                        ascupy(self._wr001_coeffs).flatten(),
                                        self.bounds, xp)
        return wr001

    def mpc_solve(self, timestep):
        sim_time = timestep * self.sim_dt
        yz0 = self.get_yz0(self.y_buffer[::-1],
                           self.u_buffer[::-1])
        self.controller.fixvar("x", 0, yz0)
        print(timestep)
        for i in range(self.n_horizon + 1):
            rel_t = sim_time + (i * self.ctrl_dt)
            mini_sps = self.sp_fcn(rel_t)
            dummy_sp = np.zeros(self.n_x)
            for j in range(len(mini_sps)):
                dummy_sp[j] = mini_sps[j]
            self.controller.par["x_sp", i] = dummy_sp
        toc = time()
        if self.force_control:
            base_u_guess = np.zeros(self.env.n_u)
            base_u_guess[0] = self.env.w001_max
            guess_xs = [yz0]
            guess_us = []
            for i in range(1, self.env.n_horizon):
                if np.any(np.isnan(self.controller.var["u", i].full().flatten())):
                    guess_us.append(base_u_guess)
                else:
                    mod_u_guess = np.zeros(self.env.n_u)
                    mod_u_guess[:5] = self.controller.var["u", i].full().flatten()[:5]
                    guess_us.append(mod_u_guess)
                guess_xs.append(self.control_policy.cs_model(guess_xs[-1],
                                                            guess_us[-1]
                                                            ).full().flatten())
                if i == self.env.n_horizon - 1:
                    if np.any(np.isnan(self.controller.var["u", i].full().flatten())):
                        guess_us.append(base_u_guess)
                    else:
                        guess_us.append(mod_u_guess)
                    guess_xs.append(self.control_policy.cs_model(guess_xs[-1],
                                                                guess_us[-1]
                                                                ).full().flatten())
            guess_xs = np.array(guess_xs)
            guess_us = np.array(guess_us)
            guess_Dus = guess_us[1:] - guess_us[:-1]
            guess_Dus = np.vstack((guess_Dus, np.zeros(self.env.n_u)))
            guess1 = dict(x=guess_xs,
                         u=guess_us,
                         Du=guess_Dus)
            guess_xs = [yz0]
            guess_us = np.zeros((self.env.n_horizon, self.env.n_u))
            for i in range(self.env.n_horizon):
                guess_xs.append(self.control_policy.cs_model(guess_xs[-1],
                                                             guess_us[i]
                                                             ).full().flatten())
            guess_xs = np.array(guess_xs)
            guess_Dus = np.zeros((self.env.n_horizon, self.env.n_u))

            guess2 = dict(x=guess_xs,
                         u=guess_us,
                         Du=guess_Dus)
            self.controller.saveguess(guess2)
            self.controller.solve()
            if 'Invalid_Number_Detected' in self.controller.stats["status"]:
                raise (ValueError("Invalid number detected in MPC controller."))
            if 'Solved_To_Acceptable_Level' in self.controller.stats["status"]:
                print('Solved to acceptable level')
            if 'Restoration_Failed' in self.controller.stats["status"]:
                self.controller.saveguess(guess1)
                self.controller.solve()
                if 'Invalid_Number_Detected' in self.controller.stats["status"]:
                    raise (ValueError(
                        "Invalid number detected in MPC controller."))
                elif 'Restoration_Failed' in self.controller.stats["status"]:
                    raise (ValueError("Restoration failed in MPC controller."))
                if 'Solved_To_Acceptable_Level' in self.controller.stats["status"]:
                    print('Solved to acceptable level')
            if self.fc_tol: 
                if self.controller.obj > self.fc_tol:
                    self.controller.saveguess(guess1)
                    self.controller.solve()
                    if 'Invalid_Number_Detected' in self.controller.stats["status"]:
                        raise (ValueError(
                            "Invalid number detected in MPC controller."))
                    elif 'Restoration_Failed' in self.controller.stats["status"]:
                        raise (ValueError("Restoration failed in MPC controller."))
                    if 'Solved_To_Acceptable_Level' in self.controller.stats["status"]:
                        print('Solved to acceptable level')
        else:
            self.controller.solve()
        if 'Invalid_Number_Detected' in self.controller.stats["status"]:
            raise (ValueError("Invalid number detected in MPC controller."))
        elif 'Restoration_Failed' in self.controller.stats["status"]:
            raise (ValueError("Restoration failed in MPC controller."))
        if 'Solved_To_Acceptable_Level' in self.controller.stats["status"]:
            print('Solved to acceptable level')
        tic = time()
        self.controller.saveguess(default=False)
        self._solve_time = tic - toc
        self._solve_status = self.controller.stats["status"]
        self._stage_cost = np.squeeze(self.control_policy.stagecost(
            np.squeeze(self.controller.var["x", 0]),
            np.squeeze(self.controller.var["u", 0]),
            np.squeeze(self.controller.par["x_sp", 0]),
            np.squeeze(self.controller.par["x_sp", 0]),
            np.squeeze(self.controller.var["Du", 0]),
        ))
        for horizon in range(self.n_horizon + 1):
            setattr(self, f'x{horizon}',
                    np.squeeze(self.controller.var["x", horizon]))
        for horizon in range(self.n_horizon):
            setattr(self, f'u{horizon}',
                    np.squeeze(self.controller.var["u", horizon]))
        if 'Infeasible_Problem_Detected' not in self.controller.stats["status"]:
            root_vals = np.squeeze(self.controller.var["u", 0]).reshape(-1, 1)
            if self.n_p > 0:
                self.u_buffer = np.roll(self.u_buffer, 1, axis=0)
                self.u_buffer[0] = root_vals.flatten()
            real = root_vals[:self.env.n_ctrl_modes]
            imag = root_vals[self.env.n_ctrl_modes:]
            #  pad imag with one zero on the left
            imag = np.concatenate((np.zeros((1, 1)), imag))
            self._wr001_coeffs = real + 1j * imag

    # @profile
    def set_forces(self, timestep):
        if hoomd.version.gpu_enabled:
            local_snapshot = self._state.gpu_local_snapshot
            force_arrays = self.gpu_local_force_arrays
        else:
            local_snapshot = self._state.cpu_local_snapshot
            force_arrays = self.cpu_local_force_arrays
        if self.track_force:
            with local_snapshot as snapshot:
                pcl_rtags = snapshot.particles.rtag
                new_val = xp.array(snapshot.particles.net_force[:, 1], copy=False)[
                    pcl_rtags]
                if self.y_forces is None:
                    self.y_forces = new_val
                else:
                    self.y_forces = self.alpha * new_val + \
                        (1 - self.alpha) * self.y_forces

        if timestep % round(self.md_ctrl_dt / self.sim_dt) == 0:
            if timestep % round(self.sample_dt / self.sim_dt) == 0:
                pcl_pos, pcl_thetas = self.measure(
                    evalulate_meas_vars=True, forces=self.y_forces)
                self.update_pcl_vars(pcl_pos)
                self.y_buffer = np.roll(self.y_buffer, 1, axis=0)
                if isinstance(self.model, str):
                    self.y_buffer[0] = asnumpy(keys_to_array(['pr000'],
                                                             self.full_state,
                                                             cutoff=self.env.n_state_modes + 1,
                                                             ))
                else:
                    state = []
                    for key in self.full_state_keys:
                        state.append(self.full_state[key])
                    self.y_buffer[0] = np.array(state)
            else:
                pcl_pos, pcl_thetas = self.measure(
                    evalulate_meas_vars=False, forces=self.y_forces)
                self.update_pcl_vars(pcl_pos)
            if timestep % round(self.ctrl_dt / self.sim_dt) == 0:
                self.mpc_solve(timestep)
            with force_arrays as arrays:
                arrays.torque[:, -1] = self.w_ctrl_fcn(pcl_pos) * xp.cos(
                    pcl_thetas) * self.zeta_r
        elif timestep % round(self.sample_dt / self.sim_dt) == 0:
            pcl_pos, pcl_thetas = self.measure(
                evalulate_meas_vars=True, forces=self.y_forces)
            self.update_pcl_vars(pcl_pos)
            self.y_buffer = np.roll(self.y_buffer, 1, axis=0)
            if isinstance(self.model, str):
                self.y_buffer[0] = asnumpy(keys_to_array(['pr000'],
                                                         self.full_state,
                                                         cutoff=self.env.n_state_modes + 1,
                                                         ))
            else:
                state = []
                for key in self.full_state_keys:
                    state.append(self.full_state[key])
                self.y_buffer[0] = np.array(state)
            if timestep % round(self.ctrl_dt / self.sim_dt) == 0:
                self.mpc_solve(timestep)

    def update_pcl_vars(self, pcl_pos):
        self._n_pcls_left = xp.sum(pcl_pos < 0)
        self._left_msd = xp.mean((pcl_pos[pcl_pos < 0] - self.env.center1)**2)
        self._right_msd = xp.mean(
            (pcl_pos[pcl_pos >= 0] - self.env.center2)**2)

    @property
    def wr001r(self):
        return utils.asnumpy(self._wr001_coeffs).real

    @property
    def wr001i(self):
        return utils.asnumpy(self._wr001_coeffs).imag

    @property
    def solve_time(self):
        return self._solve_time

    @property
    def stage_cost(self):
        return self._stage_cost

    @property
    def solve_status(self):
        return self._solve_status

    @property
    def n_pcls_left(self):
        return float(utils.asnumpy(self._n_pcls_left))

    @property
    def left_msd(self):
        return float(utils.asnumpy(self._left_msd))

    @property
    def right_msd(self):
        return float(utils.asnumpy(self._right_msd))


def cdf_termcost(policy, x, x_sp):
    env = policy.env
    step_func = policy.step_func
    state_func = policy.state_func
    quad_pts = policy.quad_pts
    quad_ig = policy.quad_ig
    p = cs.vertcat(*[x_sp[2], x_sp[3]])
    s = cs.vertcat(*[x_sp[0], x_sp[1]])
    step_vals = cs.vertcat(*[step_func(pt, p, s) for pt in quad_pts])
    state_vals = cs.vertcat(*[state_func(pt, x[:env.n_y])
                              - state_func(env.bounds[0], x[:env.n_y])
                              for pt in quad_pts])
    squared_diff = cs.sqrt((step_vals - state_vals)**2+1e-4)
    return quad_ig.T @ squared_diff


def cdf_stagecost(policy: MPCControlPolicy, x, u, x_sp, u_sp, Deltau):
    return cdf_termcost(policy, x, x_sp)


def msd_termcost(policy: MPCControlPolicy, x, x_sp):
    env = policy.env
    state_func = policy.state_func
    half_pts, half_ig = policy.half_pts, policy.half_ig

    p = cs.vertcat(*[x_sp[2], x_sp[3]])
    s = cs.vertcat(*[x_sp[0], x_sp[1]])

    left_vals = cs.vertcat(
        *[(pt-p[0])**2 * state_func(pt, x[:env.n_y]) for pt in -half_pts])
    right_vals = cs.vertcat(
        *[(pt-p[1])**2 * state_func(pt, x[:env.n_y]) for pt in half_pts])
    left_penalty = half_ig.T @ left_vals
    right_penalty = half_ig.T @ right_vals

    left_integrand = cs.vertcat(
        *[state_func(pt, x[:env.n_y]) for pt in -half_pts])
    right_integrand = cs.vertcat(
        *[state_func(pt, x[:env.n_y]) for pt in half_pts])
    left_integral = half_ig.T @ left_integrand
    right_integral = half_ig.T @ right_integrand
    ratio_penalty = ((left_integral - s[0])**2
                     + (right_integral - s[1])**2)
    total_penalty = left_penalty + right_penalty + 100 * ratio_penalty
    return total_penalty


def msd_stagecost(policy: MPCControlPolicy, x, u, x_sp, u_sp, Deltau):
    return msd_termcost(policy, x, x_sp)


def msd_termcost3(policy: MPCControlPolicy, x, x_sp, ratio_weight=100):
    env = policy.env
    half_pts, half_ig = policy.half_pts, policy.half_ig
    state_func = policy.state_func

    p = cs.vertcat(*[x_sp[2], x_sp[3]])
    s = cs.vertcat(*[x_sp[0], x_sp[1]])

    left_integrand = cs.vertcat(
        *[state_func(pt, x[:env.n_y]) for pt in -half_pts])
    right_integrand = cs.vertcat(
        *[state_func(pt, x[:env.n_y]) for pt in half_pts])
    left_integral = half_ig.T @ left_integrand
    right_integral = half_ig.T @ right_integrand
    ratio_penalty = ((left_integral - s[0])**2
                     + (right_integral - s[1])**2)
    left_vals = cs.vertcat(
        *[(pt-p[0])**2 * state_func(pt, x[:env.n_y]) for pt in -half_pts])
    right_vals = cs.vertcat(
        *[(pt-p[1])**2 * state_func(pt, x[:env.n_y]) for pt in half_pts])
    left_penalty = (half_ig.T @ left_vals)/left_integral
    right_penalty = (half_ig.T @ right_vals)/right_integral
    total_penalty = left_penalty + right_penalty + ratio_weight * ratio_penalty
    return total_penalty


def msd_stagecost3(policy: MPCControlPolicy, x, u, x_sp, u_sp, Deltau, ratio_weight=100):
    return msd_termcost3(policy, x, x_sp, ratio_weight)


def msd_termcost4(policy: MPCControlPolicy, x, x_sp, ratio_weight=100):
    env = policy.env
    half_pts, half_ig = policy.half_pts, policy.half_ig
    state_func = policy.state_func
    state_penalty_func = policy.state_penalty_func

    p = cs.vertcat(*[x_sp[2], x_sp[3]])
    s = cs.vertcat(*[x_sp[0], x_sp[1]])

    left_integrand = cs.vertcat(
        *[state_func(pt, x[:env.n_y]) for pt in -half_pts])
    right_integrand = cs.vertcat(
        *[state_func(pt, x[:env.n_y]) for pt in half_pts])
    left_integral = half_ig.T @ left_integrand
    right_integral = half_ig.T @ right_integrand
    ratio_penalty = ((left_integral - s[0])**2
                     + (right_integral - s[1])**2)
    left_vals = cs.vertcat(
        *[(pt-p[0])**2 * state_func(pt, x[:env.n_y]) for pt in -half_pts])
    right_vals = cs.vertcat(
        *[(pt-p[1])**2 * state_func(pt, x[:env.n_y]) for pt in half_pts])
    left_penalty = (half_ig.T @ left_vals)/left_integral
    right_penalty = (half_ig.T @ right_vals)/right_integral
    state_penalty = state_penalty_func(x[:env.n_y])

    total_penalty = left_penalty + right_penalty + \
        ratio_weight * ratio_penalty + 1e5*state_penalty
    return total_penalty


def msd_stagecost4(policy: MPCControlPolicy, x, u, x_sp, u_sp, Deltau, ratio_weight=100):
    return msd_termcost4(policy, x, x_sp, ratio_weight)


def cdf_termcost_force(policy: MPCControlPolicy, x, x_sp):
    env = policy.env
    step_func = policy.step_func
    state_func = policy.state_func
    quad_pts = policy.quad_pts
    quad_ig = policy.quad_ig
    p = cs.vertcat(*[x_sp[2], x_sp[3]])
    s = cs.vertcat(*[x_sp[0], x_sp[1]])
    step_vals = cs.vertcat(*[step_func(pt, p, s) for pt in quad_pts])
    state_vals = cs.vertcat(*[state_func(pt, x[:env.n_y])
                              - state_func(env.bounds[0], x[:env.n_y])
                              for pt in quad_pts])
    squared_diff = cs.sqrt((step_vals - state_vals)**2+1e-4)

    return quad_ig.T @ squared_diff * 1e-1


def cdf_stagecost_force(policy: MPCControlPolicy, x, u, x_sp, u_sp, Deltau):
    force_weight = x_sp[4]
    force_sp = x_sp[5]
    force = policy.force_model(x, u)
    force_penalty = (force - force_sp)**2

    u_penalty = u.T @ u

    return cdf_termcost_force(policy, x, x_sp) + force_weight * force_penalty + 1e-5 * u_penalty


def msd_stagecost_force(policy: MPCControlPolicy, x, u, x_sp, u_sp, Deltau):
    force_weight = x_sp[4]
    force_sp = x_sp[5]
    force = policy.force_model(x, u)
    force_penalty = (force - force_sp)**2

    u_penalty = u.T @ u

    return 1e-1 * msd_termcost(policy, x, x_sp) + force_weight * force_penalty + 1e-5 * u_penalty

def msd_stagecost_flux(policy: MPCControlPolicy, x, u, x_sp, u_sp, Deltau):
    flux_weight = x_sp[4]
    flux_sp = x_sp[5]
    ratio_weight = x_sp[7]
    flux = policy.flux_model(x, u)
    integrate_flux = cs.sum1(flux) * policy.env.plate_gap/flux.shape[0]
    flux_penalty = (integrate_flux - flux_sp)**2
    u_penalty = u.T @ u
    cost = 1e-1 * msd_termcost3(policy, x, x_sp, ratio_weight=ratio_weight) + flux_weight * flux_penalty + 1e-5 * u_penalty
    return cost

def msd_stagecost_flux2(policy: MPCControlPolicy, x, u, x_sp, u_sp, Deltau):
    flux_weight = x_sp[4]
    flux_sp = 0.1
    flux = policy.flux_model(x, u)
    integrate_flux = cs.sum1(flux) * policy.env.plate_gap/flux.shape[0]
    flux_penalty = (integrate_flux - flux_sp)**2
    u_penalty = u.T @ u
    cost = flux_weight * flux_penalty + 1e-5 * u_penalty
    return cost

def force_stagecost(policy: MPCControlPolicy, x, u, x_sp, u_sp, Deltau):
    force_weight = x_sp[4]
    force_sp = x_sp[5]
    force = policy.force_model(x, u)
    force_penalty = force_weight * (force - force_sp)**2

    u_penalty = u.T @ u
    return force_penalty + 1e-5 * u_penalty


def dummy_termcost(policy: MPCControlPolicy, x, x_sp):
    return [cs.DM(0)]
