"""
BD simulator that saves relaxing gsd file.
"""

import copy
import numpy as np
import gsd.hoomd

try:
    import hoomd
except ImportError as e:
    hoomd = None
from lib import utils
from lib import hoomd_utils
from lib import hd_utils
from lib.hd_utils import BaseController, keys_to_array
from lib.utils import asnumpy, array_module, ascupy
from scipy.interpolate import make_lsq_spline, interp1d

xp = array_module("cupy")


def random_sum_to(min_int, max_int, goal_sum):
    total_sum = 0
    samples = []

    while total_sum <= (goal_sum - max_int):
        sample = xp.random.randint(min_int, max_int + 1)
        total_sum += sample
        samples.append(sample)
    samples.append(goal_sum - total_sum)
    return xp.array(samples)


def create_u_seq(md_dt, nnam_dt, min_time, max_time, sim_time, n_u, ulb, uub, seed=0):
    xp.random.seed(seed)
    n_steps = int(sim_time / nnam_dt)
    min_u_steps = int(min_time // nnam_dt)
    max_u_steps = int(max_time // nnam_dt)
    du_steps = random_sum_to(min_u_steps, max_u_steps, n_steps)
    # Add dummy end to have correct length.
    # du_steps = xp.hstack((du_steps, xp.array([100])))
    u_seq = xp.random.uniform(ulb, uub, (n_u, len(du_steps)))
    u_seq = xp.hstack((u_seq, u_seq[:, -1:]))
    step_positions = xp.cumsum(du_steps)
    step_positions = xp.hstack((xp.zeros(1), step_positions))
    step_positions = (step_positions * nnam_dt / md_dt).astype("int64")
    return u_seq, step_positions


class Simulator:
    def __init__(
        self,
        ld2=None,
        w001_max=None,
        plate_gap=None,
        epsilon=None,
        ctrl_info_dict=None,
        n_pcls=None,
        interact=None,
        vol_frac=None,
        md_dt=None,
        md_samp_dt=None,
        md_ctrl_dt=None,
        md_seed=None,
        meas_order=None,
        n_meas_modes=None,
        xfilter_min=None,
        xfilter_max=None,
        track_force=None,
        use_spline=None,
        width_scale=None,
        init_gsd_time=None,
        n_state_pts=None,
        n_force_pts=None,
        nnam_dt=None,
        n_time_pts=None,
        n_p=None,
        nn_hidden_layer_dims=None,
        model=None,
        min_du_time=None,
        max_du_time=None,
        train_batch_size=None,
        val_batch_size=None,
        test_batch_size=None,
        train_sim_time=None,
        test_sim_time=None,
        n_train_batches=None,
        n_val_batches=None,
        n_test_batches=None,
        end_time_pad=None,
        n_horizon=None,
        state_cost=None,
        diff_cost=None,
        terminal_penalty=None,
        dz2_ctrl_cost=None,
        dt_ctrl_cost=None,
        n_constrained_pts=None,
        n_quad_pts=None,
        n_half_pts=None,
        meas_time_pad=None,
        ctrl_samp_dt=None,
        center=None,
        center1=None,
        center2=None,
        trigger_times=None,
        ctrl_sim_time=None,
        sp_step_sizes=None,
        amp=None,
        trapped_frac=None,
        trap_width=None,
        mf_ctrl_time=None,
        nx_quad_pts=None,
        nt_quad_pts=None,
        mf_discount_factor=None,
        mf_max_vel=None,
        mf_pi_k=None,
        mf_pi_i=None,
        force_control=None,
        fc_start=None,
        fc_weight=None,
        fc_amp=None,
        fc_period=None,
        fc_tol=None,
    ):
        self.input_dict = locals()
        self.input_dict.pop("self")
        self.default_dict = dict(
            ld2=2,
            w001_max=3,
            plate_gap=10,
            epsilon=5,
            ctrl_info_dict={"wr001": {"n_pts": 11}},
            n_pcls=int(1e4),
            interact=True,
            vol_frac=0.2,
            md_dt=5e-4,
            md_samp_dt=0.01,
            md_ctrl_dt=1e-2,
            md_seed=10,
            meas_order=2,
            n_meas_modes=128,
            xfilter_min="none",
            xfilter_max="none",
            track_force=False,
            use_spline=True,
            width_scale=3.0,
            init_gsd_time=100,
            n_state_pts=21,
            n_force_pts=21,
            nnam_dt=0.5,
            n_time_pts=2,
            n_p=2,
            nn_hidden_layer_dims=[64],
            model="irk_v1",
            min_du_time=5,
            max_du_time=20,
            train_batch_size=100,
            val_batch_size=100,
            test_batch_size=100,
            train_sim_time=10,
            test_sim_time=100,
            n_train_batches=10,
            n_val_batches=1,
            n_test_batches=1,
            end_time_pad=5,
            n_horizon=14,
            state_cost=6400,
            diff_cost=64000,
            terminal_penalty=6400,
            dz2_ctrl_cost=5e-5,
            dt_ctrl_cost=0.001,
            n_constrained_pts=80,
            n_quad_pts=101,
            n_half_pts=30,
            meas_time_pad=2,
            ctrl_samp_dt=1e-3,
            center=0,
            center1=-2.5,
            center2=2.5,
            trigger_times=[0, 30, 60, 90, 120],
            ctrl_sim_time=120,
            amp=3,
            sp_step_sizes=np.array([[0.5, 0.5], [0.3, 0.7], [0.7, 0.3], [0.5, 0.5]]),
            trapped_frac=0.8,
            trap_width=6,
            mf_ctrl_time=100,
            nx_quad_pts=21,
            nt_quad_pts=5,
            mf_discount_factor=1.0,
            mf_max_vel=2,
            mf_pi_k=-1,
            mf_pi_i=-1,
            force_control=False,
            fc_start=0,
            fc_weight=1e4,
            fc_amp=0.1,
            fc_period=10,
            fc_tol=False,
        )
        self.ld2 = None
        self.w001_max = None
        self.plate_gap = None
        self.epsilon = None
        self.ctrl_info_dict = None
        self.n_pcls = None
        self.interact = None
        self.vol_frac = None
        self.md_dt = None
        self.md_samp_dt = None
        self.md_ctrl_dt = None
        self.md_seed = None
        self.meas_order = None
        self.n_meas_modes = None
        self.xfilter_min = None
        self.xfilter_max = None
        self.track_force = None
        self.use_spline = None

        self.width_scale = None
        self.init_gsd_time = None

        self.n_state_pts = None
        self.n_force_pts = None
        self.nnam_dt = None
        self.n_time_pts = None
        self.n_p = None
        self.nn_hidden_layer_dims = None
        self.model = None

        self.min_du_time = None
        self.max_du_time = None
        self.train_batch_size = None
        self.val_batch_size = None
        self.test_batch_size = None
        self.train_sim_time = None
        self.test_sim_time = None
        self.n_train_batches = None
        self.n_val_batches = None
        self.n_test_batches = None
        self.end_time_pad = None

        self.n_horizon = None
        self.state_cost = None
        self.diff_cost = None
        self.terminal_penalty = None
        self.dz2_ctrl_cost = None
        self.dt_ctrl_cost = None

        self.n_constrained_pts = None
        self.n_quad_pts = None
        self.n_half_pts = None
        self.meas_time_pad = None
        self.ctrl_samp_dt = None

        self.center = None
        self.center1 = None
        self.center2 = None
        self.trigger_times = None
        self.ctrl_sim_time = None
        self.sp_step_sizes = None
        self.amp = None

        self.trapped_frac = None
        self.trap_width = None
        self.mf_ctrl_time = None
        self.nx_quad_pts = None
        self.nt_quad_pts = None
        self.mf_discount_factor = None
        self.mf_max_vel = None

        self.mf_pi_k = None
        self.mf_pi_i = None

        self.force_control = None
        self.fc_start = None
        self.fc_weight = None
        self.fc_amp = None
        self.fc_period = None
        self.fc_tol = None

        # If parameter is specified, use chosen parameter
        # Else use the value from the default_dict
        for key, val in self.input_dict.items():
            class_vars = vars(self)
            val = (
                class_vars[key]
                if class_vars[key] is not None
                else val if val is not None else self.default_dict[key]
            )
            self.input_dict[key] = val
            setattr(self, key, val)
        dx = self.plate_gap / self.n_state_pts  # Grid spacing
        self.grad_mat = np.diag(
            2 * np.pi * np.fft.fftshift(np.fft.fftfreq(self.n_state_pts, dx) * 1j)
        )
        self.n_state_modes = int((self.n_state_pts - 1) / 2)
        self.n_force_modes = int((self.n_force_pts - 1) / 2)
        # BD parameters
        ell = 1  # ell: run length,
        D_r = 1  # tau^{-1}: tau_r
        delta = ell / xp.sqrt(self.ld2)  # delta: microscopic length
        self.pcl_rad = xp.sqrt(3 / 4) * delta
        self.zeta_t = 1  # mu•tau^{-1}: mu = mass of particle
        # mu•tau^{-1}•sigma^{-1}
        eta = self.zeta_t / (6 * xp.pi * self.pcl_rad)
        self.zeta_r = 8 * xp.pi * eta * self.pcl_rad**3
        self.kbt = D_r * self.zeta_r
        D_t = 4 / 3 * self.pcl_rad**2 * D_r  # sigma^{2}•tau^{-1}
        tau = 1 / D_r  # tau
        u0 = xp.sqrt(self.ld2 * D_t / tau)  # sigma•tau^{-1}
        self.fp = self.zeta_t * u0
        self.bounds = (-self.plate_gap / 2, self.plate_gap / 2)
        self.plate_width = (
            self.n_pcls * xp.pi * self.pcl_rad**2 / (self.plate_gap * self.vol_frac)
        )

        self.full_state_keys = []
        for i in range(self.meas_order):
            if i == 0:
                self.full_state_keys.append(f"pr{i:0=3d}r")
                self.full_state_keys.append(f"pr{i:0=3d}i")
            else:
                self.full_state_keys.append(f"pr{i:0=3d}r")
                self.full_state_keys.append(f"pr{i:0=3d}i")
                self.full_state_keys.append(f"pi{i:0=3d}r")
                self.full_state_keys.append(f"pi{i:0=3d}i")
        if self.track_force:
            self.full_state_keys.append("forcer")
            self.full_state_keys.append("forcei")

        # NNAM parameters
        self.n_u = int(
            xp.sum(
                xp.asarray(
                    [
                        self.ctrl_info_dict[key]["n_pts"]
                        for key in list(self.ctrl_info_dict.keys())
                    ]
                )
            )
        )
        self.n_ctrl_modes = int((self.n_u + 1) / 2)
        self.n_y = self.n_state_pts
        self.n_g = self.n_y * self.n_time_pts
        self.n_z = self.n_p * (self.n_y + self.n_u)

        (self.time_pts, self.time_dz, self.time_dz2, self.time_iz) = (
            xp.asarray(e)
            for e in utils.colloc_v2(
                self.n_time_pts + 1, self.nnam_dt, method="radau_w_left"
            )
        )

        self.g1 = self.time_dz[1:, 0].reshape(-1, 1)
        self.g2 = self.time_dz[1:, 1:]
        self.ulb = xp.ones((self.n_u, 1)) * -self.w001_max
        self.uub = xp.ones((self.n_u, 1)) * self.w001_max
        input_dim = self.n_z + self.n_y + self.n_u
        self.nn_dims = [input_dim]
        self.nn_dims.extend(self.nn_hidden_layer_dims)
        self.nn_dims.append(self.n_force_pts)

        # Training parameters
        self.n_train_traj = self.train_batch_size * self.n_train_batches
        self.n_val_traj = self.val_batch_size * self.n_val_batches
        self.n_test_traj = self.test_batch_size * self.n_test_batches

        self.sim_time = (self.train_sim_time + self.n_p * self.nnam_dt) * (
            self.n_train_traj + self.n_val_traj
        ) + (self.test_sim_time + self.n_p * self.nnam_dt) * self.n_test_traj

    def create_init_gsd(
        self, gsd_file1, gsd_file2, samp_dt=None, u=None, step_positions=None
    ):
        if samp_dt is None:
            samp_dt = self.md_samp_dt

        xp.random.seed(self.md_seed)
        positions = utils.get_fcc_pts(
            self.width_scale * self.plate_width, self.plate_gap, 2 * self.pcl_rad, xp
        )
        positions = positions[
            xp.random.choice(positions.shape[0], self.n_pcls, replace=False), :
        ]
        thetas = xp.random.uniform(0, 2 * xp.pi, self.n_pcls).reshape(-1, 1)
        position = utils.asnumpy(xp.hstack((positions, xp.zeros((self.n_pcls, 1)))))
        position = [tuple(e) for e in position]
        snapshot = gsd.hoomd.Snapshot()
        snapshot.particles.N = self.n_pcls
        snapshot.configuration.box = [
            self.width_scale * utils.asnumpy(self.plate_width),
            self.plate_gap,
            0,
            0,
            0,
            0,
        ]
        snapshot.particles.typeid = [0] * self.n_pcls
        snapshot.particles.position = list(position[0 : self.n_pcls])
        snapshot.particles.types = ["A"]
        snapshot.particles.moment_inertia = utils.asnumpy(
            xp.ones((self.n_pcls, 3)) * 2 / 5
        )  # note we are picking rho so m=1 mu
        snapshot.particles.diameter = utils.asnumpy(
            xp.ones(self.n_pcls) * 2 * self.pcl_rad
        )
        # theta = xp.zeros(n_particles).reshape(-1, 1)

        info = hoomd_utils.get_quaternion_from_euler(asnumpy(thetas), 0, 0)
        qx, qy, qz, qw = (xp.asarray(e) for e in info)
        snapshot.particles.orientation = utils.asnumpy(xp.hstack((qx, qy, qz, qw)))
        if hoomd.version.gpu_enabled:
            device = hoomd.device.GPU()
        else:
            device = hoomd.device.CPU()
        sim = hoomd.Simulation(device=device, seed=self.md_seed)
        with gsd.hoomd.open(name=gsd_file1, mode="wb+") as f:
            f.append(snapshot)
        sim.create_state_from_snapshot(snapshot)
        integrator = hoomd.md.Integrator(dt=self.md_dt, integrate_rotational_dof=True)
        brownian = hoomd.md.methods.Brownian(kT=self.kbt, filter=hoomd.filter.All())
        brownian.gamma.default = self.zeta_t
        brownian.gamma_r.default = [self.zeta_r, self.zeta_r, self.zeta_r]
        integrator.methods.append(brownian)
        sim.operations.integrator = integrator
        active1 = hoomd.md.force.Active(filter=hoomd.filter.All())
        active1.active_force["A"] = (self.fp, 0, 0)
        integrator.forces.append(active1)
        if self.interact:
            cell = hoomd.md.nlist.Cell(buffer=0.4)
            lj_pcl = hoomd.md.pair.LJ(nlist=cell)
            lj_pcl.params[("A", "A")] = dict(
                epsilon=self.epsilon, sigma=(2 * self.pcl_rad / (2 ** (1 / 6)))
            )
            lj_pcl.r_cut[("A", "A")] = 2 * self.pcl_rad
            integrator.forces.append(lj_pcl)

        snapshot = sim.state.get_snapshot()
        box_height = snapshot.configuration.box[1]
        pcl_pos = xp.clip(
            snapshot.particles.position[:, 1], -self.plate_gap / 2, self.plate_gap / 2
        )
        pcl_quarts = xp.asarray(snapshot.particles.orientation)
        pcl_thetas = hd_utils.gpu_quaternion_to_euler_angle_vectorized3(
            pcl_quarts[:, 0], pcl_quarts[:, 3]
        )

        if self.track_force:
            forces = xp.zeros(snapshot.particles.N)
        else:
            forces = None

        full_state0 = hd_utils.samp_to_fourier(
            xp.asarray(pcl_pos),
            pcl_thetas,
            self.n_state_modes + 1,
            self.meas_order,
            self.bounds,
            forces=forces,
        )
        if u is None:
            u_seq, step_positions = (xp.zeros((self.n_u, 1)), [0])

            hoomd_controller = DataAcquistionController(
                self,
                full_state0,
                u_seq,
                step_positions,
                ctrl=False,
                samp_dt=samp_dt,
                track_force=self.track_force,
            )
        else:
            u_seq = u
            if step_positions is None:
                step_positions = xp.zeros(1)
            hoomd_controller = DataAcquistionController(
                self,
                full_state0,
                u_seq,
                step_positions,
                ctrl=True,
                samp_dt=samp_dt,
                track_force=self.track_force,
            )
        integrator.forces.append(hoomd_controller)

        ramp_time = self.init_gsd_time / 5
        t_ramp = int(ramp_time * (int(xp.round(1 / self.md_dt))))
        ramp = hoomd.variant.Ramp(A=0, B=1, t_start=0, t_ramp=t_ramp)
        initial_box = sim.state.box
        final_box = hoomd.Box.from_box(initial_box)
        final_box.Lx = self.plate_width
        box_resize_trigger = hoomd.trigger.Periodic(10)
        box_resize = hoomd.update.BoxResize(
            box1=initial_box, box2=final_box, variant=ramp, trigger=box_resize_trigger
        )
        sim.operations.updaters.append(box_resize)
        logger = hoomd.logging.Logger()
        logger["wr001r"] = (hoomd_controller, "wr001r", "sequence")
        logger["wr001i"] = (hoomd_controller, "wr001i", "sequence")
        for key in self.full_state_keys:
            logger[key] = (hoomd_controller, key, "sequence")
        n_steps = self.init_gsd_time * (int(xp.round(1 / self.md_dt))) + 1

        gsd_writer1 = hoomd.write.GSD(
            filename=gsd_file1,
            trigger=hoomd.trigger.On(n_steps - 1),
            mode="wb",
            truncate=True,
        )
        gsd_writer1.log = logger
        sim.operations.writers.append(gsd_writer1)
        if (
            "none" in str(self.xfilter_max).lower()
            or "none" in str(self.xfilter_min).lower()
        ):
            gsd_writer2 = hoomd.write.GSD(
                filename=gsd_file2,
                trigger=hoomd.trigger.Periodic(
                    int(xp.round(1 / (self.md_dt / samp_dt)))
                ),
                dynamic=["property"],
                mode="wb",
                filter=hoomd.filter.Null(),
            )
        elif (
            "all" in str(self.xfilter_max).lower()
            or "all" in str(self.xfilter_min).lower()
        ):
            gsd_writer2 = hoomd.write.GSD(
                filename=gsd_file2,
                trigger=hoomd.trigger.Periodic(
                    int(xp.round(1 / (self.md_dt / samp_dt)))
                ),
                dynamic=["property"],
                mode="wb",
            )
        else:
            filter_ = hd_utils.XFilter(self.xfilter_min, self.xfilter_max, xp=xp)
            filter_updater = hoomd.update.FilterUpdater(
                hoomd.trigger.Periodic(int(xp.round(1 / (self.md_dt / samp_dt)))),
                [filter_],
            )
            sim.operations.updaters.append(filter_updater)
            gsd_writer2 = hoomd.write.GSD(
                filename=gsd_file2,
                trigger=hoomd.trigger.Periodic(
                    int(xp.round(1 / (self.md_dt / samp_dt)))
                ),
                dynamic=["property"],
                mode="wb",
                filter=filter_,
            )

        gsd_writer2.log = logger
        sim.operations.writers.append(gsd_writer2)

        sim.run(n_steps)
        return None

    @staticmethod
    def get_xuyscales(simData):
        """Quick function to get the scaling factors
        for measurements, states, and control inputs."""

        # First get the input and output scaling. Then, specify the scaling
        # for the state variable x.
        xuyscales = utils.get_uyscaling(simData=simData)
        ymean, ystd = xuyscales["yscale"]
        xmean = xp.zeros(simData.x.shape[1:])
        xstd = xp.ones(simData.x.shape[1:])
        # xmean[:self.n_g] = ymean  # (Update Cc_mean)
        # xstd[:self.n_g] = ystd  # (Update Cc_std)
        xuyscales["xscale"] = (xmean, xstd)
        # Return.
        return xuyscales

    def create_train_gsd(self, init_gsd, dest_gsd):
        # Number of time steps in the entire training and validation data.
        if hoomd.version.gpu_enabled:
            device = hoomd.device.GPU()
        else:
            device = hoomd.device.CPU()
        sim = hoomd.Simulation(device=device, seed=self.md_seed)
        sim.timestep = 0
        sim.create_state_from_gsd(filename=init_gsd)
        integrator = hoomd.md.Integrator(dt=self.md_dt, integrate_rotational_dof=True)
        brownian = hoomd.md.methods.Brownian(kT=self.kbt, filter=hoomd.filter.All())
        brownian.gamma.default = self.zeta_t
        brownian.gamma_r.default = [self.zeta_r, self.zeta_r, self.zeta_r]
        integrator.methods.append(brownian)
        sim.operations.integrator = integrator
        active1 = hoomd.md.force.Active(filter=hoomd.filter.All())
        active1.active_force["A"] = (self.fp, 0, 0)
        integrator.forces.append(active1)
        if self.interact:
            cell = hoomd.md.nlist.Cell(buffer=0.4)
            lj_pcl = hoomd.md.pair.LJ(nlist=cell)
            lj_pcl.params[("A", "A")] = dict(
                epsilon=self.epsilon, sigma=(2 * self.pcl_rad / (2 ** (1 / 6)))
            )
            lj_pcl.r_cut[("A", "A")] = 2 * self.pcl_rad
            integrator.forces.append(lj_pcl)

        snapshot = sim.state.get_snapshot()
        box_height = snapshot.configuration.box[1]
        pcl_pos = xp.clip(
            snapshot.particles.position[:, 1], -self.plate_gap / 2, self.plate_gap / 2
        )
        pcl_quarts = xp.asarray(snapshot.particles.orientation)
        pcl_thetas = hd_utils.gpu_quaternion_to_euler_angle_vectorized3(
            pcl_quarts[:, 0], pcl_quarts[:, 3]
        )
        if self.track_force:
            forces = xp.zeros(snapshot.particles.N)
        else:
            forces = None
        full_state0 = hd_utils.samp_to_fourier(
            xp.asarray(pcl_pos),
            pcl_thetas,
            self.n_force_modes,
            self.meas_order,
            self.bounds,
            forces=forces,
        )

        u_seq, step_positions = create_u_seq(
            self.md_dt,
            self.nnam_dt,
            self.min_du_time,
            self.max_du_time,
            self.sim_time,
            self.n_u,
            self.ulb,
            self.uub,
            self.md_seed,
        )

        hoomd_controller = DataAcquistionController(
            self,
            full_state0,
            u_seq,
            step_positions,
            ctrl=True,
            track_force=self.track_force,
        )
        integrator.forces.append(hoomd_controller)
        logger = hoomd.logging.Logger()
        logger["wr001r"] = (hoomd_controller, "wr001r", "sequence")
        logger["wr001i"] = (hoomd_controller, "wr001i", "sequence")
        for key in self.full_state_keys:
            logger[key] = (hoomd_controller, key, "sequence")

        if (
            "none" in str(self.xfilter_max).lower()
            or "none" in str(self.xfilter_min).lower()
        ):
            gsd_writer1 = hoomd.write.GSD(
                filename=dest_gsd,
                trigger=hoomd.trigger.Periodic(
                    int(xp.round(1 / (self.md_dt / self.md_samp_dt)))
                ),
                dynamic=["property"],
                mode="wb",
                filter=hoomd.filter.Null(),
            )
        elif (
            "all" in str(self.xfilter_max).lower()
            or "all" in str(self.xfilter_min).lower()
        ):
            gsd_writer1 = hoomd.write.GSD(
                filename=dest_gsd,
                trigger=hoomd.trigger.Periodic(
                    int(xp.round(1 / (self.md_dt / self.md_samp_dt)))
                ),
                dynamic=["property"],
                mode="wb",
            )
        else:
            filter_ = hd_utils.XFilter(self.xfilter_min, self.xfilter_max, xp=xp)
            filter_updater = hoomd.update.FilterUpdater(
                hoomd.trigger.Periodic(
                    int(xp.round(1 / (self.md_dt / self.md_samp_dt)))
                ),
                [filter_],
            )
            sim.operations.updaters.append(filter_updater)
            gsd_writer1 = hoomd.write.GSD(
                filename=dest_gsd,
                trigger=hoomd.trigger.Periodic(
                    int(xp.round(1 / (self.md_dt / self.md_samp_dt)))
                ),
                mode="wb",
                dynamic=["property"],
                filter=filter_,
            )

        gsd_writer1.log = logger
        sim.operations.writers.append(gsd_writer1)

        sim.run((self.sim_time + self.end_time_pad) * (int(xp.round(1 / self.md_dt))))
        return None

    def create_test_gsd(
        self,
        init_file: str,
        traj_file: str,
        u_seq,
        step_positions,
        sim_time: float,
        last_frame_file: str = None,
        samp_dt: float = None,
        convert_u_to_coeffs: bool = True,
    ):
        if samp_dt is None:
            samp_dt = self.md_samp_dt
        # Number of time steps in the entire training and validation data.
        if hoomd.version.gpu_enabled:
            device = hoomd.device.GPU()
        else:
            device = hoomd.device.CPU()
        sim = hoomd.Simulation(device=device, seed=self.md_seed)
        sim.timestep = 0
        sim.create_state_from_gsd(filename=init_file)
        integrator = hoomd.md.Integrator(dt=self.md_dt, integrate_rotational_dof=True)
        brownian = hoomd.md.methods.Brownian(kT=self.kbt, filter=hoomd.filter.All())
        brownian.gamma.default = self.zeta_t
        brownian.gamma_r.default = [self.zeta_r, self.zeta_r, self.zeta_r]
        integrator.methods.append(brownian)
        sim.operations.integrator = integrator
        active1 = hoomd.md.force.Active(filter=hoomd.filter.All())
        active1.active_force["A"] = (self.fp, 0, 0)
        integrator.forces.append(active1)
        if self.interact:
            cell = hoomd.md.nlist.Cell(buffer=0.4)
            lj_pcl = hoomd.md.pair.LJ(nlist=cell)
            lj_pcl.params[("A", "A")] = dict(
                epsilon=self.epsilon, sigma=(2 * self.pcl_rad / (2 ** (1 / 6)))
            )
            lj_pcl.r_cut[("A", "A")] = 2 * self.pcl_rad
            integrator.forces.append(lj_pcl)

        snapshot = sim.state.get_snapshot()
        box_height = snapshot.configuration.box[1]
        pcl_pos = xp.clip(
            snapshot.particles.position[:, 1], -self.plate_gap / 2, self.plate_gap / 2
        )
        pcl_quarts = xp.asarray(snapshot.particles.orientation)
        pcl_thetas = hd_utils.gpu_quaternion_to_euler_angle_vectorized3(
            pcl_quarts[:, 0], pcl_quarts[:, 3]
        )
        if self.track_force:
            forces = xp.zeros(snapshot.particles.N)
        else:
            forces = None
        full_state0 = hd_utils.samp_to_fourier(
            xp.asarray(pcl_pos),
            pcl_thetas,
            self.n_force_modes,
            self.meas_order,
            self.bounds,
            forces=forces,
        )

        hoomd_controller = DataAcquistionController(
            self,
            full_state0,
            u_seq,
            step_positions,
            ctrl=True,
            track_force=self.track_force,
            convert_u_to_coeffs=convert_u_to_coeffs,
        )
        integrator.forces.append(hoomd_controller)
        logger = hoomd.logging.Logger()
        logger["wr001r"] = (hoomd_controller, "wr001r", "sequence")
        logger["wr001i"] = (hoomd_controller, "wr001i", "sequence")
        for key in self.full_state_keys:
            logger[key] = (hoomd_controller, key, "sequence")
        if (
            "none" in str(self.xfilter_max).lower()
            or "none" in str(self.xfilter_min).lower()
        ):
            traj_writer = hoomd.write.GSD(
                filename=traj_file,
                trigger=hoomd.trigger.Periodic(
                    int(xp.round(1 / (self.md_dt / samp_dt)))
                ),
                dynamic=["property"],
                mode="wb",
                filter=hoomd.filter.Null(),
            )
        elif (
            "all" in str(self.xfilter_max).lower()
            or "all" in str(self.xfilter_min).lower()
        ):
            traj_writer = hoomd.write.GSD(
                filename=traj_file,
                trigger=hoomd.trigger.Periodic(
                    int(xp.round(1 / (self.md_dt / samp_dt)))
                ),
                dynamic=["property"],
                mode="wb",
            )
        else:
            filter_ = hd_utils.XFilter(self.xfilter_min, self.xfilter_max, xp=xp)
            filter_updater = hoomd.update.FilterUpdater(
                hoomd.trigger.Periodic(int(xp.round(1 / (self.md_dt / samp_dt)))),
                [filter_],
            )
            sim.operations.updaters.append(filter_updater)
            traj_writer = hoomd.write.GSD(
                filename=traj_file,
                trigger=hoomd.trigger.Periodic(
                    int(xp.round(1 / (self.md_dt / samp_dt)))
                ),
                dynamic=["property"],
                mode="wb",
                filter=filter_,
            )
        n_steps = sim_time * (int(xp.round(1 / self.md_dt)))
        traj_writer.log = logger
        sim.operations.writers.append(traj_writer)
        if last_frame_file is not None:
            last_frame_writer = hoomd.write.GSD(
                filename=last_frame_file,
                trigger=hoomd.trigger.On(n_steps - 1),
                mode="wb",
                truncate=True,
            )
            last_frame_writer.log = logger
            sim.operations.writers.append(last_frame_writer)
        sim.run(n_steps)
        return None

    def gsd_to_arr(self, init_frames, data_frames, start_t):
        n_state_modes = self.n_state_modes

        ts = []
        xs = []
        us = []

        frame_start = max([int(start_t / self.md_samp_dt - 1), 0])
        step0 = init_frames[frame_start].configuration.step
        precision = int(-np.floor(np.log10(self.md_dt)))

        state_keys = ["pr000"]
        for i in range(1, self.meas_order):
            state_keys.append(f"pr{i:0=3d}")
            state_keys.append(f"pi{i:0=3d}")
        ctrl_keys = list(self.ctrl_info_dict.keys())

        for frame in init_frames[frame_start:]:
            step = frame.configuration.step - step0
            ts.append(np.around(step * self.md_dt, precision))
            xs.append(
                keys_to_array(state_keys, frame.log, cutoff=n_state_modes + 1, xp=np)
            )
            us.append(keys_to_array(ctrl_keys, frame.log, xp=np))

        print("starting")
        frame_start = 0
        start_t = ts[-1]
        us.pop(-1)

        us.append(keys_to_array(ctrl_keys, frame.log, xp=np))

        for i, frame in enumerate(data_frames[frame_start:]):
            if i % int(len(data_frames[frame_start:]) / 100) == 0:
                frac = int(i / len(data_frames[frame_start:]) * 100)
                print(f"\r{frac:3d}% done", end="")
            step = frame.configuration.step
            ts.append(np.around(step * self.md_dt + start_t, precision))
            xs.append(
                keys_to_array(state_keys, frame.log, cutoff=n_state_modes + 1, xp=np)
            )
            us.append(keys_to_array(ctrl_keys, frame.log, xp=np))
        ts = np.array(ts)
        xs = np.array(xs)
        us = np.array(us).squeeze()

        return dict(ts=ts, xs=xs, us=us)

    def arr_to_sim_data(self, data, spline_order, start_time_pad):
        ts, xs, us = data.ts, data.xs, data.us

        dt = self.nnam_dt
        time_pts = utils.asnumpy(self.time_pts)
        nnam_tf = ts[-1] - start_time_pad - self.end_time_pad
        nnam_ts = np.arange(0, nnam_tf, dt)
        nnam_aug_ts = (nnam_ts[:, np.newaxis] + time_pts[:0:-1]) - dt
        knots = np.arange(ts[0], ts[-1] + dt, dt)
        k = spline_order
        m = 1
        ext_knots = utils.augknt(knots, k, m)
        step_times = ts[np.where((us[1:] != us[:-1]).any(axis=1))[0] + 1]
        ext_knots = np.sort(np.hstack((ext_knots, step_times.repeat(k - 1))))
        if self.use_spline:
            spline_ts = ts
            spline_xs = xs
        else:
            sample_ts = (
                (np.arange(0, ts[-1] + dt, dt)[:, None] + time_pts[:-1])
            ).flatten()
            t_mask = utils.find_close_indices_vectorized(ts, sample_ts)
            spline_ts = ts[t_mask]
            spline_xs = xs[t_mask]
        spline = make_lsq_spline(spline_ts, spline_xs, ext_knots, k=k)
        fit_curves_matrices = spline(nnam_aug_ts + start_time_pad)
        nnam_xs = fit_curves_matrices
        nnam_ys = (nnam_xs[:, :, 0, :]).reshape(nnam_xs.shape[0], -1)
        nnam_u_coeffs = us[np.where((np.isin(ts, nnam_ts + start_time_pad)))]
        nnam_us = nnam_u_coeffs
        sim_data = utils.SimData(t=nnam_ts, x=nnam_xs, u=nnam_us, y=nnam_ys)
        return sim_data

    # @profile
    def get_training_data(self, data, start_time_pad):
        training_sim_data = self.arr_to_sim_data(data, self.n_time_pts, start_time_pad)

        train_sim_time = self.train_sim_time
        test_sim_time = self.test_sim_time
        dt = self.nnam_dt
        n_p = self.n_p

        n_train_traj = self.n_train_traj
        n_val_traj = self.n_val_traj
        n_test_traj = self.n_test_traj

        n_train_steps_per_traj = int(train_sim_time // dt)
        n_val_steps_per_traj = n_train_steps_per_traj
        n_test_steps_per_traj = int(test_sim_time // dt)

        training_data = utils.split_data_into_train_val(
            simData=training_sim_data,
            n_p=n_p,
            n_train_traj=n_train_traj,
            n_val_traj=n_val_traj,
            n_test_traj=n_test_traj,
            n_train_steps_per_traj=n_train_steps_per_traj,
            n_val_steps_per_traj=n_val_steps_per_traj,
            n_test_steps_per_traj=n_test_steps_per_traj,
        )
        # Get the scaling.
        xuyscales = self.get_xuyscales(training_sim_data)

        # Save the number of training and validation trajectories.
        num_train_traj = [n_train_traj, n_val_traj, n_test_traj]

        out = dict(
            training_sim_data=training_sim_data,
            training_data=training_data,
            xuyscales=xuyscales,
            num_train_traj=num_train_traj,
        )

        # Return.
        return out

    def get_state_space(self):
        raise NotImplementedError


class DataAcquistionController(BaseController):
    def __init__(
        self,
        env: Simulator,
        full_state0,
        u_seq,
        step_positions,
        ctrl,
        samp_dt=None,
        track_force=False,
        convert_u_to_coeffs=True,
    ):
        super().__init__(env, full_state0)
        self.n_pcls = env.n_pcls
        self.step_positions = step_positions
        self.bounds = env.bounds
        self.n_u = env.n_u
        self.wr001_coeffs_dict = {}
        for ind1, row in enumerate(u_seq.T):
            if convert_u_to_coeffs:
                coeffs = hd_utils.fourier_fit(row, self.bounds, int((self.n_u + 1) / 2))
            else:
                coeffs = row
            self.wr001_coeffs_dict[ind1] = coeffs
        self._wr001_coeffs = utils.asnumpy(self.wr001_coeffs_dict[0])
        self.sim_dt = env.md_dt
        if samp_dt is None:
            self.sample_dt = env.md_samp_dt
        else:
            self.sample_dt = samp_dt
        self.md_ctrl_dt = env.md_ctrl_dt
        self.zeta_r = env.zeta_r
        self.ctrl = ctrl
        self.track_force = track_force
        self.y_forces = None
        self.alpha = self.sim_dt / self.sample_dt

    def w_ctrl_fcn(self, z, ctrl_index):
        wr001 = hd_utils.evaluate_fourier_series(
            z, self.wr001_coeffs_dict[ctrl_index], self.bounds
        )
        return wr001

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
                    pcl_rtags
                ]
                if self.y_forces is None:
                    self.y_forces = new_val
                else:
                    self.y_forces = (
                        self.alpha * new_val + (1 - self.alpha) * self.y_forces
                    )

        if self.ctrl:
            if timestep % round(self.md_ctrl_dt / self.sim_dt) == 0:
                if timestep % round(self.sample_dt / self.sim_dt) == 0:
                    # print(timestep * self.sim_dt)
                    pcl_pos, pcl_thetas = self.measure(
                        evalulate_meas_vars=True, forces=self.y_forces
                    )
                else:
                    pcl_pos, pcl_thetas = self.measure(
                        evalulate_meas_vars=False, forces=self.y_forces
                    )
                ctrl_index = int(xp.where(timestep >= self.step_positions)[0][-1])
                self._wr001_coeffs = self.wr001_coeffs_dict[ctrl_index]
                with force_arrays as arrays:
                    arrays.torque[:, -1] = ascupy(
                        self.w_ctrl_fcn(pcl_pos, ctrl_index)
                        * xp.cos(pcl_thetas)
                        * self.zeta_r
                    )
            elif timestep % round(self.sample_dt / self.sim_dt) == 0:
                self.measure(evalulate_meas_vars=True, forces=self.y_forces)
        else:
            # print(timestep)
            if timestep % round(self.sample_dt / self.sim_dt) == 0:
                pcl_pos, pcl_thetas = self.measure(
                    evalulate_meas_vars=True, forces=self.y_forces
                )

    @property
    def wr001r(self):
        return utils.asnumpy(self._wr001_coeffs).real

    @property
    def wr001i(self):
        return utils.asnumpy(self._wr001_coeffs).imag


# class FieldSimulator:
#     def __init__(self,
#                  **kwargs
#                  ):
#         self.input_dict = kwargs
#         self.bd_env = Simulator(**self.input_dict)
#         self.fi_env = BaseFieldSimulator()
#         for key, val in self.input_dict.items():
#             if key in self.bd_env.input_dict.keys():
#                 setattr(self.fi_env, key, val)
#                 self.fi_env.input_dict[key] = val
#         self.fi_env.initialize()
