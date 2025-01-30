# [depends] train.pth
import os
from lib import utils
from lib.hd_controller import ControlSimulator, msd_stagecost, msd_termcost
import gsd.hoomd
import numpy as np
xp = utils.array_module('cupy')



def main():
    device = 'cpu'
    info = utils.lazy_info_import(__file__, use_backups=True)
    dirname, basename = utils.get_dir_base(__file__)
    destname = os.path.join(dirname,
                            f'backups/{basename}.gsd')
    init_path1 = os.path.join(dirname,
                              'backups/ctrl_init_gsd1.gsd')
    init_path2 = os.path.join(dirname,
                              'backups/ctrl_init_gsd2.gsd')
    pars_dict = info[0]['env_input_dict']
    pth_info = info[0]
    pars_dict['md_samp_dt'] = 1e-1
    pars_dict['xfilter_max'] = 20
    pars_dict['xfilter_min'] = -20
    pars_dict['n_meas_modes'] = 40
    env = ControlSimulator(pars_dict)

    def f(x, scale):
        result = -3*np.tanh(2*np.sin(2*np.pi*x/scale))
        return result
    bounds = (-5, 5)
    scale = bounds[1] - bounds[0]
    N = 6
    x = np.linspace(bounds[0], bounds[1], 2*N)[:-1]
    y = f(x,scale)
    env.create_init_gsd(init_path1,
                        init_path2,
                        env.ctrl_samp_dt,
                        xp.array(y.reshape(-1,1)))

    init_frames2 = gsd.hoomd.open(init_path2, 'rb')
    env.create_ctrl_gsd(init_path1,
                        init_frames2,
                        pth_info,
                        destname,
                        stagecost=msd_stagecost,
                        termcost=msd_termcost)


if __name__ == "__main__":
    main()

