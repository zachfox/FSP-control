import sys
sys.path.append('/Users/zachfox/projects')
import stoched.model as model
import numpy as np
import sys
from stoched.utils import mpc_utils_fsp, mpc_utils_lna, mpc_utils_det
from stoched.utils.utilities import param_utils, data_utils
import pandas as pd
import scipy
from tqdm import tqdm
import get_mpc_objects as controllers


def run_mpc_single_cell(turn_on_cells, all_particles, all_mpcs, GLOBALVARS, path_to_data, n_frames_between_update, measurement_period, readout_key='RHOD-DIRECT mean'):
    '''
    Run this function in some kind of "master loop".
    all_particles: list of all the particles so far
    all_mpcs: list of MPC objects, one for each particle
    GLOBALVARS: you know what this is
    path_to_data: path to the track_csv for this data
    n_frames_between_update: number of frames between MPC updates
    measurement_period: amound of time between different measurements in minutes.
    readout_key: which attribute of the pandas dataframe we want to control for each cell
    A
    '''
    # load these matrices becvause it takes forever.
    # check for new frames
    if GLOBALVARS['ANALYZED'] and (not GLOBALVARS['CURRENT_FRAME']%n_frames_between_update):
        turn_on_cells = []
        # load the CSV
        data = pd.read_csv(path_to_data)
        current_data = data[data.frame==GLOBALVARS['CURRENT_FRAME']]

        # find new particles that have appeared and add MPCS for them
        for particle in tqdm(current_data.particle):
            if particle not in all_particles:
                all_particles.append(particle)
                #all_mpcs.append(get_mpc_object(dt=n_frames_between_update*measurement_period))
                all_mpcs.append(controllers.get_mpc_object_fsp(dt=n_frames_between_update*measurement_period))
                all_mpcs[-1].particle_id = particle
                all_mpcs[-1].iteration = int(GLOBALVARS['CURRENT_FRAME']/n_frames_between_update)
            # update the control vector for each cell
            measurement = current_data[current_data.particle==particle][readout_key]
            _, on_or_off = all_mpcs[all_particles.index(particle)].update_control_vector(float(measurement))
            # append list of particles which should be turned on at the next frame.
            if on_or_off:
                turn_on_cells.append(particle)
    return turn_on_cells, all_mpcs, all_particles

if __name__ == '__main__':
    # eA_on = scipy.sparse.load_npz('utils/matrices/eA_on.npz')
    # eA_off = scipy.sparse.load_npz('utils/matrices/eA_off.npz')
    n_frames_between_update = 2
    measurement_period = 3 # minutes
    path_to_data = 'sample_data/track_test.csv'
    all_particles = []
    all_mpcs = []
    turn_on_cells = []
    for frame in range(30,42):
        print('FRAME: ', frame)
        gvars = {'ANALYZED':True,'CURRENT_FRAME':frame}
        ### note, if you don't want to keep this stuff alive we could save/load each of these objects,
        # but I'm not sure how well the MPC objects serialize with pickle. I'm guessing they won't.
        turn_on_cells, all_mpcs, all_particles = run_mpc_single_cell(turn_on_cells, all_particles, all_mpcs, gvars, path_to_data, n_frames_between_update, measurement_period, readout_key='RHOD-DIRECT mean')
        print(turn_on_cells)
        np.save('cells_id_pos_0.npy',turn_on_cells)
