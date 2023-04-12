import os
import dill
import numpy as np
import pandas as pd
import pathos.multiprocessing as mp
import stoched.model as model
from stoched.utils import mpc_utils_lna, mpc_utils_fsp
from stoched.get_mpc_objects import get_mpc_object_fsp, get_mpc_object_det
from stoched.utils.utilities import param_utils, data_utils
from MicroMator.FileManagement import logger, makedirs_

MODULE = 'model module'

def save_pickle_object(path, list):
    with open(path, 'wb') as file:
        dill.dump(list, file)

def load_pickle_object(path):
    with open(path, 'rb') as file:
        list_ = dill.load(file)
    return list_

def modify_list(mpc, data_now):
    mpc.update_control_vector(data_now)
    return mpc

def run_mpc_single_cell(turn_on_cells, all_particles, all_mpcs, GLOBALVARS, path_to_data, n_frames_between_update, measurement_period, position, readout_key, number_positions, get_mpc_object, stopframe, offset=120):
    '''
    Run this function in some kind of "master loop".
    all_particles: list of all the particles so far
    all_mpcs: list of MPC objects, one for each particle
    GLOBALVARS: you know what this is
    path_to_data: path to the track_csv for this data
    n_frames_between_update: number of frames between MPC updates
    measurement_period: amound of time between different measurements in minutes.
    readout_key: which attribute of the pandas dataframe we want to control for each cell
    '''
    frame = GLOBALVARS['CURRENT_FRAME'].value
    # check for new frames
    mpcs_now = []
    measurements_now = []
    turn_on_cells = []
    # load the CSV
    data = pd.read_csv(path_to_data)
    logger('offset value is {}, okay Zach. Are you happy now????!'.format(offset), print_=False, module=MODULE, frame=frame, pos=position)
    logger('read analysis data', module=MODULE, frame=frame, pos=position)
    if stopframe < frame and stopframe:
        old_particles = data.loc[data.frame < frame-n_frames_between_update].particle
        current_data = data.loc[(data.frame == frame) & (data.particle.isin(old_particles))]
    else:
        current_data = data[data.frame == frame]
    for particle in current_data.particle:
        if stopframe >= frame or not stopframe:
            if particle not in all_particles:
                #all_particles.append(particle)
                all_mpcs[particle] = get_mpc_object(dt=n_frames_between_update*measurement_period)
                all_mpcs[particle].particle_id = particle
                all_mpcs[particle].iteration = int(frame/n_frames_between_update)
                all_mpcs[particle].initial_iteration = int(frame/n_frames_between_update)
            measurements_now.append(float(current_data.loc[current_data.particle == particle][readout_key])-offset)
            mpcs_now.append(all_mpcs[particle])
        # update the control vector for each cell
        else:
            if particle in all_particles:
                measurements_now.append(float(current_data.loc[current_data.particle == particle][readout_key])-offset)
                mpcs_now.append(all_mpcs[particle])
    func = lambda x: modify_list(*x)
    with mp.Pool(20//number_positions) as pool:
        mpcs_now = pool.map(func, list(zip(mpcs_now, measurements_now)))
    for _, mpc in enumerate(mpcs_now):
        if mpc.control_vector_0:
            turn_on_cells.append(mpc.particle_id)
        all_mpcs[mpc.particle_id] = mpc
    logger('list of turned on cells: {}'.format(turn_on_cells), print_=False, module=MODULE, frame=frame, pos=position)
    logger('finished MPC prediction', module=MODULE, frame=frame, pos=position)
    return turn_on_cells, all_mpcs, list(all_mpcs.keys())

def run_mpc_population(turn_on_cells, all_particles, mpc, GLOBALVARS, path_to_data, n_frames_between_update, measurement_period, position, readout_key, number_positions, get_mpc_object, stopframe):
    '''
    Run this function in some kind of "master loop".
    all_particles: list of all the particles so far
    all_mpcs: list of MPC objects, one for each particle
    GLOBALVARS: you know what this is
    path_to_data: path to the track_csv for this data
    n_frames_between_update: number of frames between MPC updates
    measurement_period: amound of time between different measurements in minutes.
    readout_key: which attribute of the pandas dataframe we want to control for each cell
    '''
    frame = GLOBALVARS['CURRENT_FRAME'].value
    # check for new frames
    measurements_now = []
    # load the CSV
    data = pd.read_csv(path_to_data)
    logger('read analysis data', module=MODULE, frame=frame, pos=position)
    current_data = data[data.frame == frame]
    mpc.ncells = len(current_data)
    # update the control vector for each cell
    measurements_now = float(current_data[readout_key].mean())
    try:
        turn_on_cells = mpc.update_control_vector(measurements_now)
    except ValueError as err:
        logger(err, module=MODULE, frame=frame, pos=position)
        turn_on_cells = []
    except Exception as err:
        logger(err, module=MODULE, frame=frame, pos=position)
        turn_on_cells = []
    if turn_on_cells:
        turn_on_cells = list(current_data.particle)
    else:
        turn_on_cells = []
    logger('list of turned on cells: {}'.format(turn_on_cells), print_=False, module=MODULE, frame=frame, pos=position)
    logger('finished MPC prediction', module=MODULE, frame=frame, pos=position)
    return turn_on_cells, mpc, all_particles

def looper(globaldict, protocol, folder_manager_obj, position, n_frames_between_update, measurement_period, readout_key, stopframe, number_positions, pop=False, open_loop=False):
    frame = 0
    all_particles = []
    all_mpcs = {}
    if pop:
        all_mpcs = get_mpc_object_det(dt=n_frames_between_update*measurement_period, open_loop=open_loop)
    turn_on_cells = []
    data_path = os.path.join(folder_manager_obj.analysis_path, 'SegMator', 'pos{}'.format(position), 'track.csv')
    model_path = os.path.join(folder_manager_obj.math_path, 'pos{}_all_mpcs.pickle'.format(position))
    all_particles_path = os.path.join(folder_manager_obj.math_path, 'pos{}_all_particles.pickle'.format(position))
    turn_on_cells_path = os.path.join(folder_manager_obj.analysis_path, 'Signal_logs', 'pos{}'.format(position))
    makedirs_(turn_on_cells_path)
    turn_on_cells_path = os.path.join(turn_on_cells_path, 'cell_ids_{}.npy')
    save_pickle_object(all_particles_path, all_particles)
    save_pickle_object(model_path, all_mpcs)
    np.save(turn_on_cells_path.format(frame), turn_on_cells, allow_pickle=True)
    while frame <= protocol.acqNumframes-1 and not globaldict['END_ACQ'].value:
        while not globaldict['CURRENT_FRAME'].value == frame and not globaldict['END_ACQ'].value: #wait loop
            if globaldict['END_ACQ'].value:
                return
        if globaldict['ANALYSIS_PER_POS'][position].value and (not frame%n_frames_between_update):
            globaldict['ANALYSIS_PER_POS'][position].value = False
            all_particles = load_pickle_object(all_particles_path)
            all_mpcs = load_pickle_object(model_path)
            turn_on_cells = np.load(turn_on_cells_path.format(frame))
            if not pop:
                turn_on_cells, all_mpcs, all_particles = run_mpc_single_cell(list(turn_on_cells),
                                                                             all_particles,
                                                                             all_mpcs,
                                                                             globaldict,
                                                                             data_path,
                                                                             n_frames_between_update,
                                                                             measurement_period,
                                                                             position,
                                                                             readout_key=readout_key,
                                                                             number_positions=number_positions,
                                                                             get_mpc_object=get_mpc_object_fsp,
                                                                             stopframe=stopframe)
            else:
                turn_on_cells, all_mpcs, all_particles = run_mpc_population(list(turn_on_cells),
                                                                            all_particles,
                                                                            all_mpcs,
                                                                            globaldict,
                                                                            data_path,
                                                                            n_frames_between_update,
                                                                            measurement_period,
                                                                            position,
                                                                            readout_key=readout_key,
                                                                            number_positions=number_positions,
                                                                            get_mpc_object=get_mpc_object_det,
                                                                            stopframe=stopframe)
            frame += 1
            save_pickle_object(all_particles_path, all_particles)
            save_pickle_object(model_path, all_mpcs)
            np.save(turn_on_cells_path.format(frame), turn_on_cells)
        elif globaldict['ANALYSIS_PER_POS'][position].value and frame%n_frames_between_update:
            globaldict['ANALYSIS_PER_POS'][position].value = False
            frame += 1
            np.save(turn_on_cells_path.format(frame), turn_on_cells)