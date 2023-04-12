from stoched import model
from utilities import param_utils, data_utils
import numpy as np
import copy
from scipy.signal import square

from stoched.utils import mpc_utils_det, mpc_utils_lna, mpc_utils_fsp
from tqdm.notebook import tqdm
import warnings

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['axes.titlesize'] = 11
matplotlib.rcParams['axes.labelsize'] = 11
matplotlib.rcParams['xtick.labelsize'] = 11
matplotlib.rcParams['ytick.labelsize'] = 11
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

def run_lna_sim(params, full_target, n_traj):
    #dt = 6
    dt = params.dt
    mu_rfp = 50
    sigma_rfp = 50
    n_rna = 40
    n_protein = 60

    # get the LNA object
    lna = model.Model( params )
    lna.params.x0 = np.zeros(12)
    C = np.zeros((1,3))
    C[0,2] = mu_rfp
    R = np.array([[sigma_rfp]])
    lna.params.delay = 0
    # full_target = np.concatenate((20*np.ones(80),20*np.ones(80)))
    mpc = mpc_utils_lna.MPC(lna, dt, C, R, horizon=4, full_target=full_target)


    # get the ssa object
    params.tvec = np.array([0,dt])
    ssa = model.Model( params )

    all_states = []
    all_cell_fluors = []
    all_fluors = [0]
    all_predicted_distributions = []
    all_estimated_distributions = []

    mpc.control_delay = 0 # number of dt's before the controller acts on the system.
    # M = np.random.rand(3,3)
    # Z = M.dot(M.T)
    # mpc.P_est = Z
    data_now_fluor = 1
    mpc.ncells = n_traj
    data_now = np.zeros((3,1,n_traj))
    #for i in tqdm(range(0,len(full_target)-horizon-mpc.control_delay)):
    mpc.open_loop = False
    for i in tqdm(range(0,150)):
        ssa.params.u = lambda t: mpc.full_control_vector[i-mpc.control_delay]
        ssa.params.tvec = np.linspace(0,dt,10)
        # data_now = ssa.solve_ssa_from_state(data_now[:,np.newaxis,np.newaxis])
        data_now = ssa.solve_ssa_from_state(data_now)
        data_now_fluors = []
        data_now_fluors = [np.abs(data_now[-1,0,j]*mu_rfp 
                                  + data_now[-1,0,j]*np.random.randn()*np.sqrt(sigma_rfp)) for j in range(n_traj)]
        all_states.append(data_now)
        all_cell_fluors.append(data_now_fluors)
        all_fluors.append(np.mean(data_now_fluors))

        # 
        _ = mpc.update_control_vector(np.mean(data_now_fluors))

    # ignore the first data point of the fluorescence. 
    # all_states = all_states[1:]
    all_fluors = all_fluors[1:]
    return all_fluors, mpc, all_states, all_cell_fluors

def run_fsp_sim(params, full_target, n_traj):
    all_states_all = []
    all_mpcs = []
    all_fluors_all = []
    for k in range(n_traj):
        # dt=6
        dt = params.dt
        mu_rfp = 50
        sigma_rfp = 50

        horizon=4
        full_control_vector = np.zeros(len(full_target))
        full_control_vector[0] = 1
        full_control_vector[1] = 1

        n_rna = 40
        n_protein = 60

        pk = np.zeros(n_protein*n_rna)
        pk[0] = 1

        fsp = model.Model( params )
        A_on = fsp.get_A_v3(n_rna, n_protein, u=1)
        A_off = fsp.get_A_v3(n_rna, n_protein, u=0)
        fsp.As = [A_off, A_on]
        fsp.n_rna = n_rna
        fsp.n_protein = n_protein

        # get the ssa object
        params = param_utils.Parameters()
        alpha1 = .25
        alpha2 = .05

        params.kr = .2*alpha1
        params.gx = .018*alpha1

        params.kt = .25*alpha2
        params.gy = .08*alpha2
        
        params.gmx = 1
        params.kpx = 2

        params.b = params.gmx/params.kpx
            
        params.pi = np.zeros(n_protein*n_rna)
        params.pi[0] = 1
        params.delay = 0
        params.tvec = np.array([0,dt])
        ssa = model.Model( params )

        all_states = [[0,0,0]]
        all_fluors = [0]
        all_predicted_distributions = []
        all_estimated_distributions = []

        mpc = mpc_utils_fsp.MPC(fsp, dt, mu_rfp, sigma_rfp, pk, horizon, full_target)
        mpc.control_delay = 4 # number of dt's before the controller acts on the system. 
        data_now_fluor = 1
        data_now = np.array([0,0,0])
        #for i in tqdm(range(0,len(full_target)-horizon-mpc.control_delay)):
        for i in tqdm(range(0,150)):
            ssa.params.u = lambda t: mpc.full_control_vector[i-mpc.control_delay]
            ssa.params.tvec = np.linspace(0,dt,10)
            data_now = ssa.solve_ssa_from_state(data_now[:,np.newaxis,np.newaxis])[:,-1,0] 
            data_now_fluor = np.abs(data_now[-1]*mu_rfp + data_now[-1]*np.random.randn()*np.sqrt(sigma_rfp))
            all_states.append(data_now)
            all_fluors.append(data_now_fluor)
            _ = mpc.update_control_vector( data_now_fluor )

        # ignore the first data point of the fluorescence. 
        all_states = all_states[1:]
        all_fluors = all_fluors[1:]
        all_mpcs.append(mpc)
        all_states_all.append(all_states)
        all_fluors_all.append(all_fluors)
    return all_fluors_all, all_mpcs, all_states_all

def run_bang_bang_sim_pop(params, full_target, n_traj):
    dt = 6
    mu_rfp = 50
    sigma_rfp = 50
    n_rna = 40
    n_protein = 60

    # get the ssa object
    params.tvec = np.array([0,dt])
    ssa = model.Model( params )

    all_states = []
    all_cell_fluors = []
    all_fluors = [0]
    
    data_now_fluor = 1
    data_now = np.zeros((3,1,n_traj))
    #for i in tqdm(range(0,len(full_target)-horizon-mpc.control_delay)):
    full_control_vector = np.zeros(len(full_target))
    for i in tqdm(range(0,150)):
        ssa.params.u = lambda t: full_control_vector[i]
        ssa.params.tvec = np.linspace(0,dt,10)
        # data_now = ssa.solve_ssa_from_state(data_now[:,np.newaxis,np.newaxis])
        data_now = ssa.solve_ssa_from_state(data_now)
        data_now_fluors = []
        data_now_fluors = [np.abs(data_now[-1,0,j]*mu_rfp 
                                  + data_now[-1,0,j]*np.random.randn()*np.sqrt(sigma_rfp)) for j in range(n_traj)]
        all_states.append(data_now)
        all_cell_fluors.append(data_now_fluors)
        all_fluors.append(np.mean(data_now_fluors))

        # do bang bang
        if np.mean(data_now_fluors) >= full_target[i]*mu_rfp:
            full_control_vector[i+1] = 0
        else: 
            full_control_vector[i+1] = 1

    # ignore the first data point of the fluorescence. 
    # all_states = all_states[1:]
    all_fluors = all_fluors[1:]
    return all_fluors, full_control_vector, all_states,all_cell_fluors

def run_bang_bang_sim_sc(params, full_target, n_traj):
    all_states_all = []
    all_controls = []
    all_fluors_all = []
    for k in tqdm(range(n_traj)):
        dt=6

        mu_rfp = 50
        sigma_rfp = 50

        horizon=4
        full_control_vector = np.zeros(len(full_target))
        full_control_vector[0] = 1
        full_control_vector[1] = 1

        n_rna = 40
        n_protein = 60

        # get the ssa object
        params = param_utils.Parameters()
        alpha1 = .25
        alpha2 = .05

        params.kr = .2*alpha1
        params.gx = .018*alpha1

        params.kt = .25*alpha2
        params.gy = .08*alpha2
        
        params.gmx = 1
        params.kpx = 2

        params.b = params.gmx/params.kpx
            
        params.pi = np.zeros(n_protein*n_rna)
        params.pi[0] = 1
        params.delay = 0
        params.tvec = np.array([0,dt])
        ssa = model.Model( params )

        all_states = [[0,0,0]]
        all_fluors = [0]
        control_delay = 4 # number of dt's before the controller acts on the system. 
        data_now_fluor = 1
        data_now = np.array([0,0,0])
        full_control_vector = np.zeros(len(full_target)+control_delay)
        #for i in tqdm(range(0,len(full_target)-horizon-mpc.control_delay)):
        for i in range(0,150):
            ssa.params.u = lambda t: full_control_vector[i-control_delay]
            ssa.params.tvec = np.linspace(0,dt,10)
            data_now = ssa.solve_ssa_from_state(data_now[:,np.newaxis,np.newaxis])[:,-1,0] 
            data_now_fluor = np.abs(data_now[-1]*mu_rfp + data_now[-1]*np.random.randn()*np.sqrt(sigma_rfp))
            all_states.append(data_now)
            all_fluors.append(data_now_fluor)
            
            # do bang bang
            if data_now_fluor >= full_target[i]*mu_rfp:
                full_control_vector[i+1] = 0
            else: 
                full_control_vector[i+1] = 1

        # ignore the first data point of the fluorescence. 
        all_states = all_states[1:]
        all_fluors = all_fluors[1:]
        full_control_vector = full_control_vector[1:]
        all_controls.append(full_control_vector)
        all_states_all.append(all_states)
        all_fluors_all.append(all_fluors)
    return all_fluors_all, all_controls, all_states_all

def plot_lna_simulation(all_fluors,mpc,all_cell_fluors,full_target,params,n_traj,nplot=10):
    tvec = mpc.period*np.arange(len(all_fluors))
    tvec_targ = np.arange(len(full_target))*mpc.period
    full_target_fluor = full_target*params.mu_rfp
    # all_states = np.array(all_states)
    all_cell_fluors = np.array(all_cell_fluors)

    f = plt.figure(figsize=(3,3))
    ax = f.add_axes([0.2,0.3,.75,.45])
    ax.plot(tvec_targ,full_target_fluor,'k--')
    #ax.plot(tvec,all_states[:,-1,:,:].reshape(150,n_traj)*params.mu_rfp, linewidth=0.5, color='gray')
    ax.plot(tvec,all_cell_fluors[:,:nplot], linewidth=0.5, color='gray')
    ax.plot(tvec,np.array(all_fluors),'maroon')
    ax.set_xlim([tvec[0],tvec[-1]])

    # print('single cell error: {0}'.format(.1 * np.sum( (np.atleast_2d(full_target_fluor[:150]).T-all_states[:,-1,:,:].reshape(150,n_traj)*params.mu_rfp)**2 )))
    # print('Mean error: {0}'.format(np.sum( (full_target_fluor[:150]-np.array(all_fluors))**2 )))


    # ax.plot(tvec_targ,full_target,'k--')
    # ax.plot(tvec,all_states[:,1],'indianred')
    ax.set_ylim([0,2000])
    ax.set_xlabel('time [min]')
    ax.set_ylabel('a.f.u.')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax2 = f.add_axes([.2, .75, .75, .07 ])
    ax2.imshow(np.atleast_2d(mpc.full_control_vector)[:len(tvec)], cmap='Blues', aspect='auto', vmin=0, vmax=3)
    ax2.set_yticks([])
    ax2.set_yticklabels([])
    ax2.set_xticks([0,len(tvec)])
    _ = ax2.set_xticklabels([])
    return f,ax 

def plot_fsp_simulation(all_fluors,all_mpcs, all_states, full_target, params, n_traj):
    tvec = all_mpcs[0].period*np.arange(len(all_fluors))
    tvec_targ = np.arange(len(full_target))*all_mpcs[0].period
    full_target_fluor = full_target*params.mu_rfp
    all_states = np.array(all_states)
    f = plt.figure(figsize=(4,4))
    ax = f.add_axes([0.2,0.3,.75,.45])
    ax.plot(tvec_targ,full_target_fluor,'k--')
    all_measurements = []
    all_controls = []
    for i,mpc in enumerate(all_mpcs):
        if i<10:
            ax.plot(tvec, mpc.all_measurements, linewidth=0.5, color='gray')
        all_measurements.append(mpc.all_measurements)
        all_controls.append(mpc.full_control_vector)

    ax.plot(tvec,np.mean(np.array(all_measurements),axis=0),'maroon')

    print('single cell error: {0}'.format(.1 * np.sum( (np.atleast_2d(full_target_fluor[:150]).T-np.array(all_measurements).T)**2 )))
    print('Mean error: {0}'.format(np.sum( (full_target_fluor[:150]-np.mean(np.array(all_measurements),axis=0))**2 )))

    # ax.plot(tvec_targ,full_target,'k--')
    # ax.plot(tvec,all_states[:,1],'indianred')
    ax.set_ylim([0,2000])
    ax.set_xlim([0,1000])
    ax.set_xlabel('time [min]')
    ax.set_ylabel('a.f.u.')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax2 = f.add_axes([.2, .75, .75, .07 ])
    ax2.imshow(np.atleast_2d(np.array(all_controls)), cmap='Blues', aspect='auto', vmin=0, vmax=3)
    ax2.set_yticks([])
    ax2.set_yticklabels([])
    ax2.set_xticks([0,len(tvec)])
    _ = ax2.set_xticklabels([])
    return f,ax

def error_decomposition(trajectories, target, dt):
    value = .95*target[0]
    ntraj = trajectories.shape[0]
    hit_times = []
    steady_state_errors = []
    for i in range(ntraj):
        try:
            hit_times.append(np.argwhere(trajectories[i,:]>value)[0][0])
            steady_state_errors.append(np.sum((target[hit_times[-1]:] 
                                        - trajectories[i,hit_times[-1]:])**2))
        except:
            hit_times.append([np.nan])
            steady_state_errors.append([np.nan])
    return np.array(hit_times), steady_state_errors

def plot_fsp_simulation_2(all_out, full_target, params, n_traj):
    tvec = params.dt*np.arange(len(all_out[0][0][0]))
    tvec_targ = np.arange(len(full_target))*params.dt
    full_target_fluor = full_target*params.mu_rfp
    # all_states = np.array(all_states)
    f = plt.figure(figsize=(3,3))
    ax = f.add_axes([0.2,0.3,.75,.45])
    ax.plot(tvec_targ,full_target_fluor,'k--')
    all_measurements = []
    all_controls = []
    for i,out in enumerate(all_out):
        if i<10:
            ax.plot(tvec, out[-2][0], linewidth=0.5, color='gray')
        all_measurements.append(out[-2][0])
        all_controls.append(out[-1][0])

    ax.plot(tvec,np.mean(np.array(all_measurements),axis=0),'maroon')
    ax.set_xlim([tvec[0],tvec[-1]])

    # print('single cell error: {0}'.format(.1 * np.sum( (np.atleast_2d(full_target_fluor[:150]).T-np.array(all_measurements).T)**2 )))
    # print('Mean error: {0}'.format(np.sum( (full_target_fluor[:150]-np.mean(np.array(all_measurements),axis=0))**2 )))



    # ax.plot(tvec_targ,full_target,'k--')
    # ax.plot(tvec,all_states[:,1],'indianred')
    ax.set_ylim([0,2000])
    # ax.set_xlim([0,1000])
    ax.set_xlabel('time [min]')
    ax.set_ylabel('a.f.u.')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax2 = f.add_axes([.2, .75, .75, .07 ])
    ax2.imshow(np.atleast_2d(np.array(all_controls))[:,:len(tvec)], cmap='Blues', aspect='auto', vmin=0, vmax=3)
    ax2.set_yticks([])
    ax2.set_yticklabels([])
    ax2.set_xticks([0,len(tvec)])
    _ = ax2.set_xticklabels([])
    return f,ax

def plot_bang_bang_simulation(all_fluors, control, all_states, full_target,params,n_traj):
    tvec = params.dt*np.arange(len(all_fluors))
    tvec_targ = np.arange(len(full_target))*params.dt
    full_target_fluor = full_target*params.mu_rfp
    all_states = np.array(all_states)

    f = plt.figure(figsize=(4,4))
    ax = f.add_axes([0.2,0.3,.75,.45])
    ax.plot(tvec_targ,full_target_fluor,'k--')
    ax.plot(tvec,all_states[:,-1,:,:].reshape(150,n_traj)*params.mu_rfp, linewidth=0.5, color='gray')
    ax.plot(tvec,np.array(all_fluors),'maroon')

    print('single cell error: {0}'.format(.1 * np.sum( (np.atleast_2d(full_target_fluor[:150]).T-all_states[:,-1,:,:].reshape(150,n_traj)*params.mu_rfp)**2 )))
    print('Mean error: {0}'.format(np.sum( (full_target_fluor[:150]-np.array(all_fluors))**2 )))


    # ax.plot(tvec_targ,full_target,'k--')
    # ax.plot(tvec,all_states[:,1],'indianred')
    ax.set_ylim([0,2000])
    ax.set_xlabel('time [min]')
    ax.set_ylabel('a.f.u.')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax2 = f.add_axes([.2, .75, .75, .07 ])
    ax2.imshow(np.atleast_2d(control), cmap='Blues', aspect='auto', vmin=0, vmax=3)
    ax2.set_yticks([])
    ax2.set_yticklabels([])
    ax2.set_xticks([0,len(tvec)])
    _ = ax2.set_xticklabels([])
    return f,ax 

def plot_bang_bang_simulation_sc(all_fluors, control, all_states, full_target,params,n_traj):
    tvec = params.dt*np.arange(all_states.shape[1])
    tvec_targ = np.arange(len(full_target))*params.dt
    full_target_fluor = full_target*params.mu_rfp
    all_states = np.array(all_states)

    f = plt.figure(figsize=(4,4))
    ax = f.add_axes([0.2,0.3,.75,.45])
    ax.plot(tvec_targ,full_target_fluor,'k--')
    #ax.plot(tvec,all_states[:,:,-1].reshape(150,n_traj)*params.mu_rfp, linewidth=0.5, color='gray')
    ax.plot(tvec, np.array(all_fluors).T, linewidth=0.5, color='gray')
    ax.plot(tvec,np.array(all_fluors).mean(axis=0),'maroon')

    # print('single cell error: {0}'.format(.1 * np.sum( (np.atleast_2d(full_target_fluor[:150]).T-all_states[:,-1,:,:].reshape(150,n_traj)*params.mu_rfp)**2 )))
    # print('Mean error: {0}'.format(np.sum( (full_target_fluor[:150]-np.array(all_fluors))**2 )))


    # ax.plot(tvec_targ,full_target,'k--')
    # ax.plot(tvec,all_states[:,1],'indianred')
    ax.set_ylim([0,2000])
    ax.set_xlabel('time [min]')
    ax.set_ylabel('a.f.u.')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax2 = f.add_axes([.2, .75, .75, .07 ])
    ax2.imshow(np.atleast_2d(control), cmap='Blues', aspect='auto', vmin=0, vmax=3)
    ax2.set_yticks([])
    ax2.set_yticklabels([])
    ax2.set_xticks([0,len(tvec)])
    _ = ax2.set_xticklabels([])
    return f,ax 

def plot_fsp_simulation_v3(all_fluors,all_control, all_states, full_target, params, n_traj):
    tvec = params.dt*np.arange(all_fluors.shape[1])
    tvec_targ = np.arange(len(full_target))*params.dt
    full_target_fluor = full_target*params.mu_rfp
    all_states = np.array(all_states)
    f = plt.figure(figsize=(4,4))
    ax = f.add_axes([0.2,0.3,.75,.45])
    ax.plot(tvec_targ,full_target_fluor,'k--')
    for i,control in enumerate(all_control):
        if i<10:
            ax.plot(tvec, all_fluors[i,:], linewidth=0.5, color='gray')
    all_measurements = all_fluors
    all_controls = all_control
    
    ax.plot(tvec,np.mean(np.array(all_measurements),axis=0),'maroon')

    print('single cell error: {0}'.format(.1 * np.sum( (np.atleast_2d(full_target_fluor[:150]).T-np.array(all_measurements).T)**2 )))
    print('Mean error: {0}'.format(np.sum( (full_target_fluor[:150]-np.mean(np.array(all_measurements),axis=0))**2 )))

    # ax.plot(tvec_targ,full_target,'k--')
    # ax.plot(tvec,all_states[:,1],'indianred')
    ax.set_ylim([0,2000])
    # ax.set_xlim([0,1000])
    ax.set_xlabel('time [min]')
    ax.set_ylabel('a.f.u.')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax2 = f.add_axes([.2, .75, .75, .07 ])
    ax2.imshow(np.atleast_2d(np.array(all_controls)), cmap='Blues', aspect='auto', vmin=0, vmax=3)
    ax2.set_yticks([])
    ax2.set_yticklabels([])
    ax2.set_xticks([0,len(tvec)])
    _ = ax2.set_xticklabels([])
    return f,ax
