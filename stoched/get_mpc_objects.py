import stoched.model as model
import numpy as np
import sys
from stoched.utils import mpc_utils_fsp, mpc_utils_lna, mpc_utils_det
from stoched.utils.utilities import param_utils, data_utils
import pandas as pd
import scipy
from tqdm import tqdm

def get_mpc_object_lna(dt):
    '''Get the MPC object.
    period; time between model updates.
    '''
    mu_rfp = 5
    sigma_rfp = 50

    full_target = np.concatenate((5*np.ones(100),1500*np.ones(98)))

    horizon=3
    full_control_vector = np.zeros(len(full_target))
    full_control_vector[0] = 1
    full_control_vector[1] = 1

    n_rna = 50
    n_protein = 75

    pk = np.zeros(n_protein*n_rna)
    pk[0] = 1

    # get the model for control
    params = param_utils.Parameters()
    params.kr = .5
    params.gx = .023
    params.kt = .035
    params.gy = .023
    params.u = lambda t: 1
    params.x0 = np.zeros(6)
    params.tvec = np.array([0,6])
    mom = model.Model(params)
    mom.n_rna = 50
    mom.n_protein = 75
    # mom.eA_on = eA_on
    # mom.eA_off = eA_off
    # mom.eAs = [mom.eA_off, mom.eA_on]
    A_on = mom.get_A(n_rna, n_protein, u=1)
    A_off = mom.get_A(n_rna, n_protein, u=0)
    mom.As = [A_off, A_on]
    mpc = mpc_utils_fsp.MPC(mom, dt, mu_rfp, sigma_rfp, pk, horizon, full_target)
    return mpc

def get_mpc_object_lna_bis(dt):
    '''Get the MPC object.
    period; time between model updates.
    '''
    mu_f = 1
    sigma2_f = 500

    C = np.array([[0, mu_f]])
    R = np.array([[sigma2_f]])

    full_target = np.concatenate((300*np.ones(100), 1500*np.ones(98)))

    horizon = 6
    full_control_vector = np.zeros(len(full_target))
    full_control_vector[0] = 1
    full_control_vector[1] = 1

    # get the model for control
    params = param_utils.Parameters()
    alpha = 4
    alpha2 = 1.9
    params.gx = 0.08022340588820423*alpha
    params.kr = 0.018047732255497842*alpha
    params.gy = 0.0025224927384929264*alpha2
    params.kt = (32.2091432523833)*alpha2
    params.delay = 30
    params.u = lambda t: 1
    params.x0 = np.zeros(6)
    params.tvec = np.linspace(0, 500, 100)
    mom = model.Model(params)
    mpc = mpc_utils_lna.MPC(mom, dt, C, R, horizon, full_target=full_target)
    return mpc

def get_mpc_object_fsp(dt):
    '''Get the MPC object.
    period; time between model updates.
    '''
    mu_rfp = 50
    sigma_rfp = 50

    full_target = np.concatenate((10*np.ones(46), 20*np.ones(100)))

    horizon=4
    full_control_vector = np.zeros(len(full_target))
    full_control_vector[0] = 1
    full_control_vector[1] = 1

    n_rna = 40
    n_protein = 60

    pk = np.zeros(n_protein*n_rna)
    pk[0] = 1

    # get the model for control
    params = param_utils.Parameters()
    alpha1 = .25
    alpha2 = .05

    params.kr = .2*alpha1
    params.gx = .018*alpha1

    params.kt = .25*alpha2
    params.gy = .08*alpha2
    
    params.gmx = 10
    params.kpx = 20

    params.b = params.gmx/params.kpx

    params.u = lambda t: 1
    params.x0 = np.zeros(6)
    params.pi = np.zeros(n_protein*n_rna)
    params.pi[0] = 1
    params.delay = 25
    params.tvec = np.array([0,dt])

    
    fsp = model.Model( params )
    A_on = fsp.get_A_v3(n_rna, n_protein, u=1)
    A_off = fsp.get_A_v3(n_rna, n_protein, u=0)
    fsp.As = [A_off, A_on]
    fsp.n_rna = n_rna
    fsp.n_protein = n_protein
    mpc = mpc_utils_fsp.MPC(fsp, dt, mu_rfp, sigma_rfp, pk, horizon, full_target)
    mpc.control_delay = 6
    return mpc

def get_mpc_object_det(dt, open_loop=False):

    horizon=4
    full_target = np.concatenate((10*np.ones(46), 20*np.ones(100)))
    dt = 6
    mu_rfp = 50
    sigma_rfp = 50

    C = np.zeros((3,3))
    C[2,2] = mu_rfp
    params = param_utils.Parameters()
    alpha1 = .25
    alpha2 = .05

    params.kr = .2*alpha1
    params.gx = .018*alpha1

    params.kt = .25*alpha2
    params.gy = .08*alpha2

    params.gmx = 10
    params.kpx = 20

    params.b = params.gmx/params.kpx

    params.u = lambda t: 0
    params.x0 = np.zeros(3)
    params.delay = 0
    params.tvec = np.array([0,dt])

    ode = model.Model( params )
    mpc = mpc_utils_det.MPC(ode, dt, C, np.zeros((3,3)), horizon=horizon, full_target=full_target)
    mpc.open_loop = open_loop
    mpc.control_delay = 6
    return mpc
