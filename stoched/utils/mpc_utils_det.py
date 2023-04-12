import numpy as np
import copy
import itertools
import matplotlib.pyplot as plt


class StateEstimation:
    # def __init__(self, C, R):
    #     self.C = C
    #     self.R = R

    def predict_state(self, x, P, model, dt):
        '''Predict the next state based on the
        state x and the covariance estimate P, on the
        interval t, t+dt.
        '''
        model.params.x0 = np.copy(x)
        model.params.tvec = np.linspace(0,dt,10)
        soln = model.solve_mean()
        # P[-1,-1] = soln.y[-1,-1]
        if not self.open_loop:
            for i in range(P.shape[0]):
                P[i,i] = soln.y[i,-1]/self.ncells
        return soln.y[:,-1], P

    def estimate_state(self, x, P, yk):
        '''Estimate the state based on current state
        x and covariance P, using a measurement yk.
        Returns estimated x and p
        '''
        #print(np.shape(x), np.shape(P), np.shape(yk))
        d = x.shape[0]
        K = np.dot( P, self.C.T).dot(np.linalg.pinv(np.dot(self.C, P).dot(self.C.T)+self.R))
        x_est = x + K.dot(yk - np.dot(self.C, x))
        P_est = (np.eye(d) - np.dot(K,self.C)).dot(P)
        return x_est, P_est

class MPC(StateEstimation):
    def __init__(self, model, period, C, R, horizon=None, full_target=None):
        self.C = C
        self.R = R
        self.period = period
        self.full_control_vector = np.zeros(len(full_target))
        self.full_control_vector[0] = 0
        self.full_control_vector[1] = 0
        self.model = model
        self.x_est = np.zeros(3)
        self.P_est = np.zeros((3,3))
        self.iteration = 0
        self.all_state_estimates = []
        self.all_measurements = []
        self.all_pk = []
        self.open_loop = False
        self.control_delay = 0
        self.ncells = 1
        if horizon:
            self.horizon = 4
        else:
            self.horizon = horizon
        if full_target is None:
            self.full_target = np.concatenate((800*np.ones(30),1500*np.ones(50)))
        else:
            self.full_target = full_target

    def controller_func(self, t, control_vec, period ):
        '''
        t: current time
        control_vec: a vector of ones and zeros. len(control_vec)*period is
        the prediction horizon.
        period: time between control events.
        '''
        tvec = np.arange(len(control_vec))*period
        control_ind = np.where(tvec<=t)[0][-1]
        return control_vec[control_ind]

    def optimize_binary_control(self, target, x, P):
        '''optimize binary control vector.
        target_now: target over the control horizon. len(target_now)*period
        is horizon.
        period: duration of constant stimulation
        model: model object
        x: current state
        P: current state variance
        C: observation matrix
        '''
        all_costs = []
        all_control_vectors = [list(i) for i in itertools.product([0, 1], repeat=len(target))]
        self.model.params.x0 = x 
        self.model.params.tvec = np.arange(0, self.period*len(target), self.period)
        for control_vector in all_control_vectors:
            model_now = copy.deepcopy(self.model)
            all_costs.append(self.cost_function(target, control_vector, model_now, self.period))
        best_controller = all_control_vectors[np.argmin(all_costs)]
        return best_controller

    def cost_function(self, target, control_vector, model, period):
        '''compute the errors of the model to the target for a given
        control vector
        '''
        model.params.u = lambda t: self.controller_func(t, control_vector, period)
        sol = model.solve_mean()
        mm = sol.y[:,-1]
        mu = np.dot(self.C, np.atleast_2d(mm).T)
       # sig = np.dot(self.C,sol.covariances)
        return np.sum(((target*self.C[2,2]-mu)**2))

    def update_control_vector(self, yk):
        '''When a new mesurement comes in, use this function to
        update the state estimate, make predictions, etc
        '''
        # predict current state from previous state
        self.model.params.u = lambda t: self.full_control_vector[self.iteration-1-self.control_delay]
        x_pred, P_pred = self.predict_state(self.x_est, self.P_est, self.model, self.period)
        self.all_pk.append(x_pred)

        # estimate state
        self.x_est, self.P_est = self.estimate_state(x_pred, P_pred, yk)
        self.all_state_estimates.append(np.copy(self.x_est))
        self.all_measurements.append(yk)

        # predict into when we will start the next control
        x_pred_1 = np.copy(self.x_est)
        P_pred_1 = np.copy(self.P_est)
        for i in np.arange(self.control_delay+1)[::-1]:
            self.model.params.u = lambda t: self.full_control_vector[self.iteration-i]
            x_pred_1, P_pred_1 = self.predict_state(x_pred_1, P_pred_1, self.model, self.period)

        # update full control vector
        target_now = self.full_target[self.iteration+self.control_delay:self.iteration+self.horizon+self.control_delay]
        control_vector = self.optimize_binary_control(target_now, x_pred_1, P_pred_1)
        self.full_control_vector[self.iteration+1:self.iteration+1+self.horizon]=control_vector
        self.iteration += 1
        self.control_vector_0 = control_vector[0]
        return self.control_vector_0

    def plot(self):
        '''
        Plot the current state of the MPC
        '''
        tvec_state = self.period*np.arange(len(self.all_state_estimates))
        tvec_targ = np.arange(len(self.full_target))*self.period
        all_states = np.array(self.all_state_estimates)
        f = plt.figure(figsize=(5,3))
        ax = f.add_axes([0.2,0.3,.8,.45])
        ax.plot(tvec_targ,self.full_target/self.C[0,1],'k--')
        ax.plot(tvec_state,all_states[:,1]/self.C[0,1],'indianred')
        ax.set_xlabel('time [min]')
        ax.set_ylabel('# molecules')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax2 = f.add_axes([.25, .8, .7, .1 ])
        ax2.imshow(np.atleast_2d(self.full_control_vector), cmap='Blues', aspect='auto', vmin=0, vmax=3)
        ax2.set_yticks([])
        ax2.set_yticklabels([])
        ax2.set_xticks([0,len(tvec_targ)])
        _ = ax2.set_xticklabels([tvec_targ[0],tvec_targ[-1]])

        f2 = plt.figure(figsize=(5,3))
        ax = f2.add_axes([0.2,0.3,.8,.45])
        ax.plot(tvec_targ,self.full_target,'k--')
        ax.plot(tvec_state,self.all_measurements,'indianred')
        ax.set_xlabel('time [min]')
        ax.set_ylabel('a.f.u.')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax2 = f2.add_axes([.25, .8, .7, .1 ])
        ax2.imshow(np.atleast_2d(self.full_control_vector), cmap='Blues', aspect='auto', vmin=0, vmax=3)
        ax2.set_yticks([])
        ax2.set_yticklabels([])
        ax2.set_xticks([0,len(tvec_targ)])
        _ = ax2.set_xticklabels([tvec_targ[0],tvec_targ[-1]])