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
        model.params.x0 = np.concatenate((x,P.ravel()))
        model.params.tvec = np.linspace(0,dt,10)
        soln = model.solve()
        return soln.mean[:,-1], soln.covariances[:,:,-1]

    def estimate_state(self, x, P, yk ):
        '''Estimate the state based on current state
        x and covariance P, using a measurement yk.
        Returns estimated x and p
        '''
        d = x.shape[0]
        K = np.dot( P, self.C.T).dot(np.linalg.pinv(np.dot(self.C, P).dot(self.C.T)+self.R))
        x_est = x + K.dot(yk-np.dot(self.C,x))
        P_est = (np.eye(d) - np.dot(K,self.C)).dot(P)
        return x_est, P_est

class MPC(StateEstimation):
    def __init__(self, model, period, mu_rfp, sigma_rfp, pk0, horizon=None, full_target=None):
        self.mu_rfp = mu_rfp 
        self.sig_rfp = sigma_rfp 
        self.period = period
        self.full_control_vector = np.zeros(len(full_target))
        self.full_control_vector[0] = 0
        self.full_control_vector[1] = 0
        self.model = model
        self.iteration = 1
        self.initial_iteration = 0
        self.all_state_estimates = []
        self.all_measurements = []
        self.pk = pk0
        self.all_pk = []
        self.control_delay = 0
        if horizon is None:
            self.horizon = 4
        else:
            self.horizon = horizon
        if full_target is None:
            self.full_target = np.concatenate((800*np.ones(30),1500*np.ones(50)))
        else:
            self.full_target = full_target

    def optimize_light_signal_exhaustive(self, p_pred, target_nows):
        '''Exhaustively optimize the light signal. 
        p_pred should be a marginal distribution.
        '''
        all_control_vectors = [list(i) for i in itertools.product([0, 1], repeat=self.horizon)]
        # eAs = [self.model.eA_off, self.model.eA_on]
        all_costs = []
        for control_vector in all_control_vectors:
            pnow_tmp = np.copy(p_pred)
            costs = []
            for i,target_now in enumerate(target_nows):
                self.model.params.pi = pnow_tmp
                _ = self.model.solve_fsp(self.model.As[control_vector[i]])
                pnow_tmp = self.model.soln[:,-1]
                costs.append(0)
                costs.append(self.get_target_metric(self.model.get_observable(pnow_tmp), target_now))
            all_costs.append(np.sum(costs))   
        return all_control_vectors[np.argmin(all_costs)]

    def bayesian_update(self, pnm, xk ):
        '''Update state distribution according to current measuremnt and
        uncertainty.
        '''
        # check if xk is above n_protein*mu_rfp. 
        if xk >= self.model.n_protein*self.mu_rfp:
            xk = self.model.n_protein*self.mu_rfp-1
        if xk<0:
            print('NEGATIVE MEASUREMENT {0}, taking absolute value'.format(xk))
            xk = np.abs(xk)
        n = np.arange(1,self.model.n_protein)
        fn = np.arange(0,self.model.n_protein)
        pxn = (1/np.sqrt(2*np.pi*self.sig_rfp*n))*np.exp(-.5*((xk-n*self.mu_rfp)**2)/n/self.sig_rfp) 
        pxn = np.concatenate(([np.min(pxn)],pxn))
        # pnm = pnm.reshape(self.model.n_protein,self.model.n_rna)
        pnm = pnm.reshape(self.model.tensor_shape).T
        # print('predicted mean state: {0}'.format(np.sum(fn*pnm.sum(axis=1))))
        # print('measured mean state: {0}'.format(np.sum(fn*pxn)))
        # pnmx = pnm*pxn[:,None] / np.sum(pnm*pxn[:,None])
        pnmx = pnm*pxn / np.sum(pnm*pxn)
        return pnmx.T

    def get_target_metric(self, p, target_n):
        '''Given the target number of molecules k, compute a metric
        that we want to optimize (i.e. mean deviation)
        '''
        n_molecule_vals = np.arange(len(p))
        return np.sum(np.abs(n_molecule_vals-target_n)*p)

    def update_control_vector(self, yk):
        '''When a new mesurement comes in, use this function to
        update the state estimate, make predictions, etc
        '''
        # predict current state from previous state
        self.model.params.pi = self.pk.ravel()
        _ = self.model.solve_fsp(self.model.As[int(self.full_control_vector[self.iteration-1-self.control_delay])])
        self.pk = self.model.soln[:,-1]
        self.all_pk.append(np.copy(self.pk))
        # estimate current state 
        self.phatk = self.bayesian_update(self.pk, yk)
        self.all_state_estimates.append(np.copy(self.phatk))
        self.all_measurements.append(yk)

        # predict into when we will start the next control. 
        self.model.params.pi = self.phatk.ravel()
        #for i in np.arange(1,self.control_delay+1)[::-1]:
        for i in np.arange(self.control_delay+1)[::-1]:
            _ = self.model.solve_fsp(self.model.As[int(self.full_control_vector[self.iteration-i])])
            self.model.params.pi = self.model.soln[:,-1]
        pnow = self.model.soln[:,-1]

        # update full control vector
        target_now = self.full_target[self.iteration+self.control_delay:self.iteration+self.horizon+self.control_delay]
        control_vector = self.optimize_light_signal_exhaustive(pnow, target_now)
        self.full_control_vector[self.iteration+1:self.iteration+1+self.horizon] = control_vector
        self.iteration += 1
        self.control_vector_0 = control_vector[0]
        return (self.full_control_vector, control_vector[0])

    def get_target_likelihood(self, p_pred, xk):   
        return self.get_fluorescence_density(xk, p_pred)

    def gaussian(self, x, mu, sig):
        if var != 0.0:
            g = (1/np.sqrt(2*np.pi*sig))*np.exp(-.5*(((x-mu)**2)/sig))
        elif var == 0: 
            var = 1/(x[1]-x[0])
            g = (1/np.sqrt(2*np.pi*sig))*np.exp(-.5*(((x-mu)**2)/sig))  
        return g

    def get_fluorescence_density(self, x, p):
        f = np.zeros(x.shape)
        for n,pn in enumerate(p):  
            f += pn*self.gaussian(x,n*self.mu_rfp,n*sigm_rfp)
        return f   

    def plot(self):
        '''
        Plot the current state of the MPC
        '''
        all_states=np.array(self.all_measurements)/self.mu_rfp
        margerie = np.array(self.all_state_estimates).sum(axis=2)
        margerie_red = np.array(self.all_state_estimates).sum(axis=1)
        tvec = self.period*np.arange(len(all_states))

        tvec_targ = np.arange(len(self.full_target))*self.period
        full_target_fluor = self.full_target*self.mu_rfp
        all_states = np.array(all_states)
        f = plt.figure(figsize=(5,3))
        ax = f.add_axes([0.2,0.3,.8,.45])

        ax.plot(tvec_targ,full_target_fluor,'k--')
        ax.plot(tvec,np.array(self.all_measurements),'indianred')
        # ax.plot(tvec_targ,full_target,'k--')
        # ax.plot(tvec,all_states[:,1],'indianred')
        # ax.set_ylim([0,300])
        ax.set_xlabel('time [min]')
        ax.set_ylabel('a.f.u.')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax2 = f.add_axes([.25, .8, .7, .1 ])
        ax2.imshow(np.atleast_2d(self.full_control_vector), cmap='Blues', aspect='auto', vmin=0, vmax=3)
        ax2.set_yticks([])
        ax2.set_yticklabels([])
        ax2.set_xticks([0,len(tvec)])
        _ = ax2.set_xticklabels([tvec[0],tvec[-1]])

        f,ax = plt.subplots(figsize=(5,2))
        im = ax.imshow(margerie.T,cmap='Greens',origin='lower', aspect='auto')
        ax.plot(all_states[:,1],color='indianred')
        f.colorbar(im, pad=0.01, aspect=12)
        ax.set_title('protein number state estimation')
        ax.set_xlabel('time [min/6]')
        ax.set_ylabel('# proteins')
        f.tight_layout()
        predicted = np.array(self.all_pk)
        predicted = predicted.reshape(len(self.all_pk),self.model.n_protein,self.model.n_rna)
        print(predicted.shape)
        pred_marg = predicted.sum(axis=2)

        f,ax = plt.subplots(figsize=(5,2))
        im = ax.imshow(margerie_red.T,cmap='Reds',origin='lower', aspect='auto')
        ax.plot(all_states[:,0],color='indianred')
        f.colorbar(im, pad=0.01, aspect=12)
        ax.set_title('RNA number state estimation')
        ax.set_xlabel('time [min/6]')
        ax.set_ylabel('# rna')
        f.tight_layout()
        predicted = np.array(self.all_pk)
        predicted = predicted.reshape(len(self.all_pk),self.model.n_protein, self.model.n_rna)
        print(predicted.shape)
        pred_marg = predicted.sum(axis=2)

        f,ax = plt.subplots(figsize=(5,2))
        im = ax.imshow(pred_marg.T,cmap='Greens',origin='lower', aspect='auto',vmax=0.2)
        ax.plot(all_states[:,1],color='indianred')
        f.colorbar(im, pad=0.01, aspect=12)
        ax.set_title('protein number state prediction')
        ax.set_xlabel('time [min/6]')
        ax.set_ylabel('# proteins')
        f.tight_layout()


    def run_simulated_control(self, ssa, nit, dt):
        '''
        run a contorl experiment. on simulated data
        '''
        from tqdm.notebook import tqdm
        self.control_delay = 0 # number of dt's before the controller acts on the system. 
        self.full_control_vector[0] = 1
        all_states = [copy.copy(ssa.params.x0)]
        all_fluors = [copy.copy(ssa.params.x0)[self.model.observables]*self.mu_rfp + copy.copy(ssa.params.x0)[self.model.observables]*np.random.randn()*np.sqrt(self.sig_rfp) ]
        data_now_fluor = all_fluors[0]
        data_now = np.array(copy.copy(ssa.params.x0))
        for i in tqdm(range(0,nit)):
            ssa.params.u = lambda t: self.full_control_vector[i-self.control_delay]
            ssa.params.tvec = np.linspace(0,dt,10)
            data_now = ssa.solve_ssa_from_state(data_now[:,np.newaxis,np.newaxis])[:,-1,0] 
            data_now_fluor = np.abs(data_now[-1]*self.mu_rfp + data_now[-1]*np.random.randn()*np.sqrt(self.sig_rfp))
            all_states.append(data_now)
            all_fluors.append(data_now_fluor)
            _ = self.update_control_vector( data_now_fluor )
        return all_states, all_fluors    