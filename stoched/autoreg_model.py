import numpy as np
from stoched.utils.utilities import model_utils
from stoched.utils.generic_solvers import GenericSSA, GenericMoments, GenericFSP
from importlib import reload
from sympy import Matrix,zeros,Symbol
from stoched.utils.expv import expv
from scipy.sparse.linalg import onenormest
import cma
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.sparse import spdiags


class Model(model_utils.Model):
    def __init__(self,params):
        self.params = params
        self.observables = np.array([0])
        self.N = int(1)
    
    def get_matrices(self):
        #self.S = np.array([[ 1, -1]]).T
        self.S = np.array([[ 1, -1]])
        self.propensity_vec = lambda x,t: np.array([self.params.kb*self.params.u(t)+(self.params.kmax*x[0]**self.params.h)/(self.params.km**self.params.h+(x[0])**self.params.h), self.params.gamma*x[0]])

    def ode_func(self,t,x):
        '''
        '''
        dxdt = self.params.kb+(self.params.kmax*x[0]**self.params.h)/(self.params.km**self.params.h+(x[0])**self.params.h) - self.params.gamma*x[0]
        return [dxdt]

    def solve_mean(self, params=None, tvec=None):
        '''
        solve the model, with the current parmaeter object, i guess. 
        '''
        if params is not None:
            self.params=params
        if hasattr(self.params,'tvec'):
            tvec = self.params.tvec
        elif tvec is None:
            tvec = np.linspace(0,10,10)
        f = lambda t,x: self.ode_func(t, x)
        sol = solve_ivp(f, (0,tvec[-1]), self.params.x0 , method='BDF', t_eval = tvec )
        return sol 

    def solve(self, params=None, tvec=None):
        '''
        solve the model, with the current parmaeter object, i guess. 
        '''
        if params is not None:
            self.params=params
        if hasattr(self.params,'tvec'):
            tvec = self.params.tvec
        elif tvec is None:
            tvec = np.linspace(0,10,10)
        f = lambda t,x: self.ode_func(t, x)
        sol = solve_ivp(f, (0,tvec[-1]), self.params.x0 , method='BDF', t_eval = tvec )
        return self.clean_solution(sol)

    def solve_ssa(self, params=None, tvec=None, n_trajectories=1):
        '''solve the model using the stochastic simulation algorithm. 
        '''
        ssa = GenericSSA(type='nonlinear')
        self.get_matrices()
        ssa.S = self.S
        #ssa.get_P = lambda x,t: np.dot(self.W1,np.atleast_2d(x).T) + self.W0(t)
        ssa.get_P = lambda x,t: self.propensity_vec(x,t) 
        ssa.tvec = self.params.tvec
        ssa.ptimes = len(self.params.tvec)
        ssa.tf = self.params.tvec[-1]
        ssa.ti = self.params.tvec[0]
        #ssa.xi = self.params.x0[:2]
        ssa.xi = self.params.x0
        soln = ssa._solve(n_trajectories)
        return soln

    def solve_ssa_from_state(self, states, params=None, tvec=None):
        '''solve the ssa from a given set of initial states; one trajectory 
        per initial state. 
        '''
        n_states = states.shape[2]
        all_solutions = np.zeros((states.shape[0], 1, n_states))
        for i in range(n_states):
            self.params.x0 = states[:,0,i]
            soln = self.solve_ssa()
            all_solutions[:,:,i] = soln[:,-1,0][:,np.newaxis]
        return all_solutions

    def clean_solution(self,sol):
        '''
        takes the solution of the moment equations and reformats them into a more 
        convenient shape. 
        '''
        ntimes = len(sol.t)
        sol.mean = np.zeros((self.N,ntimes)) 
        sol.covariances = np.zeros((self.N,self.N,ntimes))
        for t in range(ntimes):
            sol.mean[:,t] = sol.y[:self.N,t]
            sol.covariances[:,:,t] =  np.reshape(sol.y[self.N:,t],(self.N,self.N))
        # only keep the species specified by the observables array.
        # sol.mean = self.mean[self.observables,:]
        # sol.covariances = np.array([self.covariances[self.observables,:,:]])[:,self.observables,:]
        return sol

    # def fit(self, data, loss='gaussian_likelihood'):
    #     '''
    #     fit the model to the data, starting from my current 
    #     parameter object. the data is an object.
    #     '''
    #     # get a parameter vector.
    #     self.params.tvec = data.tvec
    #     self.params.get_fit_vector()
    #     loss_func = self.__getattribute__(loss)
    #     min_func = lambda x:loss_func(x,data)
    #     results = minimize(min_func, self.params.fit_vec, bounds=[(1e-8,1e8)]*len(self.params.fit_vec),method='trust-constr')
    #     self.params.set_fit_vector(results.x) 
    #     results.params =  np.copy(self.params)
    #     return results

    def fit(self, data, loss='gaussian_likelihood', optimizer='cmaes', log_theta=True, bounds=None):
        '''
        fit the model to the data, starting from my current 
        parameter object. the data is an object.
        '''
        # get a parameter vector.
        #self.params.tvec = data.tvec
        self.params.get_fit_vector()
        loss_func = self.__getattribute__(loss)
        min_func = lambda x:loss_func(x, data, log_theta)
        optimizer_func = self.__getattribute__(optimizer)
        if bounds is None:
            bounds = [(1e-8,1e8)]*len(self.params.fit_vec)
        if log_theta:
            bounds = np.log10(bounds)
        results = optimizer_func(min_func, self.params.fit_vec, bounds, log_theta)
        print('fit complete' )
        print('likelihood: ', loss_func(results.x, data, False))
        print('parameters: ', results.x)
        self.params.set_fit_vector(results.x) 
        results.likelihood = loss_func(results.x, data, False)
        results.params =  np.copy(self.params)
        return results

    def least_squares(self, x, data):
        '''
        compute the least squares objective function
        '''
        self.params.vec = x
        self.params.set_vector()
        soln = self.solve()
        y = soln.y[0].ravel()/soln.y[0][0]
        return np.sum((data.X-y)**2)

    def cmaes(self, min_fun, x0, bounds, log_theta):
        '''
        fit using CMAES
        '''
        if log_theta:
            x0 =  np.log10(x0)
        es = cma.CMAEvolutionStrategy( x0, 0.5, {'verb_disp': 0})
        es.optimize(min_fun, iterations=100)
        es.plot()
        if log_theta:
            es.best.x = 10**es.best.x
        return es.best

    def gaussian_likelihood(self, x, data, log_theta):
        ''' 
        find the likelihood
        '''
        if log_theta:
            x = 10**x 
        n_trajectories,nt = data.X.shape
        self.params.fit_vec = x
        self.params.set_fit_vector()
        soln = self.solve()
        soln = self.to_fluorescence(soln)
        lhood = 0
        # for j in range(n_trajectories):
        #     for i in range(1,nt):
        #         lhood +=  .5 * ((np.log(np.pi*2*soln.covariances[1,1,i])+(data.X[j,i]-soln.mean[1,i])**2 /soln.covariances[1,1,i]))
        lhood = np.nansum(   ((np.log(np.sqrt(soln.cov_fluor[:,None]))+(data.X-soln.mean_fluor[:,None])**2 /soln.cov_fluor[:,None]/.5)) )
        return lhood 

    def to_fluorescence(self, soln):
        '''
        convert mean and variance to fluorescence values. 
        '''
        soln.mean_fluor = self.params.mu+soln.mean[1,:]
        soln.cov_fluor = self.params.sigma+soln.covariances[1,1,:]
        return soln

    # def get_fim(self, order=2, log=False):
    #     ''' 
    #     ''' 
    #     vec = self.params.get_vector()
    #     self.params.set_vector(vec)
    #     mom = GenericMoments()
    #     self.get_sym_matrices()
    #     self.get_matrices()
    #     mom.W0_sym = self.W0_sym
    #     mom.W1_sym = self.W1_sym 
    #     mom.S_sym = self.S_sym
    #     mom.S = self.S
    #     mom.params = self.params.vec
    #     mom.tvec = self.params.tvec
    #     mom.theta_vec = self.theta_vec
    #     mom.time_varying = self.params.u
    #     mom.dpars = self.params.free_parameters
    #     mom.W0 = self.W0
    #     mom.W1 = self.W1
    #     mom.N = 2 
    #     mom.solve = self.solve
    #     mom.param_dict = self.param_dict
    #     mom.observables = np.array([1])
    #     mom.N_observables = 1
    #     #mom.tv_dict = {'t': self.params.u}
    #     mom.tv= True
    #     mom.block_vars = np.diag(np.ones(len(self.params.tvec)))
    #     def get_W(t):
    #         mom.tv_dict = {'t': self.params.u(t)}

    #     mom.get_W = get_W

    #     mom.get_FIM(order=order, log=log)
    #     self.FIM = mom.FIM
    #     return mom

    def get_A(self, n_protein, u=1):
        '''Get the generator matrix. 
        '''
        self.tensor_shape = (n_protein,)
        self.n_protein = (n_protein)
        self.bounds = n_protein
        N = n_protein
        X = np.arange(n_protein)
        production = self.params.kb*u+(self.params.kmax*X**self.params.h)/(self.params.km**self.params.h+(X)**self.params.h)
        degradation = X*self.params.gamma
        main = -1*(production+degradation)
        #translation[n_rna-1::n_rna] = 0
        A = spdiags((main, production, degradation),(0,-1,1),N,N)
        self.A = A
        return A

    def get_A_v2(self, n_protein, u=1):
        '''Get the generator matrix. 
        '''
        self.tensor_shape = (n_protein,)
        self.n_protein = (n_protein)
        self.bounds = n_protein
        N = n_protein
        X = np.arange(n_protein)
        production = self.params.kb+(self.params.kmax*X**(self.params.h+u))/(self.params.km**(u+self.params.h)+(X)**(self.params.h+u))
        degradation = X*self.params.gamma
        main = -1*(production+degradation)
        #translation[n_rna-1::n_rna] = 0
        A = spdiags((main, production, degradation),(0,-1,1),N,N)
        self.A = A
        return A

    def get_observable(self, p):
        return p.reshape(self.tensor_shape) 

    def solve_fsp(self, A):
        '''solve the FSP
        '''
        N = A.shape[0]
        self.soln = np.zeros((N,len(self.params.tvec)))
        self.soln[:,0] = self.params.pi
        pv = np.copy(self.params.pi)
        n=int(N)
        m=30
        tol = 1e-8
        w = np.ones(n,dtype=np.float64)
        anorm = onenormest(A)
        wsp = np.zeros(7+n*(m+2)+5*(m+2)*(m+2),dtype=np.float64)
        iwsp = np.zeros(m+2,dtype=np.int32)
        for i in range(1,len(self.params.tvec)):
            try:
                pv,m,v = expv(self.params.tvec[i]-self.params.tvec[i-1],A,pv,tol = tol,m=30)
            # pv,tol0,iflag0 = dgexpv(30,self.tvec[i]-self.tvec[i-1],pv,1e-7,anorm,wsp,iwsp,self.A.dot,0)
            except:
                print('issue with FSP solving...')
                if np.sum(pv)<.99:
                    print('...leaking probability')
                if np.sum(pv) == 0.0:
                    print('...leaked all probability')
                print('acting as if there was no new solve, no propogation of distribution')
                pv = pv
            self.soln[:,i] = pv
        tensor_data = self.soln.reshape(self.bounds, len(self.params.tvec))
        # self.p = tensor_data.sum(axis=1)[:,-1]
        return tensor_data

    def solve_fsp_ts(self, p0 = None):
        '''
        Solve for the probability distribution
        '''
        # get the initial condition
        pi = self.params.pi
        # assume for now that cells start with 0 RNA
        A_on = self.get_A(self.n_protein, u=1)
        A_off = self.get_A(self.n_protein, u=0)
        N = A_on.shape[0]
        tvec = np.copy(self.params.tvec)
        ptimes = np.copy(self.params.tvec)
        all_solns = np.zeros((N,1))
        all_solns[:, 0] =  pi
        all_tvec = []
        self.uv_signal = [self.initial_uv]
        if len(self.switch_vec) == 0:
            if self.initial_uv:
                _ = self.solve_fsp(A_on)
            else:
                _ = self.solve_fsp(A_off)
            self.tensor_data = self.soln.reshape(self.n_protein,self.soln.shape[1])
        else:
            for i in range( len(self.switch_vec)-1 ):
                if i>0:
                    self.params.pi = self.soln[:,-1]
                # ton = self.switch_vec[i]+self.params.delay
                # toff = self.switch_vec[i+1]+self.params.delay
                ton = self.switch_vec[i]
                toff = self.switch_vec[i+1]
                tvec_in_range = tvec[(tvec>=ton) & (tvec<=toff)]
                self.params.tvec = np.unique(np.sort(np.concatenate(([ton,toff],tvec_in_range))))
                all_tvec = np.concatenate((all_tvec,self.params.tvec))
                if (i%2 + self.initial_uv):
                    self.uv_signal.append(0)
                    _ = self.solve_fsp(A_off)
                    all_solns = np.hstack((all_solns, self.soln))
                else:
                    self.uv_signal.append(1)
                    _ = self.solve_fsp(A_on)
                    all_solns = np.hstack((all_solns, self.soln))
                # all_solns = np.hstack((all_solns, self.soln))

            # construct data tensor, make tvec.
            all_tvec = np.concatenate( (all_tvec, [tvec[-1]]) ) 
            tvec_tmp,inds = np.unique(all_tvec,return_index=True)
            self.tensor_data = all_solns.reshape(self.n_protein, all_solns.shape[1])
            #self.tensor_data = self.tensor_data[:,:,1:]
            self.tensor_data = self.tensor_data[:,:,inds]

            # get the solution just at the print times. 
            inds = self.get_original_print_times(tvec_tmp, ptimes)
            self.params.tvec = tvec_tmp[inds] 
            self.tensor_data = self.tensor_data[:,:,inds]

        self.p = self.tensor_data

        # self.mean_x = np.sum(np.atleast_2d(np.arange(self.n_protein)).T*self.pX, axis=0)
        # self.mean_y = np.sum(np.atleast_2d(np.arange(self.tMY)).T*self.pY, axis=0)

        # self.var_x = np.sum(np.atleast_2d(np.arange(self.tMX)**2).T*self.pX, axis=0)-self.mean_x**2
        # self.var_y = np.sum(np.atleast_2d(np.arange(self.tMY)**2).T*self.pY, axis=0)-self.mean_y**2

    def get_original_print_times(self, full_times, print_times):
        ''' get the indices of the print times.  
        '''
        indices = []
        for i,t in enumerate(full_times):
            if len(np.where( t == print_times )[0]) > 0: 
                indices.append( i ) 
        return indices

    def get_time_vector_u(self, tvec, switch_vec, initial_uv):
        ''' Get a time vector.
        '''
        if len(switch_vec) == 0:
            return np.repeat(initial_uv,len(tvec))
        else:
            uv = np.zeros(len(tvec))
            uv_now = initial_uv 
            count = 0 
            for i,time in enumerate(tvec): 
                if time<=switch_vec[count]:
                    uv[i] = uv_now 
                else: 
                    count += 1
                    uv_now = not(uv_now)
                    uv[i] = uv_now
            return uv

    def gaussian(self, x, mu, var):
        if var != 0.0:
            g = (1/np.sqrt(2*np.pi*var))*np.exp(-.5*(((x-mu)**2)/var))
        elif var == 0: 
            g=0
            # var = 1/(x[1]-x[0])
            # g = (1/np.sqrt(2*np.pi*var))*np.exp(-.5*(((x-mu)**2)/var))   
        return g
    
    def get_fluorescence_density(self, x, p, mu, var, shift=0):
        f = np.zeros(x.shape)
        for n,pn in enumerate(p):  
            f += pn*self.gaussian(x,n*mu,n*var)
        return f   
    
    def get_fluorescence_moments(self, x, f):
        dx = x[1]-x[0]
        mu = np.sum(np.atleast_2d(x).T*f,axis=0)*dx
        sigma = np.sum(np.atleast_2d(x).T**2 * f, axis=0)*dx - mu**2
        return mu, sigma
