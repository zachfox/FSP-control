import numpy as np
import matplotlib.pyplot as plt

import re
import inspect #this pulls the function strings out of the lambda function for parser
import ast
import operator as op
import sys

class SaveTools:
    def save(self):
        self.save_to_directory('out_files')
        pass

    def save_to_directory(self, directory_name):
        import cloudpickle
        from pathlib import Path
        root = Path().absolute()
        directory_path = root / directory_name
        if directory_path.exists() != 1:
            directory_path.mkdir(parents=True, exist_ok=True)
        file_path = directory_path / self._generate_stamp_name()
        open_file = open(file_path, 'wb')
        cloudpickle.dump(self, open_file)
        pass

    def _generate_stamp_name(self):
        timestamp = self._generate_timestamp()
        name = self.__class__.__name__
        stamp = name + "_" + timestamp
        return stamp

    def _generate_timestamp(self):
        from datetime import datetime
        current_time = datetime.now()
        timestamp = current_time.strftime("%a_%b_%Y_%H_%M_%f")
        self._sleep_one_microsecond
        return timestamp

    def _sleep_one_microsecond(self):
        from time import sleep
        sleep(1 / 1000000.0)
        pass


class GenericSolver(SaveTools):
    def _add_metadata(self, data):
        data.meta['solver'] = self.__class__.__name__.replace('__', '')
        data.meta['date'] = self._generate_timestamp()
        data.meta['initial_state'] = self.initial_state
        data.meta['time'] = self.time
        return data

    def _generate_timestamp(self):
        from datetime import datetime
        current_time = datetime.now()
        timestamp = current_time.strftime("%a, %b, %Y, %H:%M:%S")
        return timestamp


class SolverSSA(GenericSolver):
    initial_state = None
    time = None
    stoich_matrix = None
    propensity = lambda t, x, p: np.array(None)
    parameters = None
    num_samples = 1
    precision = np.float

    def __init__(self):
        # todo
        pass

    def run(self):
        self._initialize_integration()
        trajectory_list = []
        for i in range(0, self.num_samples):
            single_trajectory = self._perform_ssa_integration()
            trajectory_list.extend([single_trajectory])
        trajectory_object = SSAData(trajectory_list)
        return trajectory_object

    def _perform_ssa_integration(self):
        time_series_data = self._integrate_time_series()
        time_series_data = self._format_trajectory(time_series_data)
        time_series_data = self._add_metadata(time_series_data)

        return time_series_data

    def _integrate_time_series(self):
        if self._is_time_sampled():
            time_series_data = self._integrate_sampled_time_series()
        else:
            time_series_data = self._integrate_unsampled_time_series()
        return time_series_data

    def _integrate_unsampled_time_series(self):
        time_list, state_list = self._initialise_linked_list()
        final_time = np.atleast_1d(self.time[-1])
        current_time = self.time[0]
        current_state = self.initial_state
        while current_time < final_time:
            cumulative_rates = self._get_cumulative_rates(current_time, current_state)
            new_time = self._step_to_new_time(current_time, cumulative_rates)
            new_state = self._step_to_new_state(current_state, cumulative_rates)
            current_state = new_state
            current_time = new_time
            time_list.append(current_time)
            state_list.append(current_state)

        time_series_data = self._post_process_list(time_list, state_list)
        return time_series_data

    def _integrate_sampled_time_series(self):
        state_array = np.zeros([len(self.time), len(self.initial_state)])
        final_time = np.atleast_1d(self.time[-1])
        current_time = self.time[0]
        current_state = self.initial_state
        state_array[0,:] = current_state
        index = 1
        while current_time < final_time:
            cumulative_rates = self._get_cumulative_rates(current_time, current_state)
            new_time = self._step_to_new_time(current_time, cumulative_rates)
            new_state = self._step_to_new_state(current_state, cumulative_rates)
            current_time = new_time
            while (current_time >= self.time[index]) & (current_time < final_time):
                state_array[index, :] = current_state
                index += 1
            current_state = new_state
        if index < len(self.time):
            state_array[index:,:] = current_state
        time_series_data = self._post_process_array(self.time, state_array)
        return time_series_data

    def _initialize_integration(self):
        self._transform_variables_to_array()
        self._check_entry_conditions()
        pass

    def _post_process_list(self, time_list, state_list):
        time_list, state_list = self._cleanup_infinity_exit(time_list, state_list)
        time_series_data = self._list_to_data(time_list, state_list)
        return time_series_data

    def _list_to_data(self, time_list, state_list):
        time_array = np.asarray(time_list).astype(np.double)
        state_array = np.asarray(state_list).astype(np.double)
        data = SSANode(time_array, state_array)
        return data

    def _post_process_array(self, time_array, state_array):
        self._cleanup_infinity_exit(time_array, state_array)
        time_series_data = SSANode(time_array, state_array)
        return time_series_data

    def _get_cumulative_rates(self, time, state):
        instantaneous_propensity = self._call_propensity(time, state)
        cumulative_rates = np.cumsum(instantaneous_propensity)
        return cumulative_rates

    def _initialise_linked_list(self):
        from collections import deque
        time_list = deque()
        time_list.append(self.time[0])
        state_list = deque()
        state_list.append(self.initial_state)
        return time_list, state_list

    def _step_to_new_time(self, time, cumulative_sum):
        new_time = time - np.log(np.random.rand()) / cumulative_sum[-1]
        return new_time

    def _step_to_new_state(self, state, rate):
        event = np.where(rate > (rate[-1] * np.random.rand()))
        if len(event[0]) == 0:
            new_state = None
        else:
            new_state = state + self.stoich_matrix[:, event[0][0]]
        return new_state

    def _check_entry_conditions(self):
        self._check_type()
        self._check_dimension()
        pass

    def _transform_variables_to_array(self):
        self.time = np.atleast_1d(self.time).astype(self.precision).flatten()
        self.initial_state = np.atleast_1d(self.initial_state).astype(self.precision).flatten()

        self.parameters = np.atleast_1d(self.parameters).astype(self.precision).flatten()
        self.stoich_matrix = np.atleast_2d(self.stoich_matrix).astype(np.int64)
        pass

    def _call_propensity(self, time, state):
        return np.atleast_1d(self.propensity(time, state, self.parameters))

    def _check_type(self):
        assert (type(self.initial_state) is np.ndarray)
        assert (type(self.time) is np.ndarray)
        assert (type(self.stoich_matrix) is np.ndarray)
        assert (type(self.propensity) is type(lambda: None))
        pass

    def _check_dimension(self):
        assert (self.time.ndim == 1)
        assert (self.initial_state.ndim == 1)
        pass

    def _cleanup_infinity_exit(self, time_list, state_list):
        if self._is_infinity_exit_condition(time_list):
            time_list, state_list = self._trim_infinity_exit_condition(time_list, state_list)
        return time_list, state_list

    def _is_infinity_exit_condition(self, time_list):
        if time_list[-1] == np.inf:
            return 1
        else:
            return 0

    def _trim_infinity_exit_condition(self, time_list, state_list):
        time_list.pop()
        state_list.pop()
        return time_list, state_list

    def _format_trajectory(self, time_series):
        # do nothing
        return time_series

    def _is_time_sampled(self):
        if self.time.size > 2:
            return 1
        else:
            return 0

    def _add_metadata(self, data):
        data = GenericSolver._add_metadata(self, data)
        data.meta['stoich_matrix'] = self.stoich_matrix
        data.meta['propensity'] = self.propensity
        data.meta['parameters'] = self.parameters
        data.meta['num_samples'] = self.num_samples
        return data


class IntegrateFSP(GenericSolver):
    initial_state = None
    time = None
    inf_generator = None
    parameters = None
    precision = np.float
    tolerance = 1e-10

    def run(self):
        self._initialise_integration()
        time_series_data = self._integrate_time_series()
        time_series_data = self._add_metadata(time_series_data)
        return time_series_data

    def _initialise_integration(self):
        self._sanitize_variables()
        self._check_dimension()
        self._check_type()
        pass

    def _sanitize_variables(self):
        self.time = np.atleast_1d(self.time).astype(self.precision)
        self.initial_state = np.atleast_1d(self.initial_state).astype(self.precision)
        self.parameters = np.atleast_1d(self.parameters).astype(self.precision)
        pass

    def _check_dimension(self):
        assert (self.time.ndim == 1)
        assert (self.initial_state.ndim == 1)
        assert (self._call_inf_generator().ndim == 2)
        assert (self.parameters.ndim == 1)
        pass

    def _check_type(self):
        assert (type(self.initial_state) is np.ndarray)
        assert (type(self.time) is np.ndarray)
        assert (type(self.parameters) is np.ndarray)
        pass

    def _integrate_time_series(self):
        from scipy.integrate import odeint
        this_inf_generator = self._call_inf_generator()
        rate_function = self._call_reaction_rates(this_inf_generator)
        state_time_series = odeint(rate_function, self.initial_state, self.time)
        time_series_data = GenericData(self.time, state_time_series)
        return time_series_data

    def _call_inf_generator(self):
        inf_generator_instance = self.inf_generator(self.parameters)
        return np.asarray(inf_generator_instance).astype(self.precision)

    def _call_reaction_rates(self, inf_generator):
        def rate_function(state, time):
            return inf_generator.dot(state)

        return rate_function

    def _add_metadata(self, data):
        data = GenericSolver._add_metadata(self, data)
        data.meta['inf_generator'] = self._call_inf_generator()
        data.meta['parameters'] = self.parameters
        return data


class GenerateFSP:
    stoich_matrix = None
    dimensions = None
    propensity = lambda t, x, p: None
    parameters = None
    time = None
    error = None
    state_mapping = None

    def get_inf_generator(self):
        self._check_inputs()
        self._sanitize_variables()
        inf_generator = self.get_inf_gen_tensor()
        inf_generator = self._post_process(inf_generator)
        return inf_generator

    def _check_inputs(self):
        # todo
        pass

    def _check_types(self):
        assert (type(self.stoich_matrix) == np.ndarray)
        assert ((type(self.dimensions) == np.ndarray) |
                (type(self.dimensions) == tuple) |
                (type(self.dimensions) == list))
        assert ((type(self.propensity)) == type(lambda: None))
        assert ((type(self.parameters) == np.ndarray) |
                (type(self.parameters) == tuple) |
                (type(self.parameters) == list))
        pass

    def _check_dimensions(self):
        assert (len(self.dimensions) > 0)
        assert (len(self.dimensions) < 5)
        pass

    def _sanitize_variables(self):
        self._initialize_function_map()
        pass

    def _initialize_function_map(self):
        size = len(self.dimensions)
        if size == 0:
            raise Exception('dimension must have a minimum size of 1')
        elif size == 1:
            self.inf_gen_function = self._get_single_rxn_inf_gen_1d
            self._flatten_inf_gen = self._flatten_1d_inf_gen
        elif size == 2:
            self.inf_gen_function = self._get_single_rxn_inf_gen_2d
            self._flatten_inf_gen = self._flatten_2d_inf_gen
        elif size == 3:
            self.inf_gen_function = self._get_single_rxn_inf_gen_3d
            self._flatten_inf_gen = self._flatten_3d_inf_gen
        elif size == 4:
            self.inf_gen_function = self._get_single_rxn_inf_gen_4d
            self._flatten_inf_gen = self._flatten_4d_inf_gen
        elif size == 5:
            self.inf_gen_function = self._get_single_rxn_inf_gen_5d
            self._flatten_inf_gen = self._flatten_5d_inf_gen
        pass

    def _post_process(self, inf_generator):
        inf_generator = self._flatten_inf_gen(inf_generator)
        return inf_generator

    def _flatten_1d_inf_gen(self, inf_generator):
        tensor_dims = inf_generator.shape
        flat_inf_generator = self._preallocate_flat_inf_gen()
        ind = 0
        self.state_mapping = []
        for i in range(tensor_dims[1]):
            inf_slice = inf_generator[:, i]
            inf_vec = inf_slice.flatten()
            flat_inf_generator[:, ind] = inf_vec
            self.state_mapping.append([i])
            ind += 1
        return flat_inf_generator

    def _flatten_2d_inf_gen(self, inf_generator):
        tensor_dims = inf_generator.shape
        flat_inf_generator = self._preallocate_flat_inf_gen()
        ind = 0
        self.state_mapping = []
        for i in range(tensor_dims[2]):
            for j in range(tensor_dims[3]):
                inf_slice = inf_generator[:, :, i, j]
                inf_vec = inf_slice.flatten()
                flat_inf_generator[:, ind] = inf_vec
                self.state_mapping.append([i, j])
                ind += 1
        return flat_inf_generator

    def _flatten_3d_inf_gen(self, inf_generator):
        tensor_dims = inf_generator.shape
        flat_inf_generator = self._preallocate_flat_inf_gen()
        ind = 0
        self.state_mapping = []
        for i in range(tensor_dims[3]):
            for j in range(tensor_dims[4]):
                for k in range(tensor_dims[5]):
                    inf_slice = inf_generator[:, :, :, i, j, k]
                    inf_vec = inf_slice.flatten()
                    flat_inf_generator[:, ind] = inf_vec
                self.state_mapping.append([i, j, k])
                ind += 1
        return flat_inf_generator

    def _flatten_4d_inf_gen(self, inf_generator):
        tensor_dims = inf_generator.shape
        flat_inf_generator = self._preallocate_flat_inf_gen()
        ind = 0
        self.state_mapping = []
        for i in range(tensor_dims[4]):
            for j in range(tensor_dims[5]):
                for k in range(tensor_dims[6]):
                    for l in range(tensor_dims[7]):
                        inf_slice = inf_generator[:, :, :, :, i, j, k, l]
                        inf_vec = inf_slice.flatten()
                        flat_inf_generator[:, ind] = inf_vec
                self.state_mapping.append([i, j, k, l])
                ind += 1
        return flat_inf_generator

    def _flatten_5d_inf_gen(self, inf_generator):
        tensor_dims = inf_generator.shape
        flat_inf_generator = self._preallocate_flat_inf_gen()
        ind = 0
        self.state_mapping = []
        for i in range(tensor_dims[5]):
            for j in range(tensor_dims[6]):
                for k in range(tensor_dims[7]):
                    for l in range(tensor_dims[8]):
                        for m in range(tensor_dims[9]):
                            inf_slice = inf_generator[:, :, :, :, :, i, j, k, l, m]
                            inf_vec = inf_slice.flatten()
                            flat_inf_generator[:, ind] = inf_vec
                self.state_mapping.append([i, j, k, l, m])
                ind += 1
        return flat_inf_generator

    def reconstruct_probability(self,flat_probability):
        probability = np.zeros(self.dimensions)
        for ind in range(len(self.state_mapping)):
            state = tuple(self.state_mapping[ind])
            probability[state]=flat_probability[ind]
        return probability

    def _preallocate_flat_inf_gen(self):
        total_dim = np.prod(self.dimensions)
        flat_inf_generator = np.zeros([total_dim, total_dim])
        return flat_inf_generator

    def _is_in_state_space(self, state):
        tf_bool = True
        for i in range(len(state)):
            if (state[i] >= self.dimensions[i]) | (state[i] < 0):
                tf_bool = False
        return tf_bool

    def _connect_states(self, from_state, to_state, state_space_stoich, reaction_index):
        reaction_rate = self._call_reaction_rate(self.time, from_state)
        to_index = tuple(to_state) + tuple(from_state)
        from_index = tuple(from_state) + tuple(from_state)
        state_space_stoich[to_index] += reaction_rate[reaction_index]
        state_space_stoich[from_index] -= reaction_rate[reaction_index]
        return state_space_stoich

    def _connect_states_exit(self,from_state,state_space_stoich, reaction_index):
        reaction_rate = self._call_reaction_rate(self.time, from_state)
        from_index = tuple(from_state) + tuple(from_state)
        state_space_stoich[from_index] -= reaction_rate[reaction_index]
        return state_space_stoich

    def _call_reaction_rate(self, time, state):
        this_reaction_rate = self.propensity(time, state, self.parameters)
        return this_reaction_rate

    def _preallocate_inf_gen(self):
        tensor_dimensions = []
        tensor_dimensions.extend(self.dimensions)
        tensor_dimensions.extend(self.dimensions)
        return np.zeros(tensor_dimensions)

    def _number_of_reactions(self):
        return self.stoich_matrix.shape[1]

    def get_inf_gen_tensor(self):
        state_space_stoich = self._preallocate_inf_gen()
        for reaction_index in range(self._number_of_reactions()):
            state_space_stoich += self.inf_gen_function(reaction_index)
        return state_space_stoich

    def _get_stoich_vector(self, reaction_index):
        return self.stoich_matrix[:, reaction_index]

    def _get_single_rxn_inf_gen_1d(self, reaction_index):
        state_space_stoich = self._preallocate_inf_gen()
        rxn_vec = self._get_stoich_vector(reaction_index)
        for state in range(self.dimensions[0]):
            if self._is_in_state_space(state + rxn_vec):
                state_space_stoich = self._connect_states([state], [state + rxn_vec], state_space_stoich,
                                                          reaction_index)
            elif ((self._is_in_state_space([state])) and ( not self._is_in_state_space(state+rxn_vec))):
                state_space_stoich = self._connect_states_exit([state], state_space_stoich,
                                                          reaction_index)
        return state_space_stoich

    def _get_single_rxn_inf_gen_2d(self, reaction_index):
        state_space_stoich = self._preallocate_inf_gen()
        rxn_vec = self._get_stoich_vector(reaction_index)
        for first_index in range(self.dimensions[0]):
            for second_index in range(self.dimensions[1]):
                state = [first_index, second_index]
                if self._is_in_state_space(state + rxn_vec):
                    state_space_stoich = self._connect_states(state, state + rxn_vec, state_space_stoich,
                                                              reaction_index)
                elif ((self._is_in_state_space(state)) and ( not self._is_in_state_space(state+rxn_vec))):
                    state_space_stoich = self._connect_states_exit(state, state_space_stoich,
                                                          reaction_index)

        return state_space_stoich

    def _get_single_rxn_inf_gen_3d(self, reaction_index):
        state_space_stoich = self._preallocate_inf_gen()
        rxn_vec = self._get_stoich_vector(reaction_index)
        for first_index in range(self.dimensions[0]):
            for second_index in range(self.dimensions[1]):
                for third_index in range(self.dimensions[2]):
                    state = [first_index, second_index, third_index]
                    if self._is_in_state_space(state + rxn_vec):
                        state_space_stoich = self._connect_states(state, state + rxn_vec, state_space_stoich,
                                                                  reaction_index)
                    elif ((self._is_in_state_space(state)) and ( not self._is_in_state_space(state+rxn_vec))):
                        state_space_stoich = self._connect_states_exit(state, state_space_stoich,
                                                              reaction_index)

        return state_space_stoich

    def _get_single_rxn_inf_gen_4d(self, reaction_index):
        state_space_stoich = self._preallocate_inf_gen()
        rxn_vec = self._get_stoich_vector(reaction_index)
        for first_index in range(self.dimensions[0]):
            for second_index in range(self.dimensions[1]):
                for third_index in range(self.dimensions[2]):
                    for fourth_index in range(self.dimensions[3]):
                        state = [first_index, second_index, third_index, fourth_index]
                        if self._is_in_state_space(state + rxn_vec):
                            state_space_stoich = self._connect_states(state, state + rxn_vec, state_space_stoich,
                                                                      reaction_index)
                        elif ((self._is_in_state_space(state)) and ( not self._is_in_state_space(state+rxn_vec))):
                            state_space_stoich = self._connect_states_exit(state, state_space_stoich,
                                                                  reaction_index)
        return state_space_stoich

    def _get_single_rxn_inf_gen_5d(self, reaction_index):
        state_space_stoich = self._preallocate_inf_gen()
        rxn_vec = self._get_stoich_vector(reaction_index)
        for first_index in range(self.dimensions[0]):
            for second_index in range(self.dimensions[1]):
                for third_index in range(self.dimensions[2]):
                    for fourth_index in range(self.dimensions[3]):
                        for fifth_index in range(self.dimensions[3]):
                            state = [first_index, second_index, third_index, fourth_index, fifth_index]
                            if self._is_in_state_space(state + rxn_vec):
                                state_space_stoich = self._connect_states(state, state + rxn_vec, state_space_stoich,
                                                                      reaction_index)
                            elif ((self._is_in_state_space(state)) and ( not self._is_in_state_space(state+rxn_vec))):
                                state_space_stoich = self._connect_states_exit(state, state_space_stoich,
                                                                      reaction_index)
        return state_space_stoich

    def _add_meta_data(self,data):
        data.meta['dimensions'] = self.dimensions
        data.meta['stoich_matrix'] = self.stoich_matrix
        return data

class SolverFSP():
    intial_state = None
    time = None
    stoich_matrix = None
    propensity=None
    parameters = None
    dimensions = None
    error = None
    generator = GenerateFSP()
    integrator = IntegrateFSP()

    def set_initial_state(self, new_initial_state):
        new_initial_state = np.array(new_initial_state).flatten()
        self.integrator.initial_state = new_initial_state
        pass

    def set_time(self, new_time):
        new_time = np.array(new_time)
        self.time = new_time
        self.generator.time = new_time
        self.integrator.time = new_time
        pass

    def set_stoich_matrix(self, new_stoich_matrix):
        self.stoich_matrix = np.atleast_2d(new_stoich_matrix)
        self.stoich_matrix = new_stoich_matrix
        self.generator.stoich_matrix = new_stoich_matrix
        pass

    def set_parameters(self, new_parameters):
        new_parameters = np.array(new_parameters)
        self.parameters = new_parameters
        self.generator.parameters = new_parameters
        self.integrator.parameters = new_parameters
        pass

    def set_dimensions(self, new_dimensions):
        new_dimensions=np.array(new_dimensions)
        self.dimensions = new_dimensions
        self.generator.dimensions = new_dimensions
        pass

    def set_time(self, new_time):
        self.time = new_time
        self.generator.time = new_time
        self.integrator.time = new_time
        pass

    def set_propensity(self,new_propensity):
        self.propensity=new_propensity
        self.generator.propensity=new_propensity
        pass

    def run(self):
        self._initialize_integration()
        data=self.integrator.run()
        data=self._format_time_series(data)
        return data

    def _initialize_integration(self):
        self.integrator._call_inf_generator = self.generator.get_inf_generator
        pass

    def _format_time_series(self,data):
        data = self._generic_data_to_fsp_data(data)
        data = self.generator._add_meta_data(data)
        return data

    def _generic_data_to_fsp_data(self,data):
        tensor=self._reconstruct_tensor(data)
        data = FSPData(self.time,tensor)
        return data

    def get_rate_function(self):
        rate_equation = self.generator.get_rate_function()
        return rate_equation

    def _reconstruct_tensor(self,data):
        tensor=[]
        for i in range(len(data.time)):
            tensor.append(self.generator.reconstruct_probability(data.state[i]))
        return tensor


class SolverODE(GenericSolver):
    initial_state = None
    time = None
    stoich_matrix = None
    propensity = lambda t, x, p: None
    parameters = None
    precision = np.float


    def run(self):
        self._initialize_integration()
        time_series_data = self._integrate_time_series()
        time_series_data = self._format_trajectory(time_series_data)
        time_series_data = self._add_metadata(time_series_data)
        return time_series_data

    def _initialize_integration(self):
        self._transform_variables_to_numpy_array()
        self._check_type()
        pass

    def _transform_variables_to_numpy_array(self):
        self.time = np.atleast_1d(self.time).astype(self.precision)
        self.initial_state = np.atleast_1d(self.initial_state).astype(self.precision)
        self.parameters = np.atleast_1d(self.parameters).astype(self.precision)
        self.stoich_matrix = np.atleast_2d(self.stoich_matrix).astype(self.precision)
        pass

    def _check_type(self):
        assert (type(self.initial_state) is np.ndarray)
        assert (type(self.time) is np.ndarray)
        assert (type(self.stoich_matrix) is np.ndarray)
        assert (type(self.parameters) is np.ndarray)
        pass

    def _integrate_time_series(self):
        from scipy.integrate import odeint
        state_time_series = odeint(self._call_reaction_rates, self.initial_state, self.time)
        time_series_data = ODEData(self.time, state_time_series)
        return time_series_data

    def _call_reaction_rates(self, state, time):
        reaction_rate = self.stoich_matrix.T.dot(self.propensity(time, state, self.parameters))
        reaction_rate_as_np = np.atleast_1d(reaction_rate).astype(np.float)
        return reaction_rate_as_np.flatten()

    def _format_trajectory(self, data):
        # do nothing
        return data

    def _add_metadata(self, data):
        data = GenericSolver._add_metadata(self, data)
        data.meta['stoich_matrix'] = self.stoich_matrix
        data.meta['parameters'] = self.parameters
        return data


class GenericData(SaveTools):
    time = None  # numpy array
    state = None  # numpy array
    meta = dict()  # Meta Object

    def __init__(self, *args):
        if len(args) == 3:
            self.time = args[0]
            self.state = args[1]
            self.meta = args[2]
        if len(args) == 2:
            self.time = args[0]
            self.state = args[1]
            self.meta = dict()
        if len(args) == 1:
            raise Exception('must have time and state input minimum to initialize data object')
        pass


class SSANode(GenericData):
    def __init__(self, *args):
        GenericData.__init__(self, *args)


class SSAData():
    node = None

    def __init__(self, ssa_list):
        self.node = ssa_list
        pass

    def __getitem__(self, index):
        return self.node[index]

    def __setitem__(self, index, new_node):
        self.node[index] = new_node
        pass

    def __len__(self):
        return len(self.node)

    def min_state(self, species_index):
        node_min = []
        for i in range(len(self)):
            node_min.append(self.node[i].state[:, species_index].min().astype(int))
        total_min = min(node_min)
        return total_min

    def max_state(self, species_index):
        node_max = []
        for i in range(len(self)):
            node_max.append(self.node[i].state[:, species_index].max().astype(int))
        total_max = min(node_max)
        return total_max

    def _get_histogram(self, data, bins):
        count_histogram, bin_edges = np.histogram(data, bins)
        probability_histogram = count_histogram / np.sum(count_histogram)
        return probability_histogram, bin_edges

    def __len__(self):
        return len(self.node)

    def _check_same_time(self):
        assert (self._is_same_time())

    def _is_same_time(self):
        is_same_time = True
        for i in range(0, len(self) - 1):
            if self.node[i].time != self.node[i + 1].time:
                is_same_time = False
        return is_same_time

    def trim_initial_condition(self, trim_index):
        self.node = self.node[trim_index:]
        pass

    def get_time_series(self, sample_index, species_index):
        time_series = self.node[sample_index].state[:, species_index]
        return time_series.flatten()

    def get_data_tensor(self):
        ntrajectories = self.__len__()
        ntimes,nspecies = self.node[0].state.shape
        data_tensor = np.zeros((ntimes,nspecies,ntrajectories))
        for i in range(ntrajectories):
            data_tensor[:,:,i] = self.node[i].state
        return data_tensor

    def get_time_series_mean(self, sample_index, species_index):
        trajectory_mean = np.mean(self.get_time_series(sample_index, species_index))
        return trajectory_mean

    def get_time_series_variance(self, sample_index, species_index):
        time_series_variance = np.var(self.get_time_series(sample_index, species_index))
        return time_series_variance

    def get_time_series_histogram(self, sample_index, species_index, bins='auto'):
        time_series_data = self.get_time_series(sample_index, species_index)
        probability_histogram, bin_edges = self._get_histogram(time_series_data, bins)
        return probability_histogram, bin_edges

    def get_snapshot(self, time_index, species_index):
        # self._check_same_time()
        snapshot = np.zeros(len(self.node))
        for i in range(0, len(self.node)):
            snapshot[i] = self.node[i].state[time_index, species_index]
        return np.asarray(snapshot).flatten()

    def get_snapshot_mean(self, time_index, species_index):
        snapshot_mean = np.mean(self.get_snapshot(time_index, species_index))
        return snapshot_mean

    def get_snapshot_variance(self, time_index, species_index):
        snapshot_variance = np.var(self.get_snapshot(time_index, species_index))
        return snapshot_variance

    def get_snapshot_histogram(self, time_index, species_index, num_bins='auto'):
        probability_histogram, bin_edges = self._get_histogram(self.get_snapshot(time_index, species_index), num_bins)
        return probability_histogram, bin_edges


class ODEData(GenericData):
    def __init__(self, *args):
        GenericData.__init__(self, *args)
        pass


class FSPData(GenericData):
    shape=None
    def __init__(self, *args):
        GenericData.__init__(self, *args)
        self.shape = self.state[0].shape
        self.rank = len(self.shape)
        pass

    def get_slice(self,time_index):
        slice=self.state[time_index]
        return slice

    def get_marginal_distribution(self,time_index,species_index):
        self._check_valid_index(species_index)
        marginal_distribution=self.get_slice(time_index)
        index=0
        for i in range(self.rank):
            if i != species_index:
                marginal_distribution=np.sum(marginal_distribution,index)
            else:
                index=index+1
        return marginal_distribution

    def _check_valid_index(self,index):
        assert(index < self.rank)
        pass

class VisualAnalysis():
    view = None
    model = None
    data = None
    _requires_update = True

    def __init__(self, model, view):
        self.model = model
        self.view = view
        pass

    def _update(self):
        if self._requires_update == True:
            self.data = self.model.run()
        self._requires_update = False
        pass

    def _intitialise_plotting(self):
        self._update()
        self._set_style()
        pass

    def _get_axes(self):
        if self.view == None:
            self.new_axes()
        pass

    def set_axes(self, new_axes):
        self.view = new_axes
        self._set_style()
        pass

    def clear_axes(self):
        self.view = None
        self.new_axes()
        pass

    def new_axes(self):
        self.view = plt.axes()
        self._set_style()
        pass

    def _set_style(self):
        plt.style.use('ggplot')
        plt.rcParams.update({'figure.autolayout': True})
        plt.tight_layout()
        self.view.autoscale(enable=True, axis='x', tight=True)
        pass

    def _generate_title(self):
        title = self.data[0].meta['solver'] + ' on ' + self.data[0].meta['date']
        return title

    def _set_generic_title(self):
        title = self._generate_title()
        self.view.set_title(title)
        pass

    def _generate_time_series_labels(self):
        self._set_generic_title()
        self.view.set_xlabel('Time')
        self.view.set_ylabel('Species Count')
        pass

    def _generate_histogram_labels(self, **kwargs):
        self._set_generic_title()
        self.view.set_xlabel('Species Count')
        self.view.set_ylabel('Probability')
        pass

class VisualSSA(VisualAnalysis):

    def set_time(self, new_time):
        self.model.time = new_time
        self._requires_update = True
        pass

    def set_initial_state(self, new_initial_state):
        self.model.initial_state = new_initial_state
        self._requires_update = True
        pass

    def set_parameters(self, new_parameters):
        self.model.parameters = new_parameters
        self._requires_update = True
        pass

    def set_stoich_matrix(self, new_stoich_matrix):
        self.model.stoich_matrix = new_stoich_matrix
        self._requires_update = True
        pass

    def set_propensity(self, new_propensity):
        self.model.propensity = new_propensity
        self._requires_update = True
        pass

    def set_cmap(self, new_cmap):
        self.view.cmap = new_cmap
        pass

    def plot_time_series(self):
        self._update()
        self._get_axes()
        for i in range(0, len(self.data)):
            self.view.step(self.data[i].time, self.data[i].state)
        self._generate_time_series_labels()
        pass

    def plot_snapshot_histogram(self, time_index, species_index, bin_width=1, *args, **kwargs):
        self._update()
        self._get_axes()
        bins = self._find_bins(species_index, bin_width)
        plot_data = self.data.get_snapshot(time_index, species_index)
        self.view.hist(plot_data, bins=bins, *args, **kwargs)
        self._generate_histogram_labels()
        pass

    def plot_time_series_histogram(self, sample_index, species_index, bin_width=1, *args, **kwargs):
        self._update()
        self._get_axes()
        bins = self._find_bins(species_index, bin_width)
        plot_data = self.data.get_time_series(sample_index, species_index)
        self.view.hist(plot_data, bins=bins, *args, **kwargs)
        self._generate_histogram_labels()
        pass

    def plot_snapshot_histogram_over_time(self, steps, species_index, bin_width=1, *args, **kwargs):
        self._update()
        self._get_axes()
        bins = self._find_bins(species_index, bin_width)
        for time_index in steps:
            plot_data = self.data.get_snapshot(time_index, species_index)
            self.view.hist(plot_data, bins=bins, *args, **kwargs)
        self._generate_histogram_labels()
        pass

    def _find_bins(self, species_index, bin_width):
        min_state = self.data.min_state(species_index)
        max_state = self.data.max_state(species_index)
        bins = range(min_state, max_state, bin_width)
        return bins

class VisualODE(VisualAnalysis):

    def set_time(self, new_time):
        self.model.time = new_time
        self._requires_update = True
        pass

    def set_initial_state(self, new_initial_state):
        self.model.initial_state = new_initial_state
        self._requires_update = True
        pass

    def set_parameters(self, new_parameters):
        self.model.parameters = new_parameters
        self._requires_update = True
        pass

    def set_stoich_matrix(self, new_stoich_matrix):
        self.model.stoich_matrix = new_stoich_matrix
        self._requires_update = True
        pass

    def set_propensity(self, new_propensity):
        self.model.propensity = new_propensity
        self._requires_update = True
        pass

    def plot_time_series(self, *args, **kwargs):
        self._update()
        self._get_axes()
        self.view.plot(self.data.time, self.data.state, *args, **kwargs)
        return

    def _get_xlim(self):
        x_min = min(self.data.time)
        x_max = max(self.data.time)
        return [x_min, x_max]

    def _get_ylim(self):
        y_min = min(self.data.state)
        y_max = max(self.data.state)
        return [y_min, y_max]

class VisualFSP(VisualAnalysis):

    def set_time(self, new_time):
        self.model.set_time(new_time)
        self._requires_update = True
        pass

    def set_initial_state(self, new_initial_state):
        self.model.set_initial_state(new_initial_state)
        self._requires_update = True
        pass

    def set_inf_generator(self, new_inf_generator):
        self.model.set_inf_generator(new_inf_generator)
        self._requires_update = True
        pass

    def set_propensity(self, new_inf_generator):
        self.model.set_inf_generator(new_inf_generator)
        self._requires_update = True
        pass

    def set_inf_generator(self, new_inf_generator):
        self.model.set_inf_generator(new_inf_generator)
        self._requires_update = True
        pass

    def set_inf_generator(self, new_inf_generator):
        self.model.set_inf_generator(new_inf_generator)
        self._requires_update = True
        pass

    def _get_states(self,probability):
        return np.arange(0,len(probability),1)

    def plot_marginal_distribution(self,time_index,species_index):
        self._update()
        self._get_axes()
        marginal_data=self.data.get_marginal_distribution(time_index, species_index)
        states=self._get_states(marginal_data)
        self.view.bar(states,marginal_data)
        pass

    def plot_marginal_distribution_over_time(self,species_index,steps=10):
        self._update()
        self._get_axes()
        for time_index in range(0,len(self.data.time),steps):
            self.plot_marginal_distribution(time_index, species_index)
        pass

class Frame:
    visual = []
    frame = None

    def __init__(self):
        pass

    def get_frame(self):
        if self.frame is None:
            self.frame = self.new_frame()
        return self.frame

    def add_view(self, new_view):
        self.visual.append(new_view)
        pass

    def new_frame(self):
        pass

    def set_style(self):
        self.frame.set_dpi(600)
        pass

class GlobalSingleton:
    __instance = None

    @staticmethod
    def get_instance():
        if GlobalSingleton.__instance == None:
            GlobalSingleton()
        return GlobalSingleton.__instance

    def __init__(self):
        if GlobalSingleton.__instance != None:
            raise Exception("This class is a singleton! Use GlobalSingleton.get_instance()")
        else:
            GlobalSingleton.__instance = self

class ModelFactory:
    visual = False

    def _transform_ssa(self, ssa_model):
        if self.visual:
            return VisualSSA(ssa_model)
        else:
            return ssa_model

    def _transform_ode(self, ode_model):
        if self.visual:
            return VisualODE(ode_model)
        else:
            return ode_model

    def _transform_fsp(self, fsp_model):
        if self.visual:
            return VisualODE(fsp_model)
        else:
            return fsp_model

    def birth_decay_ssa(self):
        ssa = SolverSSA()
        ssa.initial_state = np.array([0, 0])
        ssa.time = np.linspace(0, 100, 500)
        ssa.stoich_matrix = np.array([[-1, 1], [1, 0], [0 - 1]]).T
        ssa.parameters = np.array([1, 10, .1])
        ssa.propensity = lambda t, x, p: [p[0] * x[0], p[1], p[2]]
        ssa.num_samples = 50
        return self._transform_ssa(ssa)

    def represselator_ssa(self):
        pass

    def toggle_switch_ssa(self):
        pass

    def birth_decay_ode(self):
        ode = SolverODE()
        ode.initial_state = np.array([0, 0])
        ode.stoich_matrix = np.array([[-1, 1], [0, -1], [1, 0]]).T
        ode.propensity = lambda t, x, p: [p[0] * x[0], p[1] * x[1], p[2]]
        ode.parameters = np.array([1, .1, 10])
        ode.time = np.linspace(0, 100)
        return

    def represselator_ode(self):
        pass

    def toggle_switch_ode(self):
        pass

    def birth_decay_fsp(self):
        pass

    def represselator_fsp(self):
        pass

    def toggle_switch_fsp(self):
        pass

class GenericModel(object):
    '''
    I do all of the things, bow before me
    '''
    time = None
    parameters = None
    stoichiometry= None
    propensities = None
    num_samples = 1
    initial_state = None

    def __init__(self):
        self.model_parser = GenericModelFactory()


    def init_model(self, stoich, rxns):
        '''
        make the model!
        '''
        self.model_parser.init_model(stoich, rxns)
        self.vector_propensity = self.model_parser.vector_propensity
        #self.__get_parser_attributes()


    #def set_par



    def __fill_init_state_ssa(self):
        length = self.stoichiometry.shape[1]
        initial_state = np.zeros((1,length))
        initial_state[0] = 1
        self.initial_state = initial_state.flatten()
        return initial_state

    def __fill_fsp_dim(self):
        length = self.stoichiometry.shape[0]
        dim = np.ones((1,length))*30

        return dim.astype(int).flatten()


    def set_par_name(self,parameter_name,value):
        if parameter_name not in self.parameter_names:
            return
        i = 0
        for name in self.parameter_names:
            if name == parameter_name:
                self.parameter_dict[name] = value
                self.parameters[i] = value
                self.parameter_arr[i] = value

            i+=1


    def set_par_vec(self,parameter_vector):
        self.parameter_arr = np.array(parameter_vector).flatten()
        i = 0
        for name in self.parameter_names:
            self.parameter_dict[name] = parameter_vector[i]
            i+=1
        self.parameters = np.array(parameter_vector).flatten()



    def get_ODE(self):
        ode=SolverODE()

        # if self.initial_state == None:
        # if self.initial_state == None:
        #     ode.initial_state = self.__fill_init_state_ssa()
        # else:
        #     ode.initial_state = self.initial_state

        ode.initial_state = self.initial_state

        ode.stoich_matrix = self.stoichiometry

        ode.propensities = self.vector_propensity
        ode.propensity = self.vector_propensity

        ode.parameters = self.parameters
        if not isinstance(self.time, np.ndarray):
            ode.time = np.linspace(0,100,101)
        else:
            ode.time = self.time

        return ode


    def get_SSA(self):
        ssa = SolverSSA()

        # if self.initial_state == None:
        #     ssa.initial_state = self.__fill_init_state_ssa()
        # else:
        #     ssa.initial_state = self.initial_state

        ssa.stoich_matrix = self.stoichiometry.T

        ssa.propensity = self.vector_propensity


        ssa.num_samples = self.num_samples
        ssa.parameters = self.parameters
        if not isinstance(self.time, np.ndarray):
            ssa.time = np.linspace(0,100,101)
        else:
            ssa.time = self.time

        return ssa

    def get_FSP(self):
        fsp = SolverFSP()
        # if self.initial_state == None:
        #     fsp.initial_state = self.__fill_init_state_ssa()
        # else:
        #     fsp.initial_state = self.initial_state
        fsp.stoich_matrix = self.stoichiometry
        fsp.propensity = self.vector_propensity
        #if self.fsp_dim.any():
        #if 'self.fsp_dim' in locals():
        if self.fsp_dim.any():
            fsp.set_dimensions(self.fsp_dim)
        else:
            dim = self.__fill_fsp_dim()
            fsp.set_dimensions(dim)
        initial_state=np.zeros(fsp.dimensions)
        if isinstance(self.initial_state, np.ndarray):
            initial_state[self.initial_state] = 1
        elif isinstance(self.initial_state, list):
            initial_state[tuple(self.initial_state)] = 1
        else:
            initial_state[np.zeros((1,len(dim))).astype(int).flatten()] = 1
        fsp.set_initial_state(initial_state)
        if not isinstance(self.time, np.ndarray):
            fsp.set_time(np.linspace(0,100,101))
        else:
            fsp.set_time(self.time)
        fsp.set_stoich_matrix(self.stoichiometry)
        fsp.set_parameters(self.parameters)
        fsp.set_propensity(self.vector_propensity)
        fsp.error = 0

        return fsp




    def update_propensities(self):
        '''
        updates the model propensities if parameter values have been changed
        '''
        for reaction in self.propensities_str:
            lambdafun = self.parse_string_to_lambda(reaction)
            self.propensities.append(lambdafun)

    def __create_func(self, fun_string):
        """Attempts to create and return a timevarying function Lambda t: fun_string(t)
        """
        ##small parsing to prevent code execution inside the future execute call

        ## TODO: should switch to ast exec at some point
        problem_strs = [';', 'ast', 'exec', 'import', 'eval', 'run', 'os.', '.py', '__']
        if [char for char in problem_strs if char in fun_string] != []:
            return

        for num, var in enumerate(self.parameter_names):
            if var in fun_string:
                fun_string = fun_string.replace(var, ('p['+str(num)+']'))
        exec("TimeVar = lambda t,x,p: x*0 + "+ fun_string)
        return TimeVar



    def __create_func_from_defined_fun(self, fun_string):
        """Attempts to create and return a timevarying function Lambda t: fun_string(t)
        """
        ##small parsing to prevent code execution inside the future execute call

        ## TODO: should switch to ast exec at some point

        problem_strs = [';', 'ast', 'exec', 'import', 'eval', 'run', 'os.', '.py', '__']
        if [char for char in problem_strs if char in fun_string] != []:
            return

        for num, var in enumerate(self.parameter_names):
            if var in fun_string:
                fun_string = fun_string.replace(var, ('p['+str(num)+']'))
        exec("TimeVar = lambda t,x,p: x*0 + "+ fun_string)
        return TimeVar

class GenericParser(object):
    '''
    General parsing functions
    '''

    def __init__(self):
        pass

    def string_to_cmap(self,string):
        '''
        checks to see if a given string is a valid colormap
        returns valid colormap or nothing.
        '''
        maps = plt.colormaps()
        maps_lower = [x.lower() for x in plt.colormaps()] #get lowercase strings
        if string.lower() not in maps:
            raise ValueError\
            ('This is not a valid colormap"\
             "please use a matplotlib colormap',)


        sanitized_string = maps[maps_lower.index(string.lower())]
        return eval('plt.cm.'+sanitized_string)


    @classmethod
    def  check_matrix_is_int(cls, matrix):
        def isinteger(nparray):
            '''
            checks if integer
            '''
            return np.equal(np.mod(nparray, 1), 0)
        isint = isinteger(matrix).all()
        if isint:
            return matrix
        else:
            raise SyntaxError\
            ('Given matrix should be integers only',)

    @classmethod
    def check_matrix_dim(cls, stoich,dim = 2):
        '''
        given an object check if its a stoichiometry matrix
        '''

        if isinstance(stoich, list):
            try:
                int(stoich[0][0])
                stoich = np.array(stoich)
            except:
                raise SyntaxError\
                ('Given matrix should be a np array of dimension 2 or "\
                 "a list of lists of  dimension 2.',)


        if isinstance(stoich, np.ndarray):
            if len(stoich.shape) != dim:
                raise SyntaxError\
                ('Given matrix should be a np array of dimension 2 or" \
                 "a list of lists of  dimension 2.',)

            stoich = np.array(stoich)
            return stoich

        raise SyntaxError\
        ('Given matrix should be a np array of dimension 2 or" \
         a list of lists of  dimension 2.',)


    @classmethod
    def is_string_a_number(cls, string):
        '''
        Is this string a number?
        '''
        try:
            complex(string)
        except ValueError:
            return False
        return True

    @classmethod
    def is_obj_np_array(cls, obj):
        '''
        is this a numpy array?
        '''
        if type(obj) is np.ndarray:
            return True
        else:
            return False

class GenericModelFactory(GenericModel):
    '''
    this takes 1.

    stoichomitry matrix

    lambda functions OR strings

    returns

    propensities

    parameter object

    stoich



    '''

    def __init__(self):

        GenericModel.linear = True
        GenericModel.built = False
        GenericModel.parameter_dict = {}
        GenericModel.parameter_arr = np.array([])
        GenericModel.parameter_names = []
        GenericModel.parameter_type = {}

        GenericModel.FIM_analytical = True
        GenericModel.reaction_types = []
        self.parser = GenericParser()

    def safer_exec_lambda(self,expression_string):
        '''
        Safer execution of lambda functions from string
        '''
        tree_obj = ast.parse(('self.' + expression_string))
        exec(compile(tree_obj, filename="<ast>", mode="exec"))

        return self.TimeVar


    def safer_exec_selfvar(self,expression_string):
        '''
        Safer execution of self variables from string
        '''
        tree_obj = ast.parse(expression_string)

        exec(compile(tree_obj, filename="<ast>", mode="exec"))



    def detect_function_or_string(self,reaction):

        if callable(reaction):

            if self.__check_if_defined_function(reaction):
                parsed_reaction = self.__parse_def_function(reaction)
                fun_name = self.__get_defined_function_name(parsed_reaction)
                inputs = self.__get_defined_function_inputs(parsed_reaction)
                topindex = self.__get_defined_function_parameter_max_index(parsed_reaction)

                GenericModel.propensities_str.append((fun_name + inputs))
                if len(inputs.split(',')) > len(['x','t']): #if splits > base inputs, append additional things
                    detected_par = (fun_name + '_par')

                    self.__add_par_function(detected_par, topindex)

                self.safer_exec_selfvar('self.' + fun_name + '=' + fun_name)


                GenericModel.FIM_analytical = False

                GenericModel.function_names.append(fun_name)
                reaction_type.append('defined')

            else:
                    function_string = self.__parse_given_lambda(reaction, i)


                    GenericModel.propensities_str.append(function_string)

                    splitstring, item_indexes = self.__get_function_string_elements(function_string)

                    detected_par = self.__detect_par(splitstring, item_indexes)

                    self.__add_par(detected_par)

                    reaction_type.append('regular')



        else:
            function_string = self.__parse_function_string(reaction)
            GenericModel.propensities_str.append(function_string)
            splitstring, item_indexes = self.__get_function_string_elements(function_string)

            detected_par = self.__detect_par(splitstring, item_indexes)
            self.__add_par(detected_par)
            reaction_type.append('regular')



    def init_model(self, stoich, rxns):
        '''
        stoichiometry should be either 1. list of lists, or a np array

        '''
        GenericModel.propensities = []
        GenericModel.propensities_str = []
        stoich = self.parser.check_matrix_dim(stoich)
        stoich = self.parser.check_matrix_is_int(stoich)
        GenericModel.stoichiometry = stoich

        reaction_type = []
        GenericModel.function_names = []
        for i, reaction in enumerate(rxns):

            if callable(reaction):

                if self.__check_if_defined_function(reaction):
                    parsed_reaction = self.__parse_def_function(reaction)
                    fun_name = self.__get_defined_function_name(parsed_reaction)
                    inputs = self.__get_defined_function_inputs(parsed_reaction)
                    topindex = self.__get_defined_function_parameter_max_index(parsed_reaction)

                    GenericModel.propensities_str.append((fun_name + inputs))
                    if len(inputs.split(',')) > 2:
                        detected_par = (fun_name + '_par')

                        self.__add_par_function(detected_par, topindex)

                    self.safer_exec_selfvar('GenericModel.' + fun_name + '=' + fun_name)

                    GenericModel.FIM_analytical = False

                    GenericModel.function_names.append(fun_name)
                    reaction_type.append('defined')




                else:
                    function_string = self.__parse_given_lambda(reaction, i)


                    GenericModel.propensities_str.append(function_string)

                    splitstring, item_indexes = self.__get_function_string_elements(function_string)

                    detected_par = self.__detect_par(splitstring, item_indexes)


                    self.__add_par(detected_par)

                    reaction_type.append('regular')



            else:
                function_string = self.__parse_function_string(reaction)
                GenericModel.propensities_str.append(function_string)
                splitstring, item_indexes = self.__get_function_string_elements(function_string)

                detected_par = self.__detect_par(splitstring, item_indexes)
                self.__add_par(detected_par)
                reaction_type.append('regular')

        for i, reaction in enumerate(self.propensities_str):

            if reaction_type[i] == 'regular':
                lambdafun = self.parse_string_to_lambda(reaction)

                splitstring, item_indexes = self.__get_function_string_elements(reaction)

                tv = self.__is_timevarying(splitstring)
                nl = self.__is_nonlinear(splitstring)
                idstr = ''
                if tv:
                    idstr = idstr+ 't,'
                if nl:
                    idstr = idstr + 'x,'
                idstr = idstr + 'p'
                GenericModel.reaction_types.append(idstr)


            else:
                lambdafun = self.parse_def_to_lambda(reaction)
                GenericModel.reaction_types.append('t,x,p')
            GenericModel.propensities.append(lambdafun)







        GenericModel.built = False
        if isinstance(GenericModel.stoichiometry, np.ndarray):
            GenericModel.built = True
        k = 0
        for reaction in GenericModel.propensities:

            k += 1
            try:

                xout = reaction(1, np.ones(100), GenericModel.parameter_arr)

                if not self.parser.is_string_a_number(str(xout)):

                    GenericModel.built = False
                    print('reaction %i is invalid, double check reaction %i rate equation'%(k, k))
                else:
                    GenericModel.built = True


            except:
                print('reaction %i is invalid, double check reaction %i rate equation'%(k, k))
                GenericModel.built = False
#
#            try:
#                xout = reaction(0, np.ones(100), self.parameter_arr)
#
#                if not self.__is_np_array(xout):
#                    self.built = False
#                    print('reaction %i is invalid, double check reaction %i rate equation'%(k, k))
#                else:
#                    self.built = True
#            except:
#                print('reaction %i is invalid, double check reaction %i rate equation'%(k, k))
#                self.built = False

        combined_propensity = self.make_combined_lambda()
        self.vector_propensity = combined_propensity

        # if GenericModel.built:
        #     print('Successfully built model')

    def make_combined_lambda(self):
        newstr = '['
        for i,string in enumerate(GenericModel.propensities_str):
            for k,varname in enumerate(GenericModel.parameter_names):

                if varname in string:
                    string = string.replace(varname,('p['+str(k)+']') )

            newstr = newstr + string + ','
        fun_string = newstr[:-1] + ']'
        comb_lambda = self.safer_exec_lambda(("TimeVar = lambda t,x,p: "+ fun_string))
        #exec(("comb_lambda = lambda t,x,p: "+ fun_string))
        return comb_lambda

    def parse_string_to_lambda(self, fun_string):
        """Attempts to create and return a timevarying function Lambda t: fun_string(t)
        """
        ##small parsing to prevent code execution inside the future execute call
        ## TODO: should switch to ast exec at some point
        problem_strs = [';', 'ast', 'exec', 'import', 'eval', 'run', 'os.', '.py', '__']
        if [char for char in problem_strs if char in fun_string] != []:
            return
        for num, var in enumerate(GenericModel.parameter_names):
            if var in fun_string:
                fun_string = fun_string.replace(var, ('p['+str(num)+']'))
        TimeVar = self.safer_exec_lambda("TimeVar = lambda t,x,p: "+ fun_string)
        return TimeVar



    def parse_def_to_lambda(self, fun_string):
         return self.parse_string_to_lambda(fun_string)

#        """Attempts to create and return a timevarying function Lambda t: fun_string(t)
#        """
#        ##small parsing to prevent code execution inside the future execute call
#
#        ## TODO: should switch to ast exec at some point
#
#        problem_strs = [';', 'ast', 'exec', 'import', 'eval', 'run', 'os.', '.py', '__']
#        if [char for char in problem_strs if char in fun_string] != []:
#            return
#
#        for num, var in enumerate(self.parameter_names):
#            if var in fun_string:
#                fun_string = fun_string.replace(var, ('p['+str(num)+']'))
#
#        exec("TimeVar = lambda t,x,p: x*0 + "+ fun_string)
#
#        return TimeVar

    def __check_if_defined_function(self, func):
        '''
        checks if a given reaction is a defined function and not a lambda
        '''
        check_def = inspect.getsourcelines(func)[0]

        if len(check_def) > 1:
            if 'def ' in check_def[0]:
                defined_function = self.__parse_def_function(func)

                return True
            else:
                return False
        else:
            return False

    def __parse_given_lambda(self,function_obj, rxn_num):
        '''
        if the user gives a lambda function, parse it back out to
        make sure its valid ie only x's and t's and then detect parameters

        This then gets passed to safe exec
        '''

        function_string = self.__parse_function_list_or_lambda_to_string(function_obj,rxn_num)

        inputs,lambda_string = self.__split_function_string_to_inputs_and_func(function_string)
        ordered_inputs = self.__order_inputs(inputs,lambda_string)

        splitstring, item_indexes =self.__get_function_string_elements(lambda_string)

        operators,operator_indexes = self.__get_operators_and_operator_indexes(lambda_string)

        rebuild_order = self.__get_rebuild_order(item_indexes,operator_indexes)
        splitstring = self.__rename_inputs_to_xt(splitstring,ordered_inputs)
        finalstring = self.__rebuild_final_function_string(operators,rebuild_order,splitstring)

        return finalstring

    def __get_rebuild_order(self,items,ops):
        '''
        get the rebuild order of the split items (numbers, variables, functions)
        and operators (*, /, +) that are then used to rebuild the final parsed lambda
        string
        '''

        rebuild_order = []
        for i in range(len(items)+len(ops)):
            dothing = 0
            if ops != []:
                if items[0][0] < ops[0][0]:
                    rebuild_order.append('item')
                    items.pop(0)

                    dothing = 1
                if dothing == 0:
                    if items[0][0] >= ops[0][0]:
                        rebuild_order.append('op')
                        ops.pop(0)
                        dothing = 1
            else:

                rebuild_order.append('item')
                items.pop(0)
        return rebuild_order

    def __split_function_string_to_inputs_and_func(self,function_string):
        '''
        given a function_string, split up the inputs and the actual function as a
        string ie:

            'lambda x,t: x[0]*2'

        would become

            inputs = 'x,t'
            func = 'x[0]*2'
        '''
        inputs = function_string[:function_string.index(':')]
        func = function_string[function_string.index(':')+1:]
        func = func.replace(' ', '')
        inputs = inputs.replace('lambda', '')
        inputs = inputs.replace(' ', '')
        inputs = inputs.split(',')

        return inputs, func

    def __rename_inputs_to_xt(self,splitstring,ordered_inputs):
        '''
        rename the inputs to x,t
        '''

        for i, item in enumerate(splitstring):

            if item in ordered_inputs:

                splitstring[i] = ['x', 't'][ordered_inputs.index(item)]
        return splitstring

    def __order_inputs(self,inputs,func):
        '''
        reorder the inputs of the lambda function string
        '''
        if set(inputs) != set(['t', 'x']):
            xequiv = inputs[0]
            if '[' in func:
                opstring = '\*|\+|\(|/|-|\)|\^|\,|==|!=|\<|\>|<=|>='
                splitstring = re.split(opstring, func)
                for substring in splitstring:
                    if '[' in substring:
                        xequiv = substring[:substring.index('[')]

            ordered_inputs = []
            for i, inp in enumerate(inputs):
                if inp == xequiv:
                    ordered_inputs.append(inp)
            for i, inp in enumerate(inputs):
                if inp != xequiv:
                    ordered_inputs.append(inp)

        else:
            ordered_inputs = ['x', 't']
        return ordered_inputs

    def __rebuild_final_function_string(self,operators,rebuild_order,splitstring):
        o_in = 0
        i_in = 0
        finalstring = ''
        for op_ind, op_str  in enumerate(operators):
            if op_str == '^':
                operators[op_ind] = '**'


        for tup_ind, tup in enumerate(rebuild_order):

            if 'item' in tup:
                finalstring = finalstring + splitstring[i_in]
                i_in += 1
            if 'op' in tup:
                finalstring = finalstring + operators[o_in]
                o_in += 1

        return finalstring

    def __parse_function_list_or_lambda_to_string(self,function_obj,rxn_num):
        '''
        this parses both lists and lambdas into a single string for parsing
        '''
        function_string = str(inspect.getsourcelines(function_obj)[0])

        try:
            if isinstance(function_obj,list):


                liststart = [(m.start(), m.end()) for m in re.finditer('\[', function_string)][1]
                listend = [(m.start(), m.end()) for m in re.finditer('\]', function_string)][-2]
                rxn_str_list = function_string[liststart[0]:listend[1]].split(',')
                rebuild_list = []
                for i, item in enumerate(rxn_str_list):
                    if 'lambda' not in item:
                        if ':' not in item:
                            rebuild_list.append(item)
                    if 'lambda' in item:
                        rebuild_list.append((item.replace('[', '') + ','+ rxn_str_list[i+1]))

                    if i == len(rxn_str_list):
                        rebuild_list[-1] = rebuild_list[-1][:-1]

                function_string = rebuild_list[rxn_num]
            else:
                1/0

        except:
            function_string = str(inspect.getsourcelines(function_obj)[0])

            #function_string = function_string[:].strip("['\\n]'").split(" = ")[1]
            start_function_string = 2
            end_function_string = 4
            function_string = function_string[start_function_string:-end_function_string]
            function_string = function_string.split(' = ')[1]

        return function_string

    def __get_operators_and_operator_indexes(self,func):
        opstring = '\*|\+|\(|/|-|\)|\^|\,|==|!=|\<|\>|<=|>='   #\[|\]'
        operators = re.findall(opstring, func)
        operator_indexes = [(m.start(0), m.end(0)) for m in re.finditer(opstring, func)]
        return operators,operator_indexes

    def __parse_def_function(self, defined_function):
        source = inspect.getargspec(defined_function)
        sourcelines = inspect.getsourcelines(defined_function)

        #get the function name from any def function
        functionname = sourcelines[0][0].split('def ')[1].split('(')[0]

        return_string = sourcelines[0][-1]
        return_string = "".join(return_string.split())

        if len(return_string.split(',')) > 2:
            try:
                raise SyntaxError
            except SyntaxError:
                print ('User defined function ' + functionname + ' returns too many objects,'\
                     +' only return one object')
                return
        if len(return_string.split(',')) > 1:
            if return_string.split(',')[1] != '':

                try:
                    raise SyntaxError
                except SyntaxError:
                    print ('User defined function ' + functionname + ' returns too many objects,'\
                           + ' only return one object')
                    return

            else:
                pass

        args = source.args
        if source.defaults:
            num_required_args = len(args) - len(source.defaults)
        else:
            num_required_args = len(args)

        if num_required_args > 3:
            try:
                raise SyntaxError
            except SyntaxError:
                print ('User defined function ' + functionname + ' takes too many defined arguments, '\
                       +'its recommended the function only takes t, x, parameters')
                return

        if set(['x', 't']) == set(args):

            try:
                fout = defined_function(0, 1)
            except:
                try:
                    raise SyntaxError
                except SyntaxError:
                    print ('User defined function ' + functionname + ' doesnt accept'\
                           +' inputs t = 0, x = 1')
                    return
            if self.parser.is_string_a_number(str(fout)):
                return defined_function
            else:
                try:
                    raise SyntaxError
                except SyntaxError:
                    print ('User defined function ' + functionname + ' doesnt return '\
                           +'a number or value')
                    return
        else:
            if set(['x', 't']) == set(args[0:2]):
                par = args[2]
                topindex = None
                for line in inspect.getsourcelines()[1:-1]:
                    hits = [(m.start(), m.end()) for m in re.finditer(par, line)]
                    for hit in hits:
                        if hits[1] == '[':
                             index = line[hits[1]: hits[1] + line[hit[1]:].find(']')]
                             if index > topindex:
                                 topindex = index
                if not topindex:
                    testpar = 1
                else:
                    testpar = np.ones(topindex)

                fout = defined_function(0, 1, testpar)

                if self.parser.is_string_a_number(str(fout)):
                    return defined_function
                else:
                    try:
                        raise SyntaxError
                    except SyntaxError:
                        print ('User defined function ' + functionname + ' doesnt return '\
                               +'a number or value')
                        return


            else:
                if set(['x', 't']).issubset(args):
                    par = list(set(args) - set(['x', 't']))[0]

                    topindex = None
                    for line in inspect.getsourcelines()[1:-1]:
                        hits = [(m.start(), m.end()) for m in re.finditer(par, line)]
                        for hit in hits:
                            if hits[1] == '[':
                                 index = line[hits[1]: hits[1] + line[hit[1]:].find(']')]
                                 if index > topindex:
                                     topindex = index
                    if not topindex:
                        testpar = 1
                    else:
                        testpar = np.ones(topindex)
                    newargs = [1, 1, 1]
                    newargs[args.index(par)] = testpar

                    fout = defined_function(*newargs)

                    if self.parser.is_string_a_number(str(fout)):
                        return defined_function
                    else:
                        try:
                            raise SyntaxError
                        except SyntaxError:
                            print ('User defined function ' + functionname + ' doesnt return a number or value')
                            return


                else:

                    par = args[2]
                    topindex = None
                    for line in inspect.getsourcelines()[1:-1]:
                        hits = [(m.start(), m.end()) for m in re.finditer(par, line)]
                        for hit in hits:
                            if hits[1] == '[':
                                 index = line[hits[1]: hits[1] + line[hit[1]:].find(']')]
                                 if index > topindex:
                                     topindex = index
                    if not topindex:
                        testpar = 1
                    else:
                        testpar = np.ones(topindex)

                    fout = defined_function(0, 1, testpar)

                    if self.parser.is_string_a_number(str(fout)):
                        return defined_function
                    else:
                        try:
                            raise SyntaxError
                        except SyntaxError:
                            print ('User defined function ' + functionname + ': could not '\
                                   +'determine which input was which, please use t,x,par')
                            return

    def __get_defined_function_name(self, function):
        '''
        gets the function named from a defined function
        '''
        sourcelines = inspect.getsourcelines(function)
        function_name = sourcelines[0][0].split('def ')[1].split('(')[0]
        return function_name

    def __get_defined_function_inputs(self, defined_function):
        '''
        returns the input string in the correct order for a defined function wrapper
        '''
        source = inspect.getargspec(defined_function)
        args = source.args
        if set(['x', 't']) == set(args):
            if args.index('x') == 0:
                return '(x,t)'
            else:
                return '(t,x)'


        else:
            if set(['x', 't']) == set(args[0:2]):
                return '(x,t,p)'

            else:
                if set(['x', 't']).issubset(args):
                    par = list(set(args) - set(['x', 't']))[0]

                    topindex = None
                    for line in inspect.getsourcelines()[1:-1]:
                        hits = [(m.start(), m.end()) for m in re.finditer(par, line)]
                        for hit in hits:
                            if hits[1] == '[':
                                 index = line[hits[1]: hits[1] + line[hit[1]:].find(']')]
                                 if index > topindex:
                                     topindex = index

                    xarg = args.index('x')
                    targ = args.index('t')
                    parg = args.index(par)
                    builder = ['', '', '']
                    builder[xarg] = 'x'
                    builder[targ] = 't'
                    builder[parg] = 'p'
                    return ('(' + builder[0] + ',' + builder[1] + ',' + builder[2] + ')')



                else:

                    par = args[2]
                    topindex = None
                    for line in inspect.getsourcelines()[1:-1]:
                        hits = [(m.start(), m.end()) for m in re.finditer(par, line)]
                        for hit in hits:
                            if hits[1] == '[':
                                 index = line[hits[1]: hits[1] + line[hit[1]:].find(']')]
                                 if index > topindex:
                                     topindex = index
                    return '(t,x,p)'

    def __get_defined_function_parameter_max_index(self, defined_function):
        '''
        for the parameter input of a defined function get its largest input size
        '''
        source = inspect.getargspec(defined_function)
        args = source.args
        if set(['x', 't']) == set(args):
            return None

        else:
            if set(['x', 't']) == set(args[0:2]):
                par = args[2]
                topindex = None
                for line in inspect.getsourcelines()[1:-1]:
                    hits = [(m.start(), m.end()) for m in re.finditer(par, line)]
                    for hit in hits:
                        if hits[1] == '[':
                             index = line[hits[1]: hits[1] + line[hit[1]:].find(']')]
                             if index > topindex:
                                 topindex = index
                return topindex

            else:
                if set(['x', 't']).issubset(args):
                    par = list(set(args) - set(['x', 't']))[0]

                    topindex = None
                    for line in inspect.getsourcelines()[1:-1]:
                        hits = [(m.start(), m.end()) for m in re.finditer(par, line)]
                        for hit in hits:
                            if hits[1] == '[':
                                 index = line[hits[1]: hits[1] + line[hit[1]:].find(']')]
                                 if index > topindex:
                                     topindex = index

                    return top_index


                else:

                    par = args[2]
                    topindex = None
                    for line in inspect.getsourcelines()[1:-1]:
                        hits = [(m.start(), m.end()) for m in re.finditer(par, line)]
                        for hit in hits:
                            if hits[1] == '[':
                                 index = line[hits[1]: hits[1] + line[hit[1]:].find(']')]
                                 if index > topindex:
                                     topindex = index
                    return topindex

    def __parse_function_string(self, string):
        """ Takes an expression string and attempts to detect and find
        all parameters inside the expression

        detected_var = all parameters detected
        detected_nums = all numbers detected
        detected_functions = numpy expressions found
        finalstring = rebuilt string to include all the numpy expressions
        """
        string = string.replace(' ', '')

        def is_number(string):
            '''
            check if string is a number
            '''
            try:
                complex(string)
            except ValueError:
                return False
            return True

        finalstring = ''

        splitstring, item_indexes = self.__get_function_string_elements(string)
        rebuild_items = self.__get_rebuild_items(splitstring, item_indexes)


        full_tuple = []

        opstring = '\*|\+|\(|/|-|\)|\^|\,|==|!=|\<|\>|<=|>=|\[|\]'
        operators = re.findall(opstring, string)
        indexes = [(m.start(0), m.end(0)) for m in re.finditer(opstring, string)]

        for tup_ind, ind in enumerate(indexes):
            full_tuple.append(ind + ('op',))

        for tup_ind, ind in enumerate(item_indexes):
            full_tuple.append(ind + ('item',))

        full_tuple.sort(key=lambda tup: tup[0])
        o_in = 0
        i_in = 0
        finalstring = ''
        for op_ind, op_str  in enumerate(operators):
            if op_str == '^':
                operators[op_ind] = '**'


        for tup_ind, tup in enumerate(full_tuple):

            if 'item' in tup:
                finalstring = finalstring + rebuild_items[i_in]
                i_in += 1
            if 'op' in tup:
                finalstring = finalstring + operators[o_in]
                o_in += 1


        indexes = [(m.start(0), m.end(0)) for m in re.finditer('\[\]', finalstring)]

        for index in indexes:
            finalstring = finalstring[:index[0]] + finalstring[index[1]:]
        return finalstring

    def __get_function_string_elements(self, string):
        '''
        given a function string, return the elements and indices of them
        '''
        string = string.replace(' ', '')

        def is_number(string):
            '''
            check if string is a number
            '''
            try:
                complex(string)
            except ValueError:
                return False
            return True

        opstring = '\*|\+|\(|/|-|\)|\^|\,|==|!=|\<|\>|<=|>=|\[|\]'
        splitstring = re.split(opstring, string)  #regex to split by operators
        operators = re.findall(opstring, string) #|\[|\]
        indexes = [(m.start(0), m.end(0)) for m in re.finditer(opstring, string)]


        for i in range(len(splitstring)-1):
            if splitstring[i] == 'x':
                if operators[i] == '[':

                    newvar = splitstring[i]+operators[i]+splitstring[i+1] + operators[i+1]

                    operators.pop(i)
                    operators.pop(i)

                    splitstring.pop(i)
                    splitstring.pop(i)
                    indexes.pop(i)
                    indexes.pop(i)

                    splitstring.insert(i, newvar)

        item_indexes = []
        foundnum = []
        if indexes == []:

            item_indexes = [(0, len(string))]

        else:
            if len(indexes) == 1:
                if indexes[0][0] != 0:
                    item_indexes.append((0, indexes[0][0]))
                if indexes[0][1] != len(string):
                    item_indexes.append((indexes[0][1], len(string)))
            else:

                for nind in range(1, len(indexes)):

                    if indexes[nind-1][0] not in foundnum:
                        foundnum.append(indexes[nind-1][0])
                    if indexes[nind-1][1] not in foundnum:
                        foundnum.append(indexes[nind-1][1])

                    if nind == 1:
                        if indexes[nind-1][0] != 0:
                            item_indexes.append((0, indexes[nind-1][0]))

                    if indexes[nind-1][1]-indexes[nind][0] != 0:
                        item_indexes.append((indexes[nind-1][1], indexes[nind][0]))

                    if nind+1 == len(indexes):
                        if indexes[nind][1] < len(string):
                            item_indexes.append((indexes[nind][1], len(string)))



        splitstring[:] = [x for x in splitstring if x != '']

        return splitstring, item_indexes

    def __get_rebuild_items(self, splitstring, item_indexes):
        '''
        gathers and returns all items required to build the final string in
        parse vairables
        '''
        rebuild_items = []

        for item in splitstring:

            try:

                if self.parser.is_string_a_number(item) is False and item not in ['e', 't']:
                    if 'np.' not in item:
                        if item == 'E':
                            eval('np.' + item + '(1)')
                        else:

                            eval('np.' + item.lower() + '(1)')

                        rebuild_items.append('np.'+item.lower())
                    else:
                        if item == 'E':
                            eval(item + '(1)')
                        else:
                            eval(item.lower() + '(1)')
                        rebuild_items.append(item.lower())

                else:
                    if item in ['e']:
                        if string[item_indexes[splitstring.index(item)][1]] == '^':
                            rebuild_items.append('np.e')
                    else:
                        rebuild_items.append(item)

            except (AttributeError, ValueError, NameError, TypeError, SyntaxError, IOError) as err:

                if (sys.version_info > (3, 0)):
                    if len(err.args) == 2:
                        var = err.args[1][3].replace('(1)', '')
                        if var == item:
                            rebuild_items.append(var)
                        else:
                            rebuild_items.append(item)
                    else:

                        var = err.args[0].replace('(1)', '')
                        if var == item:
                            rebuild_items.append(var)
                        else:
                            rebuild_items.append(item)

                else:
                    '''
                    if len(err.message.split("'")) == 2:
                        var = err.args[1][3].replace('(1)', '')
                        if var == item:
                            rebuild_items.append(var)
                        else:
                            rebuild_items.append(item)
                    else:
                    '''


                    var = err.message.split("'")[-2].replace("'", '')
                    if var == item:
                        rebuild_items.append(var)
                    else:
                        rebuild_items.append(item)
        return rebuild_items

    def __detect_nums(self, splitstring, item_indexes):
        '''
        detects true numbers from function string
        '''
        detected_nums = []
        for item in splitstring:   #for each section
            try:
                if self.parser.is_string_a_number(item) is False and item not in ['e', 't']:
                    pass
                else:
                    if item in ['e']:
                        if string[item_indexes[splitstring.index(item)][1]] == '^':
                            detected_nums.append('np.e')
                    else:
                        detected_nums.append(item)
            except (AttributeError, ValueError, NameError, TypeError, SyntaxError, IOError) as err:
                pass

        return detected_nums

    def __detect_np_functions(self, splitstring, item_indexes):
        '''
        detects numpy functions from the function strings
        '''
        detected_functions = []
        for item in splitstring:
            try:
                if self.parser.is_string_a_number(item) is False and item not in ['e', 't']:
                    if 'np.' not in item:
                        if item == 'E':
                            eval('np.' + item + '(1)')
                        else:
                            eval('np.' + item.lower() + '(1)')
                        detected_functions.append('np.' + item.lower())
                    else:
                        if item == 'E':
                            eval(item + '(1)')
                        else:
                            eval(item.lower() + '(1)')
                        detected_functions.append(item.lower())
                else:
                    pass

            except (AttributeError, ValueError, NameError, TypeError, SyntaxError, IOError) as err:
                pass
        return detected_functions

    def __is_timevarying(self, splitstring):
        '''
        Detects if function is time varying
        '''
        timevarying = False
        for item in splitstring:
            if not self.parser.is_string_a_number(item) and item in ['t']:
                    timevarying = True
        return timevarying

    def __is_nonlinear(self, splitstring):
        '''
        Detects if function is Nonlinear
        '''
        nonlinear = False
        for item in splitstring:
            if 'x' in item:
                nonlinear = True
        return nonlinear

    def __detect_par(self, splitstring, item_indexes):
        '''
        this detects parameters inside function string
        '''
        detected_var = []

        for item in splitstring:

            try:

                if not self.parser.is_string_a_number(item) and item not in ['e', 't']:
                    if 'np.' not in item:
                        if item == 'E':
                            eval('np.' + item + '(1)')
                        else:

                            eval('np.' + item.lower() + '(1)')

                        detected_functions.append('np.' + item.lower())
                        rebuild_items.append('np.'+item.lower())
                    else:
                        if item == 'E':
                            eval(item + '(1)')
                        else:

                            eval(item.lower() + '(1)')

                else:
                    if item in ['e']:
                        if string[item_indexes[splitstring.index(item)][1]] == '^':
                            tmp = 1

            except (AttributeError, ValueError, NameError, TypeError, SyntaxError, IOError) as err:

                if len(err.args) == 2:
                    var = err.args[1][3].replace('(1)', '')
                    if var == item:
                        detected_var.append(var)

                    else:
                        detected_var.append(item)
                else:

                    if (sys.version_info > (3, 0)):

                        var = err.args[0].split("'")[3].replace("'", '')
                        if var == item:
                            detected_var.append(var)
                        else:
                            detected_var.append(item)

                    else:
                        var = err.message.split("'")[3].replace("'", '')
                        if var == item:
                            detected_var.append(var)
                        else:
                            detected_var.append(item)
        try:
            detected_var.remove('x')
        except:
            pass
        for var in detected_var:
            if 'x[' in var:
                detected_var.remove(var)

        return detected_var


    def __add_par(self, var):

        '''
        checks the parameter name and adds it to the model parameter dictionary
        and list, and self parameters
        '''
        if isinstance(var, list):
            for item in var:
                problem_strs = [';', 'ast', 'exec', 'import', 'eval', 'run', 'os.', '.py', '__']
                if [char for char in problem_strs if char in item] != []:
                    print('invalid character given as parameter name')
                    return
                if item not in GenericModel.parameter_names:
                    GenericModel.parameter_arr = np.hstack((GenericModel.parameter_arr, [1]))
                    GenericModel.parameter_names.append(item)
                    GenericModel.parameter_dict[item] = 1
                    GenericModel.parameter_type[item] = {'constant'}
                    self.safer_exec_selfvar(('GenericModel.' + item + '=1'))
        else:
            if var not in GenericModel.parameter_names:
                problem_strs = [';', 'ast', 'exec', 'import', 'eval', 'run', 'os.', '.py', '__']
                if [char for char in problem_strs if char in item] != []:
                    print('invalid character given as parameter name')
                    return
                GenericModel.parameter_arr = np.hstack((GenericModel.parameter_arr, [1]))
                GenericModel.parameter_names.append(var)
                GenericModel.parameter_dict[var] = 1
                GenericModel.parameter_type[item] = {'constant'}
                self.safer_exec_selfvar(('GenericModel.' + var + '=1'))

    def __add_par_function(self, var, number_of_entries):
        '''
        checks the parameter name and adds it to the model parameter dictionary
        and list, and self parameters
        '''
        problem_strs = [';', 'ast', 'exec', 'import', 'eval', 'run', 'os.', '.py', '__']
        if not number_of_entries:
            if isinstance(var, list):
                for item in var:

                    if [char for char in problem_strs if char in item] != []:
                        print('invalid character given as parameter name')
                        return
                    if item not in GenericModel.parameter_names:
                        GenericModel.parameter_arr = np.hstack((GenericModel.parameter_arr, np.ones(number_of_entries).tolist()))
                        GenericModel.parameter_names.append(item)
                        GenericModel.parameter_dict[item] = np.ones(number_of_entries).tolist()
                        GenericModel.parameter_type[item] = {'constant'}
                        self.safer_exec_selfvar(('GenericModel.' + item + '=' +str(np.ones(number_of_entries).tolist())))



            if isinstance(var, np.ndarray):
                for item in var:
                    if [char for char in problem_strs if char in item] != []:
                        print('invalid character given as parameter name')
                        return
                    if item not in GenericModel.parameter_names:
                        GenericModel.parameter_arr = np.hstack((GenericModel.parameter_arr, np.ones(number_of_entries).tolist()))
                        GenericModel.parameter_names.append(item)
                        GenericModel.parameter_dict[item] = np.ones(number_of_entries).tolist()
                        GenericModel.parameter_type[item] = {'constant'}
                        self.safer_exec_selfvar(('GenericModel.' + item + '=' +str(np.ones(number_of_entries).tolist())))



        else:
            if var not in GenericModel.parameter_names:
                problem_strs = [';', 'ast', 'exec', 'import', 'eval', 'run', 'os.', '.py', '__']
                if [char for char in problem_strs if char in item] != []:
                    print('invalid character given as parameter name')
                    return
                GenericModel.parameter_arr = np.hstack((GenericModel.parameter_arr, [1]))
                GenericModel.parameter_names.append(var)
                GenericModel.parameter_dict[var] = 1
                GenericModel.parameter_type[item] = {'constant'}
                self.safer_exec_selfvar(('GenericModel.' + var + '=1'))
