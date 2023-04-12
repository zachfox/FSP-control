import numpy as np


class Parameters:
    def __init__(self):
        self.fixed_parameters = []
        return
        
    def interact(self):
        '''interact with the data.
        '''
        return

    def get_vector(self):
        self.vec = []
        self.names = []
        for name,param in self.__dict__.items():
            if (isinstance(param,int) or isinstance(param,float)):
                self.vec.append(param)    
                self.names.append(name)
            else:
                pass
        self.vec = np.array(self.vec)
    
    def set_vector(self, vec=None):
        if vec is not None:
            self.vec = vec
        for i,value in enumerate(self.vec):
            self.__setattr__(self.names[i],value)

    def get_fit_vector(self):
        self.fit_vec = []
        self.fit_names = []
        for name,param in self.__dict__.items():
            if (isinstance(param,int) or isinstance(param,float)) and (name not in self.fixed_parameters):
                self.fit_vec.append(param)    
                self.fit_names.append(name)
            else:
                pass
        self.fit_vec = np.array(self.fit_vec)
    
    def set_fit_vector(self, fit_vec=None):
        if fit_vec is not None:
            self.fit_vec = fit_vec
        for i,value in enumerate(self.fit_vec):
            self.__setattr__(self.fit_names[i],value)
        
class ParameterCollection:
    def __init__(self):
        ''' a collection of data objects. 
        '''        
        self.parameter_collection = [] 
        return 

    def append_parameters(self, data):
        '''append a new data set.
        '''
        if isinstance(data, Data):
            self.parameter_collection.append(data)   
            self.update_ndata()
    def update_ndata(self):
        self.ndata = len(self.parameter_collection)