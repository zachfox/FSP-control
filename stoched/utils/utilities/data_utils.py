import numpy as np


class Data:
    def __init__(self):
        self.data_matrix = []
        self.times = []
        self.experiment_meta = []
        self.means = []
        self.variances = []
        return
        
    def interact(self):
        '''interact with the data.
        '''
        return
    
class DataCollection:
    def __init__(self):
        ''' a collection of data objects. 
        '''        
        self.data_collection = [] 
        return 

    def append_data(self, data):
        '''append a new data set.
        '''
        if isinstance(data, Data):
            self.data_collection.append(data)   
            self.update_ndata()
    def update_ndata(self):
        self.ndata = len(self.data_collection)


            




