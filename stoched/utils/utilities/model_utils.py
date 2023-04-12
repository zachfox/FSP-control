class Model:
    def __init__(self):
        return
        
    def interact(self):
        '''interact with the data.
        '''
        return
    
    def fit(self, data):
        '''Fit the model to the data!
        '''
        return 

    def solve(self):
        '''solve the model.
        '''
        return 
    
class ModelCollection:
    def __init__(self, model_collection=None):
        ''' a collection of data objects. 
        '''        
        if model_collection is not None:
            self.model_collection = model_collection 
            self.nmodels = len(model_collection)
        else:
            self.model_collection = []
        return 

    def append_model(self, model):
        '''append a new data set.
        '''
        if isinstance(model, Model):
            self.model_collection.append(model)   
            self.update_nmodels()
    def update_nmodels(self):
        self.nmodels = len(self.model_collection)
