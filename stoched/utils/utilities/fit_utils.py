import numpy as np
from stoched.utils.utilities import data_utils


def least_squares(data, model):
    pass

def select_fit_predict(dc, split=.75):
    '''
    takes a collection of data and splits into train/test
    for cross-validation. Split is the fraction of the 
    data that should be used for training. 
    '''
    ntrain = int(np.floor(dc.ndata*split))
    rand_inds = np.arange(dc.ndata)
    np.random.shuffle(rand_inds)
    train_inds = rand_inds[:ntrain]
    test_inds = rand_inds[ntrain:]
    return train_inds, test_inds

def train_crossval( model, data_collection, n_repeats=1):
    '''do a cross-validation for a model given the 
    current data collection. it is 
    assumed that model has a "fit" function. 
    '''
    all_results = []
    for i in range(n_repeats):
        train_inds, test_inds = select_fit_predict(data_collection)
        print('training on data: ',train_inds)
        dc_train = data_utils.DataCollection()
        dc_test = data_utils.DataCollection()
        for train_ind in train_inds:
            dc_train.append_data(data_collection.data_collection[train_ind])
        for test_ind in test_inds:
            dc_test.append_data(data_collection.data_collection[test_ind])
        results = model.fit(dc_train)
        print('Done. Testing on data ', test_inds)
        results.test_errors = model.test(dc_test)
        results.train_data = dc_train
        results.test_data = dc_test
        all_results.append(results)
    return all_results
    