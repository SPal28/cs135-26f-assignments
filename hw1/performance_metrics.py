'''
Test Cases
--------
>>> y_N = 0.0
>>> yhat_N = 4.123
>>> calc_root_mean_squared_error(y_N, yhat_N)
np.float64(4.123)

>>> y_N = np.asarray([-2, 0, 2], dtype=np.float64)
>>> yhat_N = np.asarray([-4, 0, 2], dtype=np.float64)
>>> rmse = calc_root_mean_squared_error(y_N, yhat_N)
>>> np.round(rmse, 6)
np.float64(1.154701)
'''

import numpy as np


def calc_root_mean_squared_error(y_N, yhat_N):
    r''' Compute root mean squared error given true and predicted values

    Args
    ----
    y_N : 1D array, shape (N,)
        Each entry represents 'ground truth' numeric response for an example
    yhat_N : 1D array, shape (N,)
        Each entry representes predicted numeric response for an example

    Returns
    -------
    rmse : scalar float
        Root mean squared error performance metric
        .. math:
            rmse(y,\hat{y}) = \sqrt{\frac{1}{N} \sum_{n=1}^N (y_n - \hat{y}_n)^2}
    '''
    #converts inputs to arrays to at least one dim
    y_N = np.atleast_1d(y_N)
    yhat_N = np.atleast_1d(yhat_N)
    assert y_N.ndim == 1, "its not 1"
    assert y_N.shape == yhat_N.shape, "e"
    # the size of the array will always be stored as the first element
    N = y_N.shape[0]
    # use the rsme equation to caculate rsme 
    rmse = np.sqrt(np.sum((y_N - yhat_N) ** 2) / N)

    return rmse