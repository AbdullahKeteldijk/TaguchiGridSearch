import pandas as pd
import numpy as np

import time
from joblib import Parallel, delayed
import multiprocessing

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

from progressbar import progressbar


class GridSearch():

    def __init__(self, param_grid=None):

        self.param_grid = param_grid

        return None

    # def arithmetic_sequence(self, start, end, n):
    #
    #     '''
    #
    #     This method generates an arithmetic sequence of the power 2 between two numbers.
    #     The length of the sequence is defined by the parameter 'n'.
    #
    #     The formula of the arithmetic sequence is defined as follows (Latec):
    #
    #        a_n = a_1 + 2^{(n-1)d)}
    #
    #         where a_n denotes the current number in the sequence, a_1 is the first number in the sequence,
    #         n is the total length of the sequence and d is the common difference between the numbers.
    #
    #     We generate the sequence the sequence by calculating the first and last number in the sequence.
    #     For this we only need to take two to the power of that number. The values of these two numbers allows
    #     us to calculate the common difference 'd' with the following formula:
    #
    #         d = log(a_n - a_1) / ((n-1)log(2))
    #
    #     * This is obtained by solving the first equation for 'd'.
    #
    #     parameters
    #         start: Integer - This number is the starting number for which the the arithmetic sequence is started.
    #         end: Integer - This number is the final number with which the the arithmetic sequence is ended.
    #         n: Integer - This number defines the length of the sequence.
    #
    #     return
    #         arithmetic_list: list - This is a list of the total arithmetic sequence that is generated
    #         sequence: list - This is the arithmetic sequence of length 'n'. The list is generated in
    #         a similar way as the arithmethic shown in the paper of Sano and Suzuki (2017).
    #
    #     '''
    #
    #     a_1 = 2 ** start
    #     a_n = 2 ** end
    #
    #     d = np.log(a_n - a_1) / ((n - 1) * np.log(2))
    #
    #     arithmetic_list = []
    #
    #     for i in range(-n, n + 1):
    #         a_i = a_1 + 2 ** ((i - 1) * d)
    #         arithmetic_list.append(a_i)
    #
    #     arithmetic_list.reverse()
    #
    #     sequence = np.zeros((n, 1))
    #     j = 0
    #
    #     for i in range(0, (len(arithmetic_list) - 1), 2):
    #         sequence[j] = arithmetic_list[i]
    #         j += 1
    #
    #     sequence[-1] = a_1
    #
    #     sequence = np.flip(sequence, 0)
    #
    #     return sequence

    # def SVR_model(self, theta):
    #
    #     '''
    #     Support Vector Regression formatted in such a way that it can be easily used in the scikit-optimize functions.
    #     Because of the way the optimize functions work, it is not possible to use the target and the features
    #     as parameters in this function. They are therefore called as global variables.
    #
    #     parameters
    #         theta: list - List with the parameter values that are used for parameter tuning.
    #
    #     return
    #         RMSE: float - Root Mean Square Error of the prediction of the model and the actual target.
    #
    #
    #     '''
    #
    #     C = theta[0]
    #     epsilon = theta[1]
    #     gamma = theta[2]
    #
    #     clf = SVR(C=C, epsilon=epsilon, gamma=gamma)
    #     clf.fit(self.features, self.target)
    #     Y_pred = clf.predict(self.features)
    #     self.rmse = np.sqrt(mean_squared_error(self.target, Y_pred))
    #
    #     return self.rmse


    def input_model(self, i, j, k, features, target):
        """
        Train model with given input

        parameters
            i: int - denotes index in list of parameter C.
            j: int - denotes index in list of parameter epsilon.
            k: int - denotes index in list of parameter gamma.
            FEATURES*: Numpy array - Array with the features for the prediction.
            TARGET*: Numpy array - Array with the target for the prediction.

        return
            i: int - denotes index in list of parameter C. This is returned to keep track of the results.
            j: int - denotes index in list of parameter epsilon. This is returned to keep track of the results.
            k: int - denotes index in list of parameter gamma. This is returned to keep track of the results.
            RMSE: float - Root Mean Squared Error for the model prediction.

        """

        clf = SVR(C=i, epsilon=j, gamma=k)
        clf.fit(features, target)
        Y_pred = clf.predict(features)
        RMSE = np.sqrt(mean_squared_error(target, Y_pred))

        return [i, j, k, RMSE]


    def fit(self, features, target, parameters):
        '''
        This function is a gridsearch of three parameters in the SVR model (C, epsilon, gamma).
        Each of these parameters is represented as a list of a list of 12 values as shown in the paper of
        Sano and Suzuki (2017). In this gridsearch method the model is run for each possible combination of parameter
        settings in parallel.


        parameters
            features: Numpy array - Array with the features for the prediction.
            target: Numpy array - Array with the target for the prediction.

        return
            output: numpy array - Here we store the parameters (C, epsilon, gamma), with its RMSE, for each run of the model.

        '''

        start = time.time()

        C = parameters['C']
        epsilon = parameters['epsilon']
        gamma = parameters['gamma']

        num_cores = multiprocessing.cpu_count() - 1
        print(num_cores)

        results = Parallel(n_jobs=num_cores)(delayed(self.input_model)(i, j, k, features, target)
                                             for i in C
                                             for j in epsilon
                                             for k in gamma)  # possible?

        df_results = pd.DataFrame(results, columns=['C', 'epsilon', 'gamma', 'RMSE'])

        end = time.time()
        seconds = (end - start)
        self.minutes = seconds % 3600 / 60

        return df_results

class TaguchiGridSearch(GridSearch):

    def get_OA(self):
        """
        Removing the empty columns in the orthogonal array dataframe.

        parameters
            df: pandas dataframe - Orthogonal array

        return
            OA_clean: numpy array - Array with the cleaned orthogonal array.
        """
        OA = pd.read_csv('../data/L144.csv', sep=' ', header=None).values
        OA_clean = np.zeros((len(OA), 7))
        shape = OA.shape

        for i in range(shape[0]):
            OA_clean[i] = OA[i][~np.isnan(OA[i])]

        OA_clean = OA_clean.astype(int)

        return OA_clean

    def input_model(self, row, target, features, parameters):
        '''
        Training and predicting SVR model based on the chosen parameters from the orthogonal array.
        The RMSE is then returned with the corresponding hyperparameters.

        parameters
            row: numpy array - The row out of the orthogonal array that corresponds to the current iteration of the model
            target: numpy array - The Y variable of the model.
            features: numpy array - The X variable of the model.
            parameters: dictionary - Dictionary containing a list with hyperparameter values to choose from.

        return
            list -
                    C_OA: float - C parameter picked by the orthogonal array for this iteration
                    epsilon_OA: float - epsilon parameter picked by the orthogonal array for this iteration
                    gamma_OA: float - gamma parameter picked by the orthogonal array for this iteration
                    RMSE: float - RMSE of the prediction for this iteration
        '''

        C = parameters["C"]
        epsilon = parameters["epsilon"]
        gamma = parameters["gamma"]

        C_OA = C[row[6]]
        epsilon_OA = epsilon[row[4]]
        gamma_OA = gamma[row[1]]

        clf = SVR(C=C_OA, epsilon=epsilon_OA, gamma=gamma_OA)

        clf.fit(features, target)
        Y_pred = clf.predict(features)
        RMSE = np.sqrt(mean_squared_error(target, Y_pred))

        return [C_OA, epsilon_OA, gamma_OA, RMSE]

    def fit(self, target, features, parameters, OA):
        """
        Paralellized implementation of the Taguchi gridsearch method.

        parameters
            target: numpy array - Array with the target for the prediction.
            features: numpy array - Array with the features for the prediction.
            OA: numpy array - Orthogonal array
            parameters: dictionary - Dictionary containing a list with hyperparameter values to choose from.

        return
            df_results: pandas dataframe - Dataframe with the results
            minutes: float - Elapsed time in minutes

        """

        start = time.time()
        shape = OA.shape

        num_cores = multiprocessing.cpu_count()

        results = Parallel(n_jobs=num_cores)(delayed(self.input_model)(row, target, features, parameters) for row in OA)

        df_results = pd.DataFrame(results, columns=['C', 'epsilon', 'gamma', 'RMSE'])

        end = time.time()
        seconds = (end - start)
        self.minutes = seconds % 3600 / 60

        return df_results

abalone = pd.read_csv('../data/abalone.csv')

abalone.loc[abalone.Sex == 'M', 'Sex'] = 1
abalone.loc[abalone.Sex == 'F', 'Sex'] = 2
abalone.loc[abalone.Sex == 'I', 'Sex'] = 3

feature_names = list(abalone)
feature_names.remove('Rings')

features = abalone[feature_names].values
target = abalone['Rings'].values

parameters = {
    'C' : [0.0002, 0.0011, 0.005, 0.0228, 0.1035, 0.4695, 2.13, 9.665, 43.85, 199, 902.7, 4096],
    'epsilon' : [0.00003, 0.00008, 0.00023, 0.00063, 0.00172, 0.00472, 0.01293, 0.03545, 0.09715, 0.2663, 0.7297, 2],
    'gamma' : [0.125, 0.1824, 0.2663, 0.3886, 0.5672, 0.8278, 1.208, 1.763, 2.573, 3.756, 5.481, 8]
}

grid = GridSearch()
df_grid = grid.fit(features, target, parameters)
df_grid = df_grid.sort_values(by=['RMSE'], ascending=True)
print(df_grid)

    #
    # def clean_OA(df):
    #     """
    #     Removing the empty columns in the orthogonal array dataframe.
    #
    #     parameters
    #         df: pandas dataframe - Orthogonal array
    #
    #     return
    #         OA_clean: numpy array - Array with the cleaned orthogonal array.
    #     """
    #
    #     df_ = df.values
    #     OA_clean = np.zeros((len(df_), 7))
    #     shape = df.shape
    #
    #     for i in range(shape[0]):
    #         OA_clean[i] = df_[i][~np.isnan(df_[i])]
    #
    #     OA_clean = OA_clean.astype(int)
    #
    #     return OA_clean
    #
    # def OAInput(row, target, features, parameters):
    #
    #     '''
    #     Training and predicting SVR model based on the chosen parameters from the orthogonal array.
    #     The RMSE is then returned with the corresponding hyperparameters.
    #
    #     parameters
    #         row: numpy array - The row out of the orthogonal array that corresponds to the current iteration of the model
    #         target: numpy array - The Y variable of the model.
    #         features: numpy array - The X variable of the model.
    #         parameters: dictionary - Dictionary containing a list with hyperparameter values to choose from.
    #
    #     return
    #         list -
    #                 C_OA: float - C parameter picked by the orthogonal array for this iteration
    #                 epsilon_OA: float - epsilon parameter picked by the orthogonal array for this iteration
    #                 gamma_OA: float - gamma parameter picked by the orthogonal array for this iteration
    #                 RMSE: float - RMSE of the prediction for this iteration
    #     '''
    #
    #     C = parameters["C"]
    #     epsilon = parameters["epsilon"]
    #     gamma = parameters["gamma"]
    #
    #     C_OA = C[row[6]]
    #     epsilon_OA = epsilon[row[4]]
    #     gamma_OA = gamma[row[1]]
    #
    #     clf = SVR(C=C_OA, epsilon=epsilon_OA, gamma=gamma_OA)
    #
    #     clf.fit(features, target)
    #     Y_pred = clf.predict(features)
    #     RMSE = np.sqrt(mean_squared_error(target, Y_pred))
    #
    #     return [C_OA, epsilon_OA, gamma_OA, RMSE]
    #
    # def OA_method(target, features, OA, parameters):
    #
    #     """
    #     Paralellized implementation of the Taguchi gridsearch method.
    #
    #     parameters
    #         target: numpy array - Array with the target for the prediction.
    #         features: numpy array - Array with the features for the prediction.
    #         OA: numpy array - Orthogonal array
    #         parameters: dictionary - Dictionary containing a list with hyperparameter values to choose from.
    #
    #     return
    #         df_results: pandas dataframe - Dataframe with the results
    #         minutes: float - Elapsed time in minutes
    #
    #     """
    #
    #     start = time.time()
    #     shape = OA.shape
    #
    #     num_cores = multiprocessing.cpu_count()
    #
    #     results = Parallel(n_jobs=num_cores)(delayed(OAInput)(row, target, features, parameters) for row in OA)
    #
    #     df_results = pd.DataFrame(results, columns=['C', 'epsilon', 'gamma', 'RMSE'])
    #
    #     end = time.time()
    #     seconds = (end - start)
    #     minutes = seconds % 3600 / 60
    #
    #     return df_results, minutes