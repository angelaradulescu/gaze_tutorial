# Library of functions for handling data from participant behavior when learning
# in a multidimensional environment with discrete features. 

import numpy as np

class Data(object):
    """ Container for data and methods.

    Parameters
    ----------
    behav_data: Pandas dataframe 
        Behavioral data for one participant.

    et_data: Pandas dataframe
        Feature-level eyetracking data for one participant. 
        Summarized as relative looking times by feature over time.
    ----------
    """

    def __init__(self, behav_data, et_data):
    
        ## Define data.
        self.behav_data = behav_data
        self.et_data = et_data
       
        ## Get other variables 
        self.n_trials = max(behav_data['Trial'])
        self.n_games = max(behav_data['Game'])
        self.game_length = len(behav_data.loc[(behav_data['Game'] == 1)])

        ## Add trial-within-game variable.
        self.behav_data['Trial_2'] = self.behav_data['Trial'] - (self.behav_data['Game']-1)*self.game_length

    def split_data(self, test_game):
        """ Splits behavioral data into training data (n-1 games) and test data (1 game).
        """

        ## Behavioral data.
        behav_training_data = self.behav_data.loc[self.behav_data['Game'] != test_game]
        behav_test_data = self.behav_data.loc[self.behav_data['Game'] == test_game]
        
        ## Eye-tracking data.
        training_trials = behav_training_data['Trial'].values
        test_trials = behav_test_data['Trial'].values
        et_training_data = self.et_data.loc[self.et_data['Trial'].isin(training_trials)]
        et_test_data = self.et_data.loc[self.et_data['Trial'].isin(test_trials)]

        return behav_training_data, behav_test_data, et_training_data, et_test_data 

