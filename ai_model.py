from sklearn.preprocessing import OneHotEncoder
from sklearn import svm
from sklearn.linear_model import LinearRegression
from mlxtend.regressor import StackingRegressor
from skopt.space import Real, Categorical, Integer
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import zscore
from sklearn.ensemble import StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import seaborn as sns  
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class AIModel:
    def __init__(self):
        # Initialize your model parameters here
        self.model = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=1000)
    
    def train(self, X_train, y_train):
        pass
        
    def __unique(list1):
        # ans = reduce(lambda re, x: re+[x] if x not in re else re, list1, [])                
        pass
    
    def predict(self, X_test):
        pass