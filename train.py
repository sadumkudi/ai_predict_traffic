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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from functools import reduce

# TODO: Refactor the code:- create train, test, predict functions

def train_model():
    data = pd.read_csv('data/Train.csv')
    data['is_holiday'] = data['is_holiday'].apply(lambda x: 0 if pd.isna(x) else 1)

    data = data.sort_values(
        by=['date_time'], ascending=True).reset_index(drop=True)
    last_n_hours = [1, 2, 3, 4, 5, 6]
    for n in last_n_hours:
        data[f'last_{n}_hour_traffic'] = data['traffic_volume'].shift(n)
    data = data.dropna().reset_index(drop=True)

    data['date_time'] = pd.to_datetime(data['date_time'])
    data['hour'] = data['date_time'].map(lambda x: int(x.strftime("%H")))
    data['month_day'] = data['date_time'].map(lambda x: int(x.strftime("%d")))
    data['weekday'] = data['date_time'].map(lambda x: x.weekday()+1)
    data['month'] = data['date_time'].map(lambda x: int(x.strftime("%m")))
    data['year'] = data['date_time'].map(lambda x: int(x.strftime("%Y")))
    
    warnings.filterwarnings('ignore')

    # data = data.sample(10000).reset_index(drop=True)
    label_columns = ['weather_type', 'weather_description']
    numeric_columns = ['is_holiday', 'temperature',
                        'weekday', 'hour', 'month_day', 'year', 'month']

    features = numeric_columns + label_columns
    X = data[features]

    n1 = data['weather_type']
    n2 = data['weather_description']

    def unique(list1):
        ans = reduce(lambda re, x: re+[x] if x not in re else re, list1, [])        
    
    unique(n1)
    unique(n2)

    n1features = ['Rain', 'Clouds', 'Clear', 'Snow', 'Mist',
                'Drizzle', 'Haze', 'Thunderstorm', 'Fog', 'Smoke', 'Squall']
    n2features = ['light rain', 'few clouds', 'Sky is Clear', 'light snow', 'sky is clear', 'mist', 'broken clouds', 'moderate rain', 'drizzle', 'overcast clouds', 'scattered clouds', 'haze', 'proximity thunderstorm', 'light intensity drizzle', 'heavy snow', 'heavy intensity rain', 'fog', 'heavy intensity drizzle', 'shower snow', 'snow', 'thunderstorm with rain',
                'thunderstorm with heavy rain', 'thunderstorm with light rain', 'proximity thunderstorm with rain', 'thunderstorm with drizzle', 'smoke', 'thunderstorm', 'proximity shower rain', 'very heavy rain', 'proximity thunderstorm with drizzle', 'light rain and snow', 'light intensity shower rain', 'SQUALLS', 'shower drizzle', 'thunderstorm with light drizzle']

    
    n11 = []
    n22 = []
    
    for i in range(0, data.shape[0]):
        if(n1[i]) not in n1features:
            n11.append(0)
        else:
            n11.append((n1features.index(n1[i]))+1)
        if n2[i] not in n2features:
            n22.append(0)
        else:
            n22.append((n2features.index(n2[i]))+1)

    data['weather_type'] = n11
    data['weather_description'] = n22

    features = numeric_columns+label_columns
    target = ['traffic_volume']
    X = data[features]
    y = data[target]    

    x_scaler = MinMaxScaler()
    X = x_scaler.fit_transform(X)
    y_scaler = MinMaxScaler()
    y = y_scaler.fit_transform(y).flatten()
    warnings.filterwarnings('ignore')

    regr = MLPRegressor(random_state=1, max_iter=500).fit(X, y)
    new = []
    print('predicted output :=',regr.predict(X[:10]))
    print('Actual output :=',y[:10])

    # Error eval
    trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2)
    y_pred = regr.predict(testX)
    print('Mean Absolute Error:', mean_absolute_error(testY, y_pred))

    return x_scaler, y_scaler, regr

