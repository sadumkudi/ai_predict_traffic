from sklearn.preprocessing import OneHotEncoder
from flask import Flask, render_template, request
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
# data.to_csv("traffic_volume_data.csv", index=None)

# data.columns
# sns.set()
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')
# data = pd.read_csv("traffic_volume_data.csv")
data = data.sample(10000).reset_index(drop=True)
label_columns = ['weather_type', 'weather_description']
numeric_columns = ['is_holiday', 'temperature',
                       'weekday', 'hour', 'month_day', 'year', 'month']


features = numeric_columns+label_columns
X = data[features]

from functools import reduce
def unique(list1):
    ans = reduce(lambda re, x: re+[x] if x not in re else re, list1, [])
    print(ans)

n1 = data['weather_type']
n2 = data['weather_description']
unique(n1)
unique(n2)
n1features = ['Rain', 'Clouds', 'Clear', 'Snow', 'Mist',
              'Drizzle', 'Haze', 'Thunderstorm', 'Fog', 'Smoke', 'Squall']
n2features = ['light rain', 'few clouds', 'Sky is Clear', 'light snow', 'sky is clear', 'mist', 'broken clouds', 'moderate rain', 'drizzle', 'overcast clouds', 'scattered clouds', 'haze', 'proximity thunderstorm', 'light intensity drizzle', 'heavy snow', 'heavy intensity rain', 'fog', 'heavy intensity drizzle', 'shower snow', 'snow', 'thunderstorm with rain',
              'thunderstorm with heavy rain', 'thunderstorm with light rain', 'proximity thunderstorm with rain', 'thunderstorm with drizzle', 'smoke', 'thunderstorm', 'proximity shower rain', 'very heavy rain', 'proximity thunderstorm with drizzle', 'light rain and snow', 'light intensity shower rain', 'SQUALLS', 'shower drizzle', 'thunderstorm with light drizzle']

#Data Preparation

n11 = []
n22 = []
for i in range(10000):
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


# error eval
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2)
y_pred = regr.predict(testX)
print('Mean Absolute Error:', mean_absolute_error(testY, y_pred))
##############################





# #features = numeric_columns+list(ohe_features)
# features = numeric_columns
# target = ['traffic_volume']
# X = data[features]
# y = data[target]
# x_scaler = MinMaxScaler()
# X = x_scaler.fit_transform(X)
# y_scaler = MinMaxScaler()
# y = y_scaler.fit_transform(y).flatten()
# warnings.filterwarnings('ignore')
# ##################
# regr = MLPRegressor(random_state=1, max_iter=500).fit(X, y)
# new = []
# print(regr.predict(X[:10]))
# print(y[:10])
########################################################################################################


app = Flask(__name__, static_url_path='')

@app.route('/')
def root():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    d = {}

    # for key, value in request.form.items():
    #     print(f"{key}: {value}")
    



#     d['is_holiday'] = request.form['isholiday']
#     if d['is_holiday'] == 'yes':
#         d['is_holiday'] = int(1)
#     else:
#         d['is_holiday'] = int(0)
#     d['temperature'] = int(request.form['temperature'])
#     d['weekday'] = int(0)
#     D = request.form['date']
#     d['hour'] = int(request.form['time'][:2])
#     d['month_day'] = int(D[8:])
#     d['year'] = int(D[:4])
#    # should change
#     d['month'] = int(D[5:7])
#     d['x0'] = request.form.get('x0')
#     #d['y'] = request.form.get('y')
#     d['x1'] = request.form.get('x1')
#     # #DATE = request.form['time']
#     xoval = {'x0_Clear', 'x0_Clouds', 'x0_Drizzle', 'x0_Fog', 'x0_Haze',
#              'x0_Mist', 'x0_Rain', 'x0_Smoke', 'x0_Snow', 'x0_Thunderstorm'}
#     x1val = {'x1_Sky is Clear',
#              'x1_broken clouds',
#              'x1_drizzle',
#              'x1_few clouds',
#              'x1_fog',
#              'x1_haze',
#              'x1_heavy intensity drizzle',
#              'x1_heavy intensity rain',
#              'x1_heavy snow',
#              'x1_light intensity drizzle',
#              'x1_light intensity shower rain',
#              'x1_light rain',
#              'x1_light rain and snow',
#              'x1_light shower snow',
#              'x1_light snow',
#              'x1_mist',
#              'x1_moderate rain',
#              'x1_overcast clouds',
#              'x1_proximity shower rain',
#              'x1_proximity thunderstorm',
#              'x1_proximity thunderstorm with drizzle',
#              'x1_proximity thunderstorm with rain',
#              'x1_scattered clouds',
#              'x1_shower drizzle',
#              'x1_sky is clear',
#              'x1_sleet',
#              'x1_smoke',
#              'x1_snow',
#              'x1_thunderstorm',
#              'x1_thunderstorm with heavy rain',
#              'x1_thunderstorm with light drizzle',
#              'x1_thunderstorm with light rain',
#              'x1_thunderstorm with rain',
#              'x1_very heavy rain'
#              }
#     # print(xoval)
#     x0 = {}
#     x1 = {}
#     for i in xoval:
#         x0[i] = 0
#     for i in x1val:
#         x1[i] = 0
#     x0[d['x0']] = 1
#     x1[d['x1']] = 1
#     # print(x0)
#     # print(x1)
#     # print(x0)
#     # print(d)
#     final = []
#     final.append(d['is_holiday'])
#     final.append(d['temperature'])
#     final.append(d['weekday'])
#     final.append(d['hour'])
#     final.append(d['month_day'])
#     final.append(d['year'])
#     final.append(d['month'])
#     for i in x0:
#         final.append(x0[i])
#     for i in x1:
#         final.append(x1[i])
#     # print(d)
#     # print(len(final))

    input_values=[0,89,2,288.28,1,9,2,2012,10]
    input_values = x_scaler.transform([input_values])
    out=regr.predict(input_values)
    print(f'Before inverse Scaling : {out}')

    y_pred = y_scaler.inverse_transform([out])
    print(f'Traffic Volume : {y_pred}')

    predict =''
    if(y_pred<=1000):        
        predict = "Clear Roads"        
    elif y_pred>1000 and y_pred<=3000:
        predict = "Moderate Flow"         
    elif y_pred>3000 and y_pred<=5500:
        predict = "High Congestion"        
    else:
        predict = "Severe Delays"

    output = print(regr.predict(input_values))
    print(f"Testing****** {output}")
    return render_template('output.html', prediction=predict)
    

if __name__ == '__main__':
    app.run(debug=True)
