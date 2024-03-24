from flask import Flask, render_template, request
from train import train_model

app = Flask(__name__, static_url_path='')

x_scaler, y_scaler, regr = train_model()

@app.route('/')
def root():
    return render_template('index.html', field_values = {}  )

@app.route('/predict', methods=['POST'])
def predict():
    # ['is_holiday', 'temperature', 'weekday', 'hour', 'month_day', 'year', 'month', 'weather_type', 'weather_description']
    d = {}
    d['is_holiday'] = int(request.form['isholiday'])  
    d['temperature'] = int(request.form['temperature'])
    d['weekday'] = int(request.form['day'])
    d['hour'] = int(request.form['time'][:2])
    D = request.form['datePicker']    
    d['month_day'] = int(D[8:])
    d['year'] = int(D[:4])   
    d['month'] = int(D[5:7])
    d['x0'] = request.form.get('x0') # Climate
    d['x1'] = request.form.get('x1') # Weather 

    
    # xoval = {'x0_Clear', 'x0_Clouds', 'x0_Drizzle', 'x0_Fog', 'x0_Haze',
    #          'x0_Mist', 'x0_Rain', 'x0_Smoke', 'x0_Snow', 'x0_Thunderstorm'}

    # x1val = {'x1_Sky is Clear',
    #          'x1_broken clouds',
    #          'x1_drizzle',
    #          'x1_few clouds',
    #          'x1_fog',
    #          'x1_haze',
    #          'x1_heavy intensity drizzle',
    #          'x1_heavy intensity rain',
    #          'x1_heavy snow',
    #          'x1_light intensity drizzle',
    #          'x1_light intensity shower rain',
    #          'x1_light rain',
    #          'x1_light rain and snow',
    #          'x1_light shower snow',
    #          'x1_light snow',
    #          'x1_mist',
    #          'x1_moderate rain',
    #          'x1_overcast clouds',
    #          'x1_proximity shower rain',
    #          'x1_proximity thunderstorm',
    #          'x1_proximity thunderstorm with drizzle',
    #          'x1_proximity thunderstorm with rain',
    #          'x1_scattered clouds',
    #          'x1_shower drizzle',
    #          'x1_sky is clear',
    #          'x1_sleet',
    #          'x1_smoke',
    #          'x1_snow',
    #          'x1_thunderstorm',
    #          'x1_thunderstorm with heavy rain',
    #          'x1_thunderstorm with light drizzle',
    #          'x1_thunderstorm with light rain',
    #          'x1_thunderstorm with rain',
    #          'x1_very heavy rain'
    #          }
    
    # print(xoval)
    # x0 = {}
    # x1 = {}
    # for i in xoval:
    #     x0[i] = 0
    # for i in x1val:
    #     x1[i] = 0
    # x0[d['x0']] = 1
    # x1[d['x1']] = 1
    # print(x0)
    # print(x1)

    d['x0'] = 3  # TODO: Change the logic to fetch Climate and Weather
    d['x1'] = 7
   
    final = []
    final.append(d['is_holiday'])
    final.append(d['temperature'])
    final.append(d['weekday'])
    final.append(d['hour'])
    final.append(d['month_day'])
    final.append(d['year'])
    final.append(d['month'])
    final.append(d['x0'])
    final.append(d['x1'])    

    # for i in x0:
    #     final.append(x0[i])
    # for i in x1:
    #     final.append(x1[i])
        
    input_values = x_scaler.transform([final])
    out=regr.predict(input_values)
    print(f'Before inverse Scaling : {out}')

    y_pred = y_scaler.inverse_transform([out])
    print(f'Traffic Volume : {y_pred}')

    prediction =''
    if(y_pred<=1000):        
        prediction = "Clear Roads"        
    elif y_pred>1000 and y_pred<=3000:
        prediction = "Moderate Flow"         
    elif y_pred>3000 and y_pred<=5500:
        prediction = "High Congestion"        
    else:
        prediction = "Severe Delays" 

    return render_template('output.html', prediction=prediction)
    

if __name__ == '__main__':
    app.run(debug=True)