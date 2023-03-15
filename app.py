from flask import Flask, request, render_template, Response, redirect, url_for
import os
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import func
from datetime import datetime, timedelta, date
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib import pylab
from sklearn.model_selection import cross_validate
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
import requests
import time
import webbrowser
import json
from multiprocessing import Process
import sys
import threading


basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] =\
        'sqlite:///' + os.path.join(basedir, 'database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Saves the id of the buzzer and a boolean symbolies if its active or not
class Buzzer(db.Model):
    is_active = db.Column(db.Boolean, default=False)
    id = db.Column(db.Integer, nullable=True, primary_key=True)

    def __repr__(self):
        return f'<Buzzer {self.is_active}>'

# Saves the id of the user and a boolean symbolies if its in predicting mode or not 
class Predicting(db.Model):
    id = db.Column(db.Integer, nullable=False, primary_key=True)
    is_active = db.Column(db.Boolean, default=False)


    def __repr__(self):
        return f'<Predicting {self.is_active}>'

# Saves the id of the user and the time he/she set the alarm to
class Alarm(db.Model):
    id = db.Column(db.Integer, nullable=False, primary_key=True)
    time = db.Column(db.String, nullable=False)

    def __repr__(self):
        return f'<Alarm {self.id}>'

#--------------------------------------------------------------

# Return the buzzer state (on or off)
@app.route('/buzzer_state', methods=('GET', 'POST'))
def buzeerState():
    buzzer = Buzzer.query.get_or_404(1)

    if (buzzer.is_active):
        return "True", 200

    return "False", 200

# Sets buzzer to on mode
@app.route('/buzzer_on', methods=('GET', 'POST'))
def toggleBuzzerOn():
    buzzer = Buzzer.query.get_or_404(1)
    buzzer.is_active = True
    db.session.add(buzzer)
    db.session.commit()
    return "Buzzer On", 200

# Sets buzzer to off mode
@app.route('/buzzer_off', methods=('GET', 'POST'))
def toggleBuzzerOff():
    buzzer = Buzzer.query.get_or_404(1)
    buzzer.is_active = False
    db.session.add(buzzer)
    db.session.commit()
    return "Buzzer Off", 200

# Check the predicting status (started or not)
def getPredictStatus(id):
    predict = Predicting.query.get_or_404(id)

    if (predict.is_active):
        return "True", 200

    return "False", 200

# Change the predicting status to on
def PredictStatusOn(id):
    predict = Predicting.query.get_or_404(id)
    predict.is_active = True
    db.session.add(predict)
    db.session.commit()
    return "predict On", 200

# Remove the predicting status - after the predicting is done
def removePredictStatus(id):
    predict_to_delete = Predicting.query.get_or_404(id)
    db.session.delete(predict_to_delete)
    db.session.commit()
    return "predict was removed", 200

# Add new predicting status for a user
def addPredictStatus(id):
    new_predict = Predicting.query.get(id)
    if new_predict == None:
        new_predict = Predicting(id=id, is_active=False)
    else:
        new_predict.is_active = False
    db.session.add(new_predict)
    db.session.commit()
    return "predict was added", 200

# Remove an alarm (after the time has arrived)
def removeAlarm(id):
    alarm_to_delete = Alarm.query.get_or_404(id)
    db.session.delete(alarm_to_delete)
    db.session.commit()
    return "alarm was removed", 200

# Add new alarm to the alarm table
def addAlarmToDB(time, id):
    new_alarm = Alarm.query.get(id)
    if new_alarm == None:
        new_alarm = Alarm(time=time, id=id)
    else:
        new_alarm.time = time
    db.session.add(new_alarm)
    db.session.commit()
    return "alarm was added", 200


# Runs in the background and checks the first alarm which is set
# Removes the alarm after predicting
def AlarmClock():
    while True:
        now = datetime.now() + timedelta(hours=8)
        time = now.strftime("%I:%M %p")
        next_alarm = Alarm.query.filter_by(time=time).first()

        if next_alarm != None:
            id = next_alarm.id
            toggleBuzzerOn() 
            removeAlarm(id)
            threading.Thread(target=predict, args=(id,), daemon=True).start()


# Load the ML model
def load_model():
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    return loaded_model

# Train the ML model
def train_model():
    # Load the dataset
    dataset = pd.read_csv('samples.csv')

    # shuffle and split the data
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=94)

    train_features = train_set.drop(['Class'], axis=1).copy()
    test_features = test_set.drop(['Class'], axis=1).copy()

    # create CNN model and train it
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_features, train_set['Class'], epochs=30)
    model.evaluate(test_features, test_set['Class'])

    # Save model to disk:
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

    return model


# Create a single route to read data from our WeMOS(s)
@app.route('/data', methods=['GET'])
def addData():
    ''' The one and only route. It extracts the
    data from the request, converts to float if the
    data is not None, then calls the callback if it is set
    '''

    global _callback_

    data0Str = request.args.get('data0')
    data1Str = request.args.get('data1')
    data2Str = request.args.get('data2')
    data3Str = request.args.get('data3')
    data4Str = request.args.get('data4')
    data5Str = request.args.get('data5')
    data6Str = request.args.get('data6')
    data7Str = request.args.get('data7')
    data8Str = request.args.get('data8')
    data9Str = request.args.get('data9')
    data10Str = request.args.get('data10')
    data11Str = request.args.get('data11')
    data12Str = request.args.get('data12')

    data0 = str(data0Str) if data0Str else None
    data1 = float(data1Str) if data1Str else None
    data2 = float(data2Str) if data2Str else None
    data3 = float(data3Str) if data3Str else None
    data4 = float(data4Str) if data4Str else None
    data5 = float(data5Str) if data5Str else None
    data6 = float(data6Str) if data6Str else None
    data7 = float(data7Str) if data7Str else None
    data8 = float(data8Str) if data8Str else None
    data9 = float(data9Str) if data9Str else None
    data10 = float(data10Str) if data10Str else None
    data11 = float(data11Str) if data11Str else None
    data12 = float(data12Str) if data12Str else None

    predict_status = Predicting.query.get_or_404(data0)
    if (predict_status.is_active == False):
        return "NOT NEEDED YET", 200

    # start writing to the file only when the time is arrived
    file_name = f"samples_{data0}.csv"
    f = open(file_name, "a")
    if os.stat(file_name).st_size == 0:
        headline = f"Acc_x_1,Acc_y_1,Acc_z_1,Gyr_x_1,Gyr_y_1,Gyr_z_1,Acc_x_2,Acc_y_2,Acc_z_2,Gyr_x_2,Gyr_y_2,Gyr_z_2"
        f.write(headline)
    line = f"\n{data1},{data2},{data3},{data4},{data5},{data6},{data7},{data8},{data9},{data10},{data11},{data12}"
    f.write(line)
    f.close()

    return "OK", 200


# Start predicting and update the predicting state accordignly
def predict(id):
    PredictStatusOn(id)
    start = time.time()
    now = datetime.now() + timedelta(hours=8)
    time24 = now.strftime("%H:%M")
    todays_date = date.today()
    todays_day = todays_date.day
    todays_month = todays_date.month
    todays_year = todays_date.year
    samples_file_name = f"samples_{id}.csv"
    time.sleep(2) # give the WeMos time to send data (and write to the file) before start predicting
    while True:
        data = pd.read_csv(samples_file_name)
        if len(data) > 10:
            df = pd.DataFrame(data)
            df = df.tail(10) # make prediction based on the last 10 samples
            df = df.to_numpy()
            predictions = model.predict(df)
            classes = np.argmax(predictions, axis=1)
            print(classes)
            if np.sum(classes) > 7: # a threshold for deciding the person is awake (80% or more)
                end = time.time()
                total_time = end - start
                data_file_name = f"data_{id}.csv"
                f = open(data_file_name, "a")
                if os.stat(data_file_name).st_size == 0:
                    headline = f"Day,Month,Year,Alarm Time,Total Time Until Fully Awake"
                    f.write(headline)
                line = f"{todays_day},{todays_month},{todays_year},{time24},{total_time}\n"
                f.write(line)
                f.close()
                os.remove(samples_file_name) # delete the data file used for prediction
                removePredictStatus(id)
                toggleBuzzerOff()
                return

# Landing page
@app.route('/hello')
def hello():
    return render_template('landing-page.html')

# Alarm On page
@app.route('/alarm_on', methods=['POST', 'GET'])
def alarmOn():
    user_id = request.form['id']
    user_name = request.form['name']
    print(f"User_name: {user_name}, User_id: {user_id}")
    return render_template('alarm_on.html', user_name=user_name, user_id=user_id)

# Set Alarm page
@app.route('/set_alarm', methods=['POST', 'GET'])
def setAlarm():
    user_id = request.form['id']
    user_name = request.form['name']
    return render_template('set-alarm.html', user_id=user_id, user_name=user_name)

# Get the time which was set in the set-alarm page
@app.route('/ProcessUserinfo/<string:userinfo>', methods=['POST'])
def processUserinfo(userinfo):
    userinfo = json.loads(userinfo)
    alarmTime = userinfo[0]
    userID = userinfo[1]
    if( alarmTime == "" ):
        removeAlarm(userID)
        removePredictStatus(userID)
    else:
        addAlarmToDB(alarmTime, userID)
        addPredictStatus(userID)

    return "OK", 200
    
# Analytics page
@app.route('/analytics', methods=['POST'])
def startAnalytics():
    user_id = request.form['id']
    user_name = request.form['name']

    dataFrame = pd.read_csv(f"data_{user_id}.csv")
    # time took to get up
    time_amount_column = dataFrame.iloc[:,-1]
    time_amount_column.to_numpy()
    # day
    day_column = dataFrame.iloc[: , 0]
    day_column.to_numpy()
    # month
    month_column = dataFrame.iloc[: , 1]
    month_column.to_numpy()

    months, avg_time = calc_avg_time_per_month(dataFrame)
    plot_avg_time_per_month(time_amount_column, months, avg_time)
    plot_last_month_data(dataFrame, month_column)

    alarm_time_column = dataFrame.iloc[:,3]
    alarm_time_column.to_numpy()
    hours, num_of_times = np.unique(alarm_time_column, return_counts=True)
    plot_alarm_times_count(hours, num_of_times)
    return render_template('analytics-page.html', user_name=user_name, user_id=user_id)

def calc_avg_time_per_month(dataFrame):
    # calculate average time of getting up per each month
    avg_time = []
    months = ["Jan", "Feby", "Mar", "Ap", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    Jan = dataFrame.loc[dataFrame['Month'] == 1]
    Jan = Jan.iloc[:,-1]
    Jan.to_numpy()
    avg_time.append(np.average(Jan))

    Feb = dataFrame.loc[dataFrame['Month'] == 2]
    Feb = Feb.iloc[:,-1]
    Feb.to_numpy()
    avg_time.append(np.average(Feb))

    Mar = dataFrame.loc[dataFrame['Month'] == 3]
    Mar = Mar.iloc[:,-1]
    Mar.to_numpy()
    avg_time.append(np.average(Mar))

    Ap = dataFrame.loc[dataFrame['Month'] == 4]
    Ap = Ap.iloc[:,-1]
    Ap.to_numpy()
    avg_time.append(np.average(Ap))

    May = dataFrame.loc[dataFrame['Month'] == 5]
    May = May.iloc[:,-1]
    May.to_numpy()
    avg_time.append(np.average(May))

    Jun = dataFrame.loc[dataFrame['Month'] == 6]
    Jun = Jun.iloc[:,-1]
    Jun.to_numpy()
    avg_time.append(np.average(Jun))

    Jul = dataFrame.loc[dataFrame['Month'] == 7]
    Jul = Jul.iloc[:,-1]
    Jul.to_numpy()
    avg_time.append(np.average(Jul))

    Aug = dataFrame.loc[dataFrame['Month'] == 8]
    Aug = Aug.iloc[:,-1]
    Aug.to_numpy()
    avg_time.append(np.average(Aug))

    Sep = dataFrame.loc[dataFrame['Month'] == 9]
    Sep = Sep.iloc[:,-1]
    Sep.to_numpy()
    avg_time.append(np.average(Sep))

    Oct = dataFrame.loc[dataFrame['Month'] == 10]
    Oct = Oct.iloc[:,-1]
    Oct.to_numpy()
    avg_time.append(np.average(Oct))

    Nov = dataFrame.loc[dataFrame['Month'] == 11]
    Nov = Nov.iloc[:,-1]
    Nov.to_numpy()
    avg_time.append(np.average(Nov))

    Dec = dataFrame.loc[dataFrame['Month'] == 12]
    Dec = Dec.iloc[:,-1]
    Dec.to_numpy()
    avg_time.append(np.average(Dec))

    return [months, avg_time]

def plot_avg_time_per_month(time_amount_column, months, avg_time):
    # plotting line graph
    general_avg_time = round(np.average(time_amount_column),2)
    text_for_box = "Avg time in total: {} seconds".format(general_avg_time)

    fig, ax = plt.subplots()
    # build a rectangle in axes coords
    left, width = .25, .5
    bottom, height = .4, .5
    right = left + width
    top = bottom + height
    p = plt.Rectangle((left, bottom), width, height, fill=False)
    p.set_transform(ax.transAxes)
    p.set_clip_on(False)

    plt.title("Average amount of time took to get up by months")
    plt.xlabel("Month")
    plt.ylabel("Avg Time (in seconds)")
    plt.plot(months,avg_time, color ="red")
    ax.text(right, top, text_for_box , style='italic', bbox={'alpha': 0.5, 'pad': 10, 'facecolor': "white"}, horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)

    fig.savefig('static/files/avg_time_per_month_plot.png')

def plot_last_month_data(dataFrame, month_column):
    # Last month data
    last_month_data = dataFrame.tail(30)

    last_month_days = last_month_data.iloc[:,0]
    last_month_days.to_numpy()
    last_month_months = last_month_data.iloc[:,1]
    last_month_months.to_numpy()
    last_month_times = last_month_data.iloc[:,-1]
    last_month_times.to_numpy()

    days_in_str = [str(x) for x in last_month_days] # days string
    month_in_str = [str(x) for x in last_month_months] # month string
    labels = [i+'.'+j for i,j in list(zip(days_in_str,month_in_str))]

    days_list = last_month_days.values.tolist()
    time_list = last_month_times.values.tolist()

    fig = plt.figure(figsize = (8, 5))

    # creating the bar plot
    plt.bar(labels, time_list, color ='maroon')

    plt.xticks(np.arange(0, len(days_list), step=1), labels, rotation = 45)

    plt.xlabel("Days")
    plt.ylabel("Time (seconds)")
    plt.title("Your last month history")
    fig.savefig('static/files/last_month_data_plot.png')
    

def plot_alarm_times_count(hours, num_of_times):
    hours_list = hours.tolist()
    amount_of_times_list = num_of_times.tolist()

    fig = plt.figure(figsize = (8, 4))
    # creating the bar plot
    plt.bar(hours_list, amount_of_times_list, color ='#C39BD3',
            width = 0.4)

    x_pos = np.arange(len(hours_list))
    plt.xticks(x_pos, hours_list, rotation = 45)
    plt.xlabel("Time of alarm")
    plt.ylabel("Number of times")
    plt.title("Number of times the alarm was set to each time")
    fig.savefig('static/files/alarm_times_count_plot.png')


def main():
    global model
    model = load_model()
    threading.Thread(target=AlarmClock, daemon=True).start()
    app.run(host="0.0.0.0", port='3237')


if __name__ == '__main__':
    main()
