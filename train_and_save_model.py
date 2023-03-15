#from crypt import methods
from flask import Flask, request, render_template, Response
import os
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


# Load model
def load_model():
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")


def train_model():
    # Load the dataset
    dataset = pd.read_csv('samples.csv')

    # shuffle and split the data
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    train_set, test_set = train_test_split(dataset, test_size=0.2, random_state = 94) 

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


app = Flask(__name__)

# Create just a single route to read data from our WeMOS
@app.route('/data', methods = ['GET'])
def addData():
    ''' The one and only route. It extracts the
    data from the request, converts to float if the
    data is not None, then calls the callback if it is set
    '''
    global _callback_

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
    
    f = open(file_name, "a")
    if os.stat(file_name).st_size == 0:
        headline = f"Acc_x_1,Acc_y_1,Acc_z_1,Gyr_x_1,Gyr_y_1,Gyr_z_1,Acc_x_2,Acc_y_2,Acc_z_2,Gyr_x_2,Gyr_y_2,Gyr_z_2"
        f.write(headline)
  
    line = f"\n{data1},{data2},{data3},{data4},{data5},{data6},{data7},{data8},{data9},{data10},{data11},{data12}"
    f.write(line)
    f.close()
    
    return "OK", 200


@app.route('/predict')
def predict():
    print("START PREDICT!!")
    #time.sleep(7)
    while True:
        f = open(file_name, "r")
        data = pd.read_csv(file_name)
        if len(data) > 10:
            df = pd.DataFrame(data)
            df = df.tail(10)
            df = df.to_numpy()
            predictions = model.predict(df) # make prediction based on the last 10 samples
            classes = np.argmax(predictions, axis=1)
            if np.sum(classes) > 7:
                print("WAKE!!")
                print(classes)
                # if the alarm is still on and the person is awake - stop the buzzer
                # webbrowser.open('http://172.20.10.2/toggleSound')
                webbrowser.open('https://www.google.com/webhp?hl=iw&sa=X&ved=0ahUKEwj6v6qR4Pj6AhWF03MBHWDsC3QQPAgI')
                x = requests.get('http://172.20.10.2/toggleSound')
                print(x.status_code)
                print("AFTER TOGGLE SOUND")
                f.close()
                print("RIGHT BEFORE RETURN")
                break
        f.close()
    return "OK", 200


@app.route('/hello')
def hello():
    return render_template('landing-page.html')


@app.route('/alarm_on')
def alaromOn():
    requests.get('http://172.20.10.2/toggleSound')
    requests.get('http://167.99.78.90:3237/predict')
    return render_template('alarm_on.html')


@app.route('/set_alarm', methods=['GET', 'POST'])
def setAlarm():
    global userId
    userId = request.args.get('id')
    #file_name = f"DataFile_{userId}.csv"
    return render_template('set-alarm.html')

def main():
    global model
    global file_name
    file_name = "DataFile_default.csv"
    model = train_model()
    app.run(host = "0.0.0.0", port = '3237')


if __name__ == '__main__':
    main()
