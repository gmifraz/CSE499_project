from flask import Flask, request, redirect, url_for, flash, jsonify, render_template

import joblib
import pandas as pd #dataframe
import numpy as np #mathematical computations
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__, template_folder='template')


@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")



@app.route('/api/', methods=['POST'])
def get_price():
    transmission = request.form.get('transmission')
    fuel_type = request.form.get('fuel_type')
    owner_type = request.form.get('owner_type')
    model_year = int(request.form.get('model_year'))
    km_driven = int(request.form.get('km_driven'))

    features = ['Automatic', 'Manual', 'Diesel', 'Other', 'Petrol', 'owner', 'year', 'km_driven']
    fuel_dict = {'CNG': 0, 'Diesel': 1, 'Electric': 2, 'LPG': 3, 'Petrol': 4}
    owner_dict = {'First Owner': 0, 'Fourth & Above Owner': 1, 'Second Owner': 2, 'Test Drive Car': 3, 'Third Owner': 4}
    # fuel_type
    automatic = 0
    manual = 0

    # deciding fuel type
    Diesel = 0
    Other = 0
    Petrol = 0

    if fuel_type == 'Diesel':
        Diesel = 1
    elif fuel_type == 'Other':
        Other = 1
    elif fuel_type == 'Petrol':
        Petrol = 1

    # deciding transmission type
    if transmission == '1':
        automatic = 1
    else:
        manual = 1

    test_array = [automatic, manual, Diesel, Other, Petrol, owner_dict[owner_type], model_year, km_driven]

    test_array = np.array(test_array)  # convert into numpy array

    test_array = test_array.reshape(1, -1)  # reshape
    test_df = pd.DataFrame(test_array, columns=features)

    scaler_filename = "scaler.save"
    scaler = joblib.load(scaler_filename)
    scaler.clip = False
    test_df[features] = scaler.transform(test_df[features])

    # declare path where you saved your model
    model_path = 'xg_boost_model.pkl'
    # open file
    file = open(model_path, "rb")
    # load the trained model
    trained_model = joblib.load(file)

    prediction = int(trained_model.predict(test_df))
    return render_template("prediction.html", prediction = prediction)

if __name__ == '__main__':
     app.run(debug=True)














