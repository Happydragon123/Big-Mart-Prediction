from flask import Flask, request, render_template
import numpy as np
import xgboost as xgb
import os

# Loading the updated model
new_model_path = os.path.expanduser('~/Desktop/BigMartPrediction/new_model.xgb')
booster = xgb.Booster()
booster.load_model(new_model_path)

# Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        Item_Identifier = request.form["Item_Identifier"]
        Item_weight = request.form["Item_weight"]
        Item_Fat_Content = request.form["Item_Fat_Content"]
        Item_visibility = request.form["Item_visibility"]
        Item_Type = request.form["Item_Type"]
        Item_MRP = request.form["Item_MRP"]
        Outlet_identifier = request.form["Outlet_identifier"]
        Outlet_established_year = request.form["Outlet_established_year"]
        Outlet_size = request.form["Outlet_size"]
        Outlet_location_type = request.form["Outlet_location_type"]
        Outlet_type = request.form["Outlet_type"]

        # Prepare the features array
        features = np.array([[Item_Identifier, Item_weight, Item_Fat_Content, Item_visibility, Item_Type, Item_MRP, Outlet_identifier, Outlet_established_year, Outlet_size, Outlet_location_type, Outlet_type]], dtype=np.float32)
        transformed_feature = features.reshape(1, -1)

        # Prediction using the loaded booster model
        dmatrix = xgb.DMatrix(transformed_feature)
        prediction = booster.predict(dmatrix)[0]

        return render_template('index.html', prediction=prediction)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
