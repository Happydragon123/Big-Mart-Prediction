import pickle
import xgboost as xgb
import os

# Define the paths
model_path = os.path.expanduser('~/Desktop/BigMartPrediction/model.pkl')
new_model_path = os.path.expanduser('~/Desktop/BigMartPrediction/new_model.xgb')

# Load the model using pickle
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Check if the loaded model is an instance of Booster or XGBModel
if isinstance(model, xgb.Booster):
    # Save the model using XGBoost's save_model method
    model.save_model(new_model_path)
    print(f"Model updated and saved successfully to {new_model_path}.")
elif isinstance(model, xgb.XGBModel):
    # Extract the Booster and save it
    model.get_booster().save_model(new_model_path)
    print(f"Model updated and saved successfully to {new_model_path}.")
else:
    raise ValueError("Loaded model is not an instance of xgboost.Booster or xgboost.XGBModel")
