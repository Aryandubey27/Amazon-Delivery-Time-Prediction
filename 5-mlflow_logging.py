# 5-mlflow_logging.py
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime

# Set experiment name
mlflow.set_experiment("Amazon_Delivery_Time_Prediction")

# Load data
print("Loading data and models...")
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

with open('best_model.pkl', 'rb') as f:
    best_model = pickle.load(f)

with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Load feature importance
feature_importance = pd.read_csv('feature_importance.csv')

print("\n" + "="*60)
print("STARTING MLFLOW LOGGING")
print("="*60)

# Start MLflow run
with mlflow.start_run(run_name=f"Best_Model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Metrics:")
    print(f"  MAE:  {mae:.2f} mins")
    print(f"  RMSE: {rmse:.2f} mins")
    print(f"  R²:   {r2:.4f}")
    
    # Log parameters
    if hasattr(best_model, 'get_params'):
        params = best_model.get_params()
        for key, value in params.items():
            if value is not None and not callable(value):
                mlflow.log_param(key, value)
    
    # Log metrics
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("mse", mse)
    
    # Log model
    mlflow.sklearn.log_model(
        best_model, 
        "model",
        registered_model_name="amazon_delivery_predictor"
    )
    
    # Log preprocessor
    mlflow.sklearn.log_model(preprocessor, "preprocessor")
    
    # Log feature importance as artifact
    feature_importance.to_csv('temp_feature_importance.csv', index=False)
    mlflow.log_artifact('temp_feature_importance.csv', 'feature_importance')
    
    # Log additional info
    mlflow.set_tag("model_type", type(best_model).__name__)
    mlflow.set_tag("dataset", "Amazon Delivery Dataset")
    mlflow.set_tag("preprocessing", "StandardScaler + OneHotEncoder")
    
    # Create and log prediction samples
    sample_predictions = pd.DataFrame({
        'Actual': y_test[:20],
        'Predicted': y_pred[:20],
        'Error': np.abs(y_test[:20] - y_pred[:20])
    })
    sample_predictions.to_csv('temp_predictions.csv', index=False)
    mlflow.log_artifact('temp_predictions.csv', 'sample_predictions')
    
    print("\n✅ MLflow logging complete!")
    print(f"Run ID: {mlflow.active_run().info.run_id}")
    print(f"\nView your experiment at: http://localhost:5000")

print("\n" + "="*60)
print("TO VIEW MLFLOW UI, RUN:")
print("mlflow ui")
print("="*60)