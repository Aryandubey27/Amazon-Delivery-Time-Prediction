#!/usr/bin/env python3
"""
train.py
Reproducible training script for Amazon delivery time prediction.
Usage:
    python train.py --data /path/to/amazon_delivery.csv --outdir /path/to/output_dir
Outputs:
 - best_pipeline.joblib
 - model_metrics.json
 - predictions_<model>.csv
Requires: pandas, numpy, scikit-learn, joblib, matplotlib
"""
import argparse
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import math
import warnings
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def haversine_km(lat1, lon1, lat2, lon2):
    if any(pd.isnull([lat1, lon1, lat2, lon2])):
        return np.nan
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2.0)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2.0)**2
    c = 2 * math.asin(math.sqrt(a))
    return 6371.0 * c

def feature_engineering(df):
    # Datetime parsing
    df['Order_DateTime'] = pd.to_datetime(df.get('Order_Date','') .astype(str) + ' ' + df.get('Order_Time','').astype(str), errors='coerce')
    df['Pickup_DateTime'] = pd.to_datetime(df.get('Pickup_Time',''), errors='coerce')
    time_only_mask = df['Pickup_DateTime'].isna() & df['Pickup_Time'].notna()
    if time_only_mask.any():
        comb = pd.to_datetime(df.loc[time_only_mask, 'Order_Date'].astype(str) + ' ' + df.loc[time_only_mask, 'Pickup_Time'].astype(str), errors='coerce')
        df.loc[time_only_mask, 'Pickup_DateTime'] = comb
    df['order_hour'] = df['Order_DateTime'].dt.hour
    df['order_dayofweek'] = df['Order_DateTime'].dt.dayofweek
    # distance
    df['distance_km'] = df.apply(lambda r: haversine_km(r.get('Store_Latitude'), r.get('Store_Longitude'), r.get('Drop_Latitude'), r.get('Drop_Longitude')), axis=1)
    # pickup delay minutes
    df['pickup_delay_min'] = (df['Pickup_DateTime'] - df['Order_DateTime']).dt.total_seconds() / 60.0
    # ensure numeric target
    df['Delivery_Time'] = pd.to_numeric(df.get('Delivery_Time'), errors='coerce')
    return df

def build_preprocessor(num_features, cat_features):
    num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))])
    preprocessor = ColumnTransformer(transformers=[('num', num_pipeline, num_features), ('cat', cat_pipeline, cat_features)], sparse_threshold=0)
    return preprocessor

def main(args):
    data_path = Path(args.data)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Loading data from:", data_path)
    df = pd.read_csv(data_path)
    print("Rows:", len(df), "Columns:", len(df.columns))
    df = feature_engineering(df)
    num_features = [c for c in ['Agent_Age','Agent_Rating','distance_km','pickup_delay_min','order_hour','order_dayofweek'] if c in df.columns]
    cat_features = [c for c in ['Weather','Traffic','Vehicle','Area','Category'] if c in df.columns]
    print("Numeric features:", num_features)
    print("Categorical features:", cat_features)
    df_model = df[df['Delivery_Time'].notna()].copy()
    df_model = df_model[df_model['Delivery_Time'] > 0].reset_index(drop=True)
    X = df_model[num_features + cat_features].copy()
    y = df_model['Delivery_Time'].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preprocessor = build_preprocessor(num_features, cat_features)
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1, max_depth=12),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=50, random_state=42, max_depth=6)
    }
    results = {}
    trained = {}
    for name, model in models.items():
        print("Training:", name)
        pipe = Pipeline([('preprocessor', preprocessor), ('model', model)])
        try:
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            rmse = mean_squared_error(y_test, preds, squared=False)
            r2 = r2_score(y_test, preds)
            results[name] = {'mae': float(mae), 'rmse': float(rmse), 'r2': float(r2)}
            trained[name] = pipe
            # save predictions
            pd.DataFrame({'y_true': y_test.values, 'y_pred': preds}).reset_index(drop=True).to_csv(out_dir / f'predictions_{name}.csv', index=False)
            print(" -> Done:", results[name])
        except Exception as e:
            print(" -> Failed to train", name, ":", type(e).__name__, e)
    if not results:
        raise RuntimeError("No model was trained successfully.")
    best = min(results.keys(), key=lambda k: results[k]['rmse'])
    print("Best model:", best, results[best])
    # Refit best on full dataset for deployment
    best_pipe = trained[best]
    best_pipe.fit(X, y)
    model_path = out_dir / 'best_pipeline.joblib'
    joblib.dump(best_pipe, model_path)
    metrics_path = out_dir / 'model_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump({'model_evaluations': results, 'best_model': best}, f, indent=2)
    print("Saved model to:", model_path)
    print("Saved metrics to:", metrics_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train delivery time prediction model")
    parser.add_argument('--data', type=str, default='/mnt/data/amazon_delivery.csv', help='Path to CSV dataset')
    parser.add_argument('--outdir', type=str, default='/mnt/data/project_outputs_safe', help='Output directory')
    args = parser.parse_args()
    main(args)
