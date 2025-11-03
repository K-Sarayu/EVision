# train_model.py
# Usage: python train_model.py --data data/car_price_prediction_.csv --target price
import argparse, os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def main(args):
    df = pd.read_csv(args.data)
    if args.target not in df.columns:
        raise SystemExit(f"Target column '{args.target}' not found in data columns: {list(df.columns)}")
    X = df.drop(columns=[args.target])
    y = df[args.target]

    # simple split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object','category']).columns.tolist()

    # preprocessing pipelines
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numeric_cols),
        ('cat', cat_pipeline, categorical_cols)
    ], remainder='drop')

    # full pipeline
    model = Pipeline([
        ('pre', preprocessor),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"Test MSE: {mse:.4f}")
    print(f"Test R^2: {r2:.4f}")

    os.makedirs('models', exist_ok=True)
    joblib.dump(model, os.path.join('models', 'rf_model.joblib'))
    print("Model saved to models/rf_model.joblib")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--target', default='price')
    args = p.parse_args()
    main(args)
