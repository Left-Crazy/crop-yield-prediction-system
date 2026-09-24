import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Load the merged dataset
df = pd.read_csv('merged_crop_data.csv')

# 2. Separate Features (X) and Target (y) -- DROPPED 'Year'
X = df.drop(columns=['Record_ID', 'Yield_tons_per_ha', 'Year'])
y = df['Yield_tons_per_ha']

categorical_cols = ['State', 'District', 'Crop', 'Season', 'Soil_Type']

# 3. Create Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

# 4. Build Model Pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Fit Model
model_pipeline.fit(X_train, y_train)

# 7. Evaluate Performance
y_pred = model_pipeline.predict(X_test)
print("\n================ MODEL EVALUATION RESULT ================")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f} tons/ha")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f} tons/ha")
print("=========================================================")

# 8. Save updated model
joblib.dump(model_pipeline, 'crop_yield_model.pkl')
print("\nModel saved successfully without 'Year' as 'crop_yield_model.pkl'!")