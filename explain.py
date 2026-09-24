import pandas as pd
import joblib

# Load the saved model pipeline
model = joblib.load('crop_yield_model.pkl')

# Extract feature names after one-hot encoding
feature_names = model.named_steps['preprocessor'].get_feature_names_out()
importances = model.named_steps['regressor'].feature_importances_

# Create a clean DataFrame
feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_df = feature_df.sort_values(by='Importance', ascending=False)

print("\n--- Top Factors Influencing Crop Yield ---")
print(feature_df.head(10))