import pandas as pd

# 1. Load both CSV datasets
soil_df = pd.read_csv('national_soil_nutrients.csv')
weather_df = pd.read_csv('national_weather_yield.csv')

print("Soil Data Shape:", soil_df.shape)
print("Weather Data Shape:", weather_df.shape)

# 2. Merge on 'Record_ID' along with shared contextual columns
merge_keys = ['Record_ID', 'State', 'District', 'Crop', 'Year']
merged_df = pd.merge(soil_df, weather_df, on=merge_keys)

print("\n--- Merged Dataset Preview ---")
print(merged_df.head())
print("\nMerged Dataset Columns:", merged_df.columns.tolist())

# 3. Export the combined dataset for model training
merged_df.to_csv('merged_crop_data.csv', index=False)
print("\nSaved combined data to 'merged_crop_data.csv'")