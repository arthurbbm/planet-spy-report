import pandas as pd
import json
import os

# Load CSV into a DataFrame
csv_file = '/home/planet-spy-report/volume/locations_without_geojson.csv'
GEOJSON_PATH = '/home/planet-spy-report/volume/locations.geojson'

df = pd.read_csv(csv_file, dtype={"geojson": str})

# If 'geojson' column doesn't exist, create it
if 'geojson' not in df.columns:
    df['geojson'] = None

# Load the GeoJSON file into a dictionary
with open(GEOJSON_PATH, 'r') as f:
    geojson_data = json.load(f)

# Iterate over the rows of the DataFrame
for index, row in df.iterrows():
    for feature in geojson_data['features']:
        if feature['properties']['name'] == row['field_id']:
            # Convert the feature to a string and store it in the 'geojson' column
            df.at[index, 'geojson'] = json.dumps(feature)
            break

# Save the updated DataFrame back to CSV
df.to_csv(csv_file, index=False)
