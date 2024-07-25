import folium
from graph_consts import Consts
import pandas as pd
import matplotlib.pyplot as plt
import os
df_cameras= pd.read_csv(Consts['camera_path'])
df_cameras = df_cameras.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
df_cameras.dropna(subset=['x', 'y', 'siteName'], inplace=True)
# Convert the coordinates to the proper format if needed
df_cameras['x'] = pd.to_numeric(df_cameras['x'], errors='coerce')
df_cameras['y'] = pd.to_numeric(df_cameras['y'], errors='coerce')
df_cameras.dropna(subset=['x', 'y'], inplace=True)
coordinates =  list(zip(df_cameras['x'], df_cameras['y'],df_cameras["siteName"]))
# Create a map centered around the first coordinate
m = folium.Map(location=[df_cameras['x'][0],df_cameras['y'][0]], zoom_start=5)

# Add points to the map with custom icons and popups
for coord in coordinates:
    folium.Marker(
        location=[coord[0], coord[1]],
        tooltip=f"Site: {coord[2]}<br>Coordinates: {coord[0]}, {coord[1]}",
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(m)

# Save the map to an HTML file
m.save('map.html')