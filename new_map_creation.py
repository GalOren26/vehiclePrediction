import os

import numpy as np
from consts import Consts
import utils
import dash
from dash import html, dcc
import dash_leaflet as dl
from dash.dependencies import Input, Output,State,ALL,MATCH
import dash_bootstrap_components as dbc
import colorsys

def generate_colors(num_colors, saturation=1, lightness=1):
    return [
        '#%02x%02x%02x' % tuple(int(c * 255) for c in colorsys.hls_to_rgb(i / num_colors, lightness, saturation))
        for i in range(num_colors)
    ]

# Generate colors for voyages and clusters
voyage_colors = generate_colors(50)  # Bright colors for voyages
cluster_colors = generate_colors(10, lightness=0.8)  # Lighter colors for clusters    

voyages_path=os.path.join(Consts['trail_path'],'inffered_voyages.pkl')
clustters_path=os.path.join(Consts['trail_path'],'clustters_content.pkl')
voyages=utils.load_dict_from_pickle(voyages_path)    
# coords=[(coords[1],coords[0]) for coords in voyages['1109938'][0]['gt_data']]
clustters=utils.load_dict_from_pickle(clustters_path)    
for lp in voyages:
    for voyage_idx,voyage in enumerate(voyages[lp]):
        voyages[lp][voyage_idx]["path"]=[]
        voyages[lp][voyage_idx]["nodes"]={}
        for idx,(lat, lon,_) in  enumerate(voyages[lp][voyage_idx]['gt_data']):
            voyages[lp][voyage_idx]["path"].append((lon,lat))
            if  voyages[lp][voyage_idx]['nodes'].get((lon, lat)) is None:
                voyages[lp][voyage_idx]['nodes'][(lon, lat)]=[idx+1]
            else :
                voyages[lp][voyage_idx]['nodes'][(lon, lat)].append(idx+1)

# filer nan and replace lat/lon
for lp in clustters:
            for key_cluster,cluster in clustters[lp]['Clusters'].items():
                clustters[lp]['Clusters'][key_cluster]= [[sublist[1],sublist[0]] for sublist in cluster if not any(np.isnan(item) if isinstance(item, float) else False for item in sublist)]



def generate_colors(num_colors):
    import colorsys
    colors = []
    for i in range(num_colors):
        hue = i / num_colors
        saturation = 0.9
        lightness = 0.6 if i % 2 == 0 else 0.3  # Alternate lightness for more variety
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        hex_color = '#%02x%02x%02x' % (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        colors.append(hex_color)
    return colors

# Use this function to generate 50 distinct colors
color_palette = generate_colors(50)



# Define a function to draw a map with a line
def draw_map(centerMapCords, app):
    app.layout = html.Div(
        style={'display': 'flex', 'flexDirection': 'row'},
        children=[
            html.Div(
                style={'width': '25%', 'padding': '10px', 'overflowY': 'scroll', 'height': '500px', 'border': '1px solid lightgray'},
                children=[
                    dcc.Checklist(
                        id='lp-checklist',
                        options=[{'label': lp, 'value': lp} for lp in voyages.keys()],
                        value=[],
                        style={'display': 'block'}
                    ),
                    html.Hr(),
                    html.Div(id='lp-checklist-container')
                ]
            ),
            html.Div(style={'width': '75%'},
                     children=[
                         dl.Map(id="map", center=centerMapCords, zoom=13, children=[
                             dl.TileLayer(),
                             dl.LayerGroup(id="markers-layer")
                         ], style={'width': '1000px', 'height': '500px'}),
                         html.Button("Add Marker", id="add-marker-btn", n_clicks=0),
                        dcc.Store(id='marker-store', data={'mode': 'add', 'markers': []})
                     ])
        ])

    @app.callback(
        Output('lp-checklist-container', 'children'),
        Input('lp-checklist', 'value')
    )
    def update_lp_checklist(selected_lps):
        checklists = []
        for lp in selected_lps:
            checklists.append(
                html.Div(
                    children=[
                        html.H5(lp, style={'margin': '0'}),
                        dcc.Checklist(
                            options=[{'label': str(idx), 'value': str(idx)} for idx, voyage in enumerate(voyages[lp])],
                            value=[],
                            id={'type': 'voyage-toggle', 'index': lp},
                            style={'marginBottom': '10px'}
                        ),
                    ],
                    style={'borderBottom': '1px solid lightgray', 'padding': '10px'}
                )
            )
        return checklists
                    
    @app.callback(
        Output('markers-layer', 'children'),
        [
            Input({'type': 'voyage-toggle', 'index': ALL}, 'value'),
            Input('lp-checklist', 'value'),
            Input('add-marker-btn', 'n_clicks')
        ],
         State({'type': 'voyage-toggle', 'index': ALL}, 'id'),
    )
    def update_voyage_presentation(selected_voyages, selected_lps,add_marker_clicks,voyage_ids):
        layers = []
        ctx = dash.callback_context
        triggered = ctx.triggered[0]['prop_id'].split('.')[0]
        
       
        # Update clusters
        for lp in selected_lps:
            for i,cluster in enumerate(clustters[lp]['Clusters'].values()):
                color = cluster_colors[i % len(cluster_colors)]
                # Assuming clusters contain coordinates
                layers.append(dl.Polyline(positions=cluster, color=color, weight=6, dashArray="10,10"))
        # Update voyages
        for voyage_ids, id_info in zip(selected_voyages, voyage_ids):
            lp = id_info['index']
            for i,voyage_id in enumerate(voyage_ids):
                    color = 'red'
                    layers.append(dl.Polyline(positions=voyages[lp][int(voyage_id)]['path'], color=color, weight=8))
                    if 'add-marker-btn' in triggered:
                        for (lon, lat) in voyages[lp][int(voyage_id)]["nodes"].keys():
                            idx_list = voyages[lp][int(voyage_id)]["nodes"][(lon, lat)]
                            # Circle Marker with Tooltip
                            circle_marker = dl.Marker(position=[lon, lat], children=[
                                dl.CircleMarker(center=[lon, lat], radius=5, color='red', fill=True, fillOpacity=1.0),
                                dl.Tooltip(f"Coordinates: {lon}, {lat} | Indexes: {idx_list}")
                            ])
                            layers.append(circle_marker)
        return layers




app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                external_scripts=['https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js'])
centerCords=[32.047018,35.289289] 
color_palette = generate_colors(50)
draw_map(centerMapCords=centerCords,app=app)
app.run_server(debug=True)
