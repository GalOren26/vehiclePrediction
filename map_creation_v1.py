import os
import folium
from consts import Consts
import utils
import dash
from dash import html, dcc
import dash_leaflet as dl
from dash.dependencies import Input, Output,State,ALL,MATCH
import dash_bootstrap_components as dbc

voyages_path=os.path.join(Consts['trail_path'],'inffered_voyages.pkl')
clustters_path=os.path.join(Consts['trail_path'],'clustters_content.pkl')
voyages=utils.load_dict_from_pickle(voyages_path)    
# coords=[(coords[1],coords[0]) for coords in voyages['1109938'][0]['gt_data']]
clustters=utils.load_dict_from_pickle(clustters_path)    

coords={"path":[], "nodes":{}}
for idx,(lat, lon,_) in  enumerate(voyages['1109938'][0]['gt_data']):
    coords["path"].append((lon,lat))
    if  coords['nodes'].get((lon, lat)) is None:
        coords['nodes'][(lon, lat)]=[idx+1]
    else :
        coords['nodes'][(lon, lat)].append(idx+1)



# Define a function to draw a map with a line
def draw_map(coordinates,app):
    # Check if there are enough coordinates to draw a line
    if len(coordinates) < 2:
        return "Need at least two coordinates to draw a line."
    # Create a map centered around the first coordinate
    
    app.layout =  html.Div(
    style={'display': 'flex', 'flexDirection': 'row'}, 
    children=[
     html.Div(
            style={'width': '25%', 'padding': '10px'},  # Sidebar for accordion and switches
            children=[
                html.Div([  # Container for each LP and its toggle
                    html.Div([  # Sub-container for title and toggle
                        html.Div(f"LP {lp}", style={'flexGrow': 1}),
                        dbc.Switch(
                            id={'type': 'master-toggle', 'index': lp},
                            label="Toggle All",
                            value=False,
                            style={'marginBottom': '0.5rem'}
                        )
                    ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'}),
                    dbc.Accordion([
                        dbc.AccordionItem(
                            title="Voyages",
                            children=[
                                dbc.Checklist(
                                    options=[{'label': str(idx), 'value': idx} for idx, voyage in enumerate(voyages)],
                                    value=[],
                                    id={'type': 'voyage-toggle', 'index': lp},
                                    switch=True,
                                )
                            ]
                        )
                    ], start_collapsed=True)
                ]) for lp, voyages in voyages.items()
            ]
        ),
    html.Div(
                    style={'width': '75%'},  # Map takes the larger portion of the space
            children=[
    dl.Map(id ="map",center=coordinates["path"][0], zoom=13, children=[
        dl.TileLayer(),
        dl.Polyline(positions=coordinates["path"], color="blue", weight=5, opacity=0.8, dashArray="5"), 
        dl.LayerGroup(id="markers-layer")
    ], style={'width': '1000px', 'height': '500px'}),
    html.Button("Add Marker", id="add-marker-btn", n_clicks=0)
])])
    
        # Callback to add markers
    @app.callback(
        Output("markers-layer", "children"),
        [Input("add-marker-btn", "n_clicks"),
        Input({'type': 'voyage-toggle', 'index': ALL}, "value")],
        [State({'type': 'voyage-toggle', 'index': ALL}, "id")]
    )
    def update_markers_and_voyages(n_clicks, selected_voyages, ids):
        ctx = dash.callback_context
        triggered = ctx.triggered[0]['prop_id'].split('.')[0]

        layers = []
        
        if 'add-marker-btn' in triggered:
            for (lon, lat) in coords["nodes"].keys():
                idx_list = coordinates["nodes"][(lon, lat)]
                text_html = f"<div style='background-color: white; padding: 2px 5px; border-radius: 3px;'>{idx_list}</div>"
                # Circle Marker with Tooltip
                circle_marker = dl.Marker(position=[lon, lat], children=[
                    dl.CircleMarker(center=[lon, lat], radius=5, color='red', fill=True, fillOpacity=1.0),
                    dl.Tooltip(f"Coordinates: {lon}, {lat} | Indexes: {idx_list}")
                ])
                layers.append(circle_marker)
        else:  # Handling voyage toggles
            # Flatten the list of selected voyages as it's a list of lists
            selected_voyages_flat = [item for sublist in selected_voyages for item in sublist]
            for id_info, voyage_list in zip(ids, selected_voyages_flat):
                lp = id_info['index']
                for voyage_id in voyage_list:
                    voyage = voyages[lp].get(voyage_id, None)
                    if voyage:
                        path = voyage['path']
                        layers.append(dl.Polyline(positions=path, color="blue", weight=5))

        return layers
    
    @app.callback(
        Output({'type': 'voyage-toggle', 'index': MATCH}, 'value'),
        Input({'type': 'master-toggle', 'index': MATCH}, 'value'),
        State({'type': 'voyage-toggle', 'index': MATCH}, 'options')
    )
    def toggle_all_voyages(master_toggle_value, options):
        return [option['value'] for option in options] if master_toggle_value else []

# Draw the map with the coordinates
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP],
                external_scripts=['https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js'])
draw_map(coords,app)
app.run_server(debug=True)
