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
def draw_map(coordinates, app):
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
                         dl.Map(id="map", center=coordinates["path"][0], zoom=13, children=[
                             dl.TileLayer(),
                             dl.Polyline(positions=coordinates["path"], color="blue", weight=5, opacity=0.8, dashArray="5"),
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
                            value=[str(idx) for idx, voyage in enumerate(voyages[lp]) if voyage.get('visible', False)],
                            id={'type': 'voyage-toggle', 'index': lp},
                            style={'marginBottom': '10px'}
                        ),
                        html.Div(id=f'voyage-container-{lp}', style={'height': '150px', 'overflowY': 'scroll', 'marginBottom': '20px'})
                    ],
                    style={'borderBottom': '1px solid lightgray', 'padding': '10px'}
                )
            )
        return checklists

    @app.callback(
        Output('markers-layer', 'children'),
        Input({'type': 'voyage-toggle', 'index': ALL}, 'value'),
        State({'type': 'voyage-toggle', 'index': ALL}, 'id')
    )
    def update_map(selected_voyages, ids):
        layers = []
        for voyage_ids, id_info in zip(selected_voyages, ids):
            lp = id_info['index']
            for voyage in voyages[lp]:
                voyage_id = str(voyages[lp].index(voyage))
                if voyage_id in voyage_ids:
                    voyage['visible'] = True
                    coords={"path":[], "nodes":{}}
                    for idx,(lat, lon,_) in  enumerate(voyage['gt_data']):
                        coords["path"].append((lon,lat))
                        if  coords['nodes'].get((lon, lat)) is None:
                            coords['nodes'][(lon, lat)]=[idx+1]
                        else :
                            coords['nodes'][(lon, lat)].append(idx+1)
                    layers.append(dl.Polyline(positions=coords['nodes'], color="blue", weight=5))
                else:
                    voyage['visible'] = False
        return layers

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                external_scripts=['https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js'])
draw_map(coords, app)
app.run_server(debug=True)