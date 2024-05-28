import dash
from dash import html, dcc
import dash_leaflet as dl
from dash.dependencies import Input, Output

app = dash.Dash(__name__)

app.layout = html.Div([
    dl.Map(center=[45.5236, -122.6750], zoom=10, children=[
        dl.TileLayer(),
        dl.LayerGroup(id="layer")
    ], style={'width': '1000px', 'height': '500px'}),
    html.Button("Add Marker", id="add-marker-btn", n_clicks=0)
])

# @app.callback(
#     Output('layer', 'children'),
#     [Input('add-marker-btn', 'n_clicks')],
#     prevent_initial_call=True
# )
# def add_marker(n_clicks):
#     # This will add a marker at a fixed location every time the button is clicked
#     return [dl.Marker(position=[45.5236, -122.6750 + 0.01 * n_clicks], children=[
#         dl.Tooltip("Marker {}".format(n_clicks)),
#         dl.Popup("Popup for Marker {}".format(n_clicks))
#     ])]

if __name__ == '__main__':
    app.run_server(debug=True)
    
    