from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd


class DashFig:
    def __init__(self):
        # assume you have a "long-form" data frame
        # see https://plotly.com/python/px-arguments/ for more options
        self.app: Dash = Dash(__name__)

    def create_fig(self, fig, title):
        self.app.layout = html.Div(children=[
            html.H1(children=title),

            html.Div(),

            dcc.Graph(
                id='Graph',
                figure=fig
            )
        ])
        self.app.run_server(debug=False)

    def update_fig(self, fig, title):
        self.app.layout = html.Div(children=[
            html.H1(children=title),

            html.Div(),

            dcc.Graph(
                id='Graph',
                figure=fig
            )
        ])