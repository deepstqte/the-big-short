"""
Dash port of Shiny iris k-means example:

https://shiny.rstudio.com/gallery/kmeans-example.html
"""
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
from sklearn import datasets
import glob
import ntpath
import re


iris_raw = datasets.load_iris()
iris = pd.DataFrame(iris_raw["data"], columns=iris_raw["feature_names"])

data_dir = "data/"

dataset_names = [re.sub('\.csv$', '', ntpath.basename(p)) for p in glob.glob(data_dir + "*.csv")]

dfs = {}
# for df_name in dataset_names:
    # dfs[df_name] = pd.read_csv(f"{data_dir}{df_name}.csv")

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server

controls = dbc.Card(
    [
        dbc.FormGroup(
            [
                dbc.Label("Select one or more datasets"),
                dbc.Checklist(
                    options=[
                        {"label": col, "value": col} for col in dataset_names
                    ],
                    value=[],
                    id='datasets',
                    switch=True,
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Numeric feature filling aggregation function"),
                dcc.Dropdown(
                    id="numfiller-func",
                    options=[
                        {"label": col, "value": col} for col in ["mean", "median"]
                    ],
                    value="mean",
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Categorical feature encoding method threshold"),
                dbc.Input(id="method-switch", type="number", value=10),
            ]
        ),
        dbc.Button("Go for it.", id="go-for-it", size="lg", outline=True, color="primary"),
    ],
    body=True,
)

app.layout = dbc.Container(
    [
        html.H1("The Big Short: Credit Risk Analysis"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(controls, md=4),
                dbc.Col(html.Div(id="results-display"), md=8),
            ],
            align="center",
        ),
    ],
    fluid=True,
)

@app.callback(
    Output("results-display", "children"),
    Input("go-for-it", "n_clicks"),
    [
        State("datasets", "value"),
        State("numfiller-func", "value"),
        State("method-switch", "value"),
    ],
)
def make_graph(n_clicks, datasets, numfiller_func, method_switch):
    if n_clicks:
        datasets_str = ", ".join(datasets)
        output = html.Ul(children=[
            html.Li(f"The selected datasets: {datasets_str}"),
            html.Li(f"The aggregation function used to fill numeric empty values: {numfiller_func}"),
            html.Li(f"The threshold of number of unique values used to decide whether to use One-Hot or Label encoding: {method_switch}"),
        ])
        return(output)


if __name__ == "__main__":
    app.run_server(debug=True, port=8888)