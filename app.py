import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State

import pandas as pd
import plotly.graph_objs as go

import glob
import ntpath
import pickle
import re
import os, psutil
import json
from hurry.filesize import size

from utils import *

data_dir = "data/"

# The length and order here should be identical to the length and order of the show_table_data callback Inputs
dataset_names = [re.sub('\.csv$', '', ntpath.basename(p)) for p in glob.glob(data_dir + "*.csv")]
secondary_dataset_names = [item for item in dataset_names if item not in ["application_train"]]
id_features = ["SK_ID_PREV", "SK_ID_CURR"]

merge_aggr_funcs = ['mean','max','min','median','sum']
custom_merge_aggr_funcs = {
    "max_min_diff": lambda x: (x.max()-x.min(), "max_min_diff")
}

dfs = {}
for df_name in dataset_names:
    dfs[df_name] = pd.read_csv(f"{data_dir}{df_name}.csv")

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server

def aggr_dict_to_datatable(dataset="", aggr_dict={}):
    if dataset not in secondary_dataset_names:
        return False
    if not aggr_dict:
        aggr_dict = {}
    features_aggr_func_enabled = []
    df_numeric_features = dfs[dataset].select_dtypes(include=np.number).columns.tolist()
    for feature in df_numeric_features:
        if feature in id_features:
            continue
        enabled_values = {"Feature": feature}
        for func in merge_aggr_funcs + list(custom_merge_aggr_funcs.keys()):
            if feature in aggr_dict and func in aggr_dict[feature]:
                enabled_values[func] = 1
            else:
                enabled_values[func] = None
        features_aggr_func_enabled.append(enabled_values)
    return features_aggr_func_enabled

def datatable_to_aggr_dict(table=[]):
    non_empty = {}
    for row in table:
        non_empty[row["Feature"]] = []
        for k, v in row.items():
            if v is not None and k != "Feature":
                non_empty[row["Feature"]].append(k)
    return {k: v for k, v in non_empty.items() if v}

dash_datatable_tabs = {}
for dataset in secondary_dataset_names:
    dash_datatable_tabs[dataset] = dash_table.DataTable(
                                        id=f'{dataset}_table',
                                        columns=[{"name": i, "id": i} for i in ["Feature"] + merge_aggr_funcs + list(custom_merge_aggr_funcs.keys())],
                                        editable=True
                                    )
tabs = dbc.Tabs([dbc.Tab(tab, label=dataset) for dataset, tab in dash_datatable_tabs.items()])

merge_card = dbc.Card(
    [
        dbc.Row([
            dbc.Col(tabs, md=8),
            dbc.Col(html.Pre(id="aggr-dicts-value"), md=4),
        ])
    ],
    body=True,
)

columns_card = dbc.Card(
    [
        dbc.FormGroup(
            [
                dbc.Label("Numeric feature filling aggregation function:"),
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
                dbc.Label("Categorical feature encoding method threshold:"),
                dbc.Input(id="method-switch", type="number", value=10),
            ]
        ),
        dbc.Button("Go.", id="go-for-it", size="lg", outline=True, color="primary"),
    ],
    body=True,
)

app.layout = dbc.Container(
    [dcc.Store(id=f"{dataset}_aggr_dicts", storage_type='local') for dataset in secondary_dataset_names] +
    [dcc.Store(id="main_df", storage_type='memory')] +
    [
        html.H1("The Big Short: Credit Risk Analysis"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col([
                    html.H2("Merges"),
                    merge_card,
                    html.H2("More pre-processing"),
                    columns_card,
                    html.Div(id="results-display"),
                ], md=12),
                
            ],
            align="center",
        ),
    ],
    fluid=True,
)

# if k in list(custom_merge_aggr_funcs.keys()):
#     non_empty[row["Feature"]].append((custom_merge_aggr_funcs[k], k))
# else:
#     non_empty[row["Feature"]].append(k)

@app.callback(
    [Output(f"{dataset}_aggr_dicts", "data") for dataset in secondary_dataset_names] +
    [Output(f"{dataset}_table", "data") for dataset in secondary_dataset_names] +
    [Output("aggr-dicts-value", "children")],
    [Input(f"{dataset}_aggr_dicts", "data") for dataset in secondary_dataset_names] +
    [Input(f"{dataset}_table", "data") for dataset in secondary_dataset_names],
)
def save_load_merge_tables(*args):
    args_list = list(args)
    aggr_dict_value = {dataset: {} for dataset in secondary_dataset_names}
    ouput_indexes = [f"{dataset}_aggr_dicts" for dataset in secondary_dataset_names] + [f"{dataset}_table" for dataset in secondary_dataset_names]
    ctx = dash.callback_context
    for trigger in ctx.triggered:
        trigger_id = trigger["prop_id"].split(".")[0]
        if "_aggr_dicts" in trigger_id:
            args_list[ouput_indexes.index(trigger_id) + len(secondary_dataset_names)] = aggr_dict_to_datatable(re.sub('\_aggr_dicts$', '', trigger_id), args[ouput_indexes.index(trigger_id)])
        elif "_table" in trigger_id:
            args_list[ouput_indexes.index(trigger_id) - len(secondary_dataset_names)] = datatable_to_aggr_dict(args[ouput_indexes.index(trigger_id)])
        aggr_dict_value = {dataset: args_list[secondary_dataset_names.index(dataset)] for dataset in secondary_dataset_names}
    return args_list + [json.dumps(aggr_dict_value, indent=2)]

@app.callback(
    Output("main_df", "data"),
    Input("go-for-it", "n_clicks"),
    [State("numfiller-func", "value"), State("method-switch", "value")] +
    [State(f"{dataset}_aggr_dicts", "data") for dataset in secondary_dataset_names],
)
def produce_main_df(*args):
    n_clicks = args[0]
    numfiller_func = args[1]
    method_switch = args[2]
    aggr_dicts = list(args)[3:]
    if n_clicks:
        process = psutil.Process(os.getpid())
        memory_usage = size(process.memory_info().rss)
        main_df = dfs["application_train"]
        for table in secondary_dataset_names:
            for feature, func_list in aggr_dicts[secondary_dataset_names.index(table)].items():
                for func in func_list:
                    if func in custom_merge_aggr_funcs:
                        aggr_dicts[secondary_dataset_names.index(table)][feature][func_list.index(func)] = custom_merge_aggr_funcs[func]
            main_df = merge_with_aggr(main_df, dfs[table], "SK_ID_CURR", aggr_dicts[secondary_dataset_names.index(table)], table)
        main_df = str_catencoder(main_df, method_switch)
        main_df = na_numfiller(main_df, numfiller_func)
        main_df = na_catfiller(main_df)
        output = html.Ul(children=[
            html.Li(f"Memory Usage: {memory_usage}"),
            html.Li(f"The aggregation function used to fill numeric empty values: {numfiller_func}"),
            html.Li(f"The threshold of number of unique values used to decide whether to use One-Hot or Label encoding: {method_switch}"),
        ])
        return main_df.to_dict('records')


if __name__ == "__main__":
    app.run_server(debug=True, port=8888)