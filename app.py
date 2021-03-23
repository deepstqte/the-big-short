import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State
import lightgbm as lgb
from sklearn.model_selection import train_test_split

import pandas as pd
import plotly.graph_objs as go
from dict_hash import sha256

import glob
import sys
import ntpath
import pickle
import re
import os, psutil
from os import path
import json
from hurry.filesize import size

from utils import *

data_dir = "data/"

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


def train_model(df, target_feat="TARGET", exclude_feats=["SK_ID_CURR"], n_estimators=100000, boost_from_average='false', learning_rate=0.01, num_leaves=64, num_threads=2, max_depth=-1, tree_learner="serial", feature_fraction=0.7, bagging_freq=5, bagging_fraction=0.7, min_data_in_leaf=100, max_bin=255, bagging_seed=11, early_stopping_rounds=300, test_size=0.1):
    model_lgb = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        boost_from_average=boost_from_average,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        num_threads=num_threads,
        max_depth=max_depth,
        tree_learner=tree_learner,
        feature_fraction=feature_fraction,
        bagging_freq=bagging_freq,
        bagging_fraction=bagging_fraction,
        min_data_in_leaf=min_data_in_leaf,
        silent=-1,
        verbose=-1,
        max_bin=max_bin,
        bagging_seed=bagging_seed,
    )
    features = [f for f in df.columns if f not in exclude_feats + [target_feat]]
    X = df[features]
    y = df[target_feat]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, shuffle=True, random_state=10, test_size=test_size)
    model_lgb.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='auc', verbose=200,
              early_stopping_rounds=early_stopping_rounds)
    return model_lgb

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
            dbc.Col(children=[
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
                dbc.Button("Produce Main Dataframe.", id="produce-main-df", size="lg", outline=True, color="primary"),
                dbc.FormGroup(
                    [
                        dbc.Label("Early Stopping Rounds:"),
                        dbc.Input(id="early-stopping-rounds", type="number", value=300, min=0, max=800),
                    ]
                ),
                dbc.FormGroup(
                    [
                        dbc.Label("Number of parallel threads:"),
                        dbc.Input(id="num-threads", type="number", value=6, min=1, max=8),
                    ]
                ),
                dbc.FormGroup(
                    [
                        dbc.Label("Test size:"),
                        dbc.Input(id="test-size", type="number", value=0.1, min=0.1, max=0.9, step=0.1),
                    ]
                ),
                dbc.Button("Train The Model.", id="train-model", size="lg", outline=True, color="primary"),
            ], md=4),
        ])
    ],
    body=True,
)

results_card = dbc.Card(
    [
        dbc.Row([

            dbc.Col([dbc.Label("Aggregation functions dict:"),html.Pre(id="aggr-dicts-value")], md=4),
            dbc.Col([
                dbc.Alert(
                    id="main-df-alert",
                    dismissable=True,
                    is_open=False,
                ),
                dbc.Alert(
                    id="model-training-alert",
                    dismissable=True,
                    is_open=False,
                ),
                dbc.Label("Model training logs:"),
                html.Div(id="logs-text")
            ], md=8),
        ])
    ],
    body=True,
)

app.layout = dcc.Loading(
    id="loading-1",
    fullscreen=True,
    type="default",
    children=dbc.Container(
        [dcc.Store(id=f"{dataset}_aggr_dicts", storage_type='local') for dataset in secondary_dataset_names] +
        [dcc.Store(id="main_df_params_hash", storage_type='local')] +
        [
            html.H1("The Big Short: Credit Risk Analysis"),
            html.Hr(),
            dbc.Row(
                [
                    dbc.Col([
                        html.H2("Feature engineering and model tuning"),
                        merge_card,
                        html.H2("Results"),
                        results_card,
                        html.Div(id="results-display"),
                        html.Div(id="hidden-div", style={"display":"none"}),
                        html.Div(id="hidden-div-2", style={"display":"none"})
                    ], md=12),
                    
                ],
                align="center",
            ),
        ],
        fluid=True,
    )
)

@app.callback(
    [Output(f"{dataset}_aggr_dicts", "data") for dataset in secondary_dataset_names] +
    [Output(f"{dataset}_table", "data") for dataset in secondary_dataset_names] +
    [Output("aggr-dicts-value", "children"), Output("main_df_params_hash", "data")],
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
    return args_list + [json.dumps(aggr_dict_value, indent=2), sha256(aggr_dict_value)]

@app.callback(
    [
        Output("logs-text", "children"),
        Output("model-training-alert", "is_open"),
        Output("model-training-alert", "children"),
        Output("model-training-alert", "color"),
    ],
    Input("train-model", "n_clicks"),
    [
        State("early-stopping-rounds", "value"),
        State("num-threads", "value"),
        State("test-size", "value"),
        State("numfiller-func", "value"),
        State("method-switch", "value")
    ] +
    [State(f"{dataset}_aggr_dicts", "data") for dataset in secondary_dataset_names],
)
def train_model_callback(*args):
    main_df_hash = sha256({k: v for k, v in enumerate(list(args)[4:])})
    n_clicks = args[0]
    early_stopping_rounds = args[1]
    num_threads = args[2]
    test_size = args[3]
    logs_output = ""
    if n_clicks:
        if path.isfile(f"data/cache/{main_df_hash}.csv"):
            main_df = pd.read_csv(f"data/cache/{main_df_hash}.csv")
            with open('data/logs.txt', 'w') as f:
                sys.stdout = f
                model = train_model(
                    main_df,
                    early_stopping_rounds=early_stopping_rounds,
                    num_threads=num_threads,
                    test_size=test_size
                )
        else:
            return dash.no_update, True, "Main Dataframe does not exist, produce it first.", "danger"
        logs_file = open('data/logs.txt', 'r')
        logs_lines = logs_file.readlines()
        logs_output = [html.Div(line) for line in logs_lines]
    return logs_output, False, "", "info"

@app.callback(
    [
        Output("main-df-alert", "is_open"),
        Output("main-df-alert", "children"),
        Output("main-df-alert", "color"),
    ],
    Input("produce-main-df", "n_clicks"),
    [State("numfiller-func", "value"), State("method-switch", "value")] +
    [State(f"{dataset}_aggr_dicts", "data") for dataset in secondary_dataset_names],
)
def produce_main_df(*args):
    main_df_hash = sha256({k: v for k, v in enumerate(list(args)[1:])})
    n_clicks = args[0]
    numfiller_func = args[1]
    method_switch = args[2]
    aggr_dicts = list(args)[3:]
    color = "info"
    if n_clicks:
        message = ""
        if path.isfile(f"data/cache/{main_df_hash}.csv"):
            message = "Main Dataframe exists already"
        else:
            main_df = dfs["application_train"]
            for table in secondary_dataset_names:
                if aggr_dicts[secondary_dataset_names.index(table)]:
                    for feature, func_list in aggr_dicts[secondary_dataset_names.index(table)].items():
                        for func in func_list:
                            if func in custom_merge_aggr_funcs:
                                aggr_dicts[secondary_dataset_names.index(table)][feature][func_list.index(func)] = custom_merge_aggr_funcs[func]
                main_df = merge_with_aggr(main_df, dfs[table], "SK_ID_CURR", aggr_dicts[secondary_dataset_names.index(table)], table)
            main_df = str_catencoder(main_df, method_switch)
            main_df = na_numfiller(main_df, numfiller_func)
            main_df = na_catfiller(main_df)
            main_df.to_csv(f'data/cache/{main_df_hash}.csv', index=False)
            message = "Main Dataframe saved"
            color = "success"
        return True, message, color
    else:
        return False, "", color


if __name__ == "__main__":
    app.run_server(debug=True, port=8888)