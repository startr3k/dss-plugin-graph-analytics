import dataiku
from flask import request
import simplejson as json
import traceback
import logging
import numpy as np
from dku_filtering.filtering import filter_dataframe
from dku_graph.graph import Graph


def convert_numpy_int64_to_int(o):
    if isinstance(o, np.int64):
        return int(o)
    raise TypeError


@app.route('/get_graph_data', methods=['POST'])
def get_graph_data():
    s = ""
    try:

        # Flask's built-in way to parse JSON requests
        # It handles decoding and content-type checks automatically
        data = request.get_json(force=True)  
        if data is None:
            raise Exception("Invalid or missing JSON payload")

        config_val = data.get('config', '{}')
        if isinstance(config_val, bytes):
            config_val = config_val.decode('utf-8')
        config = json.loads(config_val)

        filters_val = data.get('filters', '[]')
        if isinstance(filters_val, bytes):
            filters_val = filters_val.decode('utf-8')
        filters = json.loads(filters_val)

        s = "1"
        scale_ratio = float(data.get('scale_ratio', 1))
        s = "2"
        dataset_name = config.get('dataset_name')
        s = "3"
        df = dataiku.Dataset(dataset_name).get_dataframe(limit=100000)
        if df.empty:
            raise Exception("Dataframe is empty")

        if len(filters) > 0:  # apply filters to dataframe
            s = "4"
            df = filter_dataframe(df, filters)

        graph = Graph(config)
        s = "5"
        graph.create_graph(df)

        scale = np.sqrt(len(graph.nodes)) * 100
        graph.compute_layout(scale=scale, scale_ratio=scale_ratio)

        nodes, edges = list(graph.nodes.values()), list(graph.edges.values())

        return json.dumps({'nodes': nodes, 'edges': edges, 'groups': graph.groups}, ignore_nan=True, default=convert_numpy_int64_to_int)

    except Exception as e:
        logging.error(traceback.format_exc())
        return str(e), 500
