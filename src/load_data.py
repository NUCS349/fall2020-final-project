import json
import os
import pickle

def load_synth_data(dataset_name, base_folder='data'):
    """
    This function loads the synthesized data provided in a picke file in the
    /data directory.
    """

    data_path = os.path.join(base_folder, dataset_name)

    with open(data_path, 'rb') as handle:
        data = pickle.load(handle)

    trainX = data['trainX']
    trainY = data['trainY']

    return trainX, trainY

def load_data(path):
    if 'cache' not in load_data.__dict__:
        load_data.cache = dict()
    if path not in load_data.cache:
        load_data.cache[path] = dict()
        with open(os.path.join(path, 'train.jsonl')) as f:
            train_exs = [json.loads(line) for line in f]
        with open(os.path.join(path, 'dev.jsonl')) as f:
            dev_exs = [json.loads(line) for line in f]
        with open(os.path.join(path, 'test.jsonl')) as f:
            test_exs = [json.loads(line) for line in f]
        load_data.cache[path] = {
            'train_exs': train_exs,
            'dev_exs': dev_exs,
            'test_exs': test_exs,
            }
    return load_data.cache[path]['train_exs'], load_data.cache[path]['dev_exs'], load_data.cache[path]['test_exs']
