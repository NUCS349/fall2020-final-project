import numpy as np

from src import TextClassificationModel, load_data

def test_dbpedia():
    train_acc, dev_acc, test_acc = _test_text_classification_dataset('data/dbpedia_small')
    assert test_acc > 0.945

def test_extracredit_dbpedia():
    train_acc, dev_acc, test_acc = _test_text_classification_dataset('data/dbpedia_small')
    assert test_acc > 0.96

def test_agn():
    train_acc, dev_acc, test_acc = _test_text_classification_dataset('data/agn_small')
    assert test_acc > 0.84

def test_extracredit_agn():
    train_acc, dev_acc, test_acc = _test_text_classification_dataset('data/agn_small')
    assert test_acc > 0.87

def _test_text_classification_dataset(dataset_path):
    # Because we run the same dataset multiple times in some cases (for the
    # extra credit tests), cache the results in case the model is expensive to
    # run.
    if 'cached_results' not in _test_text_classification_dataset.__dict__:
        _test_text_classification_dataset.cached_results = dict()
    if dataset_path not in _test_text_classification_dataset.cached_results.keys():
        train_exs, dev_exs, test_exs = load_data(dataset_path)
        split_exs = {
            'train': train_exs,
            'dev': dev_exs,
            'test': test_exs,
        }

        model = TextClassificationModel()
        model.train([x['text'] for x in train_exs], [int(x['label']) for x in train_exs])

        accuracies = dict()
        for splitname, exs in split_exs.items():
            preds = model.predict([x['text'] for x in exs])
            accuracies[splitname] = sum((1 if pred == int(x['label']) else 0) for pred, x in zip(preds, exs)) / len(exs)

        _test_text_classification_dataset.cached_results[dataset_path] = accuracies
    accuracies = _test_text_classification_dataset.cached_results[dataset_path]
    return accuracies['train'], accuracies['dev'], accuracies['test']

