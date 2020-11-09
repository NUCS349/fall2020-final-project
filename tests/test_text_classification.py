import numpy as np

from src import TextClassificationModel, load_data

def test_dbpedia():
    np.random.seed(0)
    rng = np.random.default_rng(0)

    train_exs, dev_exs, test_exs = load_data('data/dbpedia_small')

    model = TextClassificationModel()
    model.train([x['text'] for x in train_exs], [int(x['label']) for x in train_exs])

    preds = model.predict([x['text'] for x in test_exs])
    accuracy = sum((1 if pred == int(x['label']) else 0) for pred, x in zip(preds, test_exs)) / len(test_exs)

    print('accuracy', accuracy)
    assert accuracy > 0.96

