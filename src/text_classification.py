import numpy as np
import sklearn


class TextClassificationModel:
    def __init__(self):
        raise NotImplementedError()

    def train(self, texts, labels):
        """Trains the model.  You can implement this however you
        like, using the ML methods you learned during this course.  The inputs
        are raw strings, so you will need to find a way to represent them as
        feature vectors to apply most ML methods.

        Arguments:
            texts - A list of strings representing the inputs to the model
            labels - A list of integers representing the class label for each string
        Returns:
            Nothing (just updates the parameters of the model)
        """
        raise NotImplementedError()

    def predict(self, texts):
        """Predicts labels for the given texts.

        Arguments:
            texts - A list of strings
        Returns:
            A list of integers representing the corresponding class labels for the inputs
        """
        raise NotImplementedError()
