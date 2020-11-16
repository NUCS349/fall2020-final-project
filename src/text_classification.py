import numpy as np
import sklearn

class TextClassificationModel:
    def __init__(self):
        raise NotImplementedError()

    def train(self, texts, labels):
        """
        Trains the model.  The texts are raw strings, so you will need to find
        a way to represent them as feature vectors to apply most ML methods.

        You can implement this using any ML method you like.  You are also
        allowed to use third-party libraries such as sklearn, scipy, nltk, etc,
        with a couple exceptions:

        - The classes in sklearn.feature_extraction.text are *not* allowed, nor
          is any other library that provides similar functionality (creating
          feature vectors from text).  Part of the purpose of this project is
          to do the input featurization yourself.  You are welcome to look
          through sklearn's documentation for text featurization methods to get
          ideas; just don't import them.  Also note that using a library like
          nltk to split text into a list of words is fine.

        - An exception to the above exception is that you *are* allowed to use
          pretrained deep learning models that require specific featurization.
          For example, you might be interested in exploring pretrained
          embedding methods like "word2vec", or perhaps pretrained models like
          BERT.  To use them you have to use the same input features that the
          creators did when pre-training them, which usually means using the
          featurization code provided by the creators.  The rationale for
          allowing this is that we want you to have the opportunity to explore
          cutting-edge ML methods if you want to, and doing so should already
          be enough work that you don't need to also bother with doing
          featurization by hand.

        - When in doubt, ask an instructor or TA if a particular library
          function is allowed or not.

        Hints:
        - Don't reinvent the wheel; a little reading on what techniques are
          commonly used for featurizing text can go a long way.  For example,
          one such method (which has many variations) is TF-IDF:
          https://en.wikipedia.org/wiki/Tf-idf
          https://en.wikipedia.org/wiki/SMART_Information_Retrieval_System

        - There are multiple ways to complete the assignment.  With the right
          featurization strategy, you can pass the basic tests with one of the
          ML algorithms you implemented for the previous homeworks.  To pass
          the extra credit tests, you may need to use torch or sklearn unless
          your featurization is exceptionally good or you make some special
          modeifications to your previous homework code.

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
