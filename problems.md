# Coding (7 points)

This assignment is a combination of two assignments.  The first part is convolutional neural networks, which builds on your neural network homework.  The second part is a small project in which you will use the knowledge you gained over the course of this quarter to build a machine learning system that classifies snippets of text into categories.

The convolutional neural net task involves implementing the models in `src/models.py`.  The text classification task involves implementing the text classification class in `src/text_classificaion.py`, which involves two main parts:

- Converting raw text into feature vectors that can be processed by your machine learning algorithm.  We've been giving you ready-to-use feature matrices up until this point in the quarter, but deciding how to make features from raw data is an important part of machine learning in practice, so that's what you'll be doing here.
- Choosing a machine learning model to learn from the features

Read the comments in the `train` method of the TextClassificationModel class for requirements and hints.  You aren't required to use any particular algorithm; the goal is to have you explore various featurization and machine learning methods to find a combination that acheives the desired accuracy, which is how ML commonly works in the real world.

**IMPORTANT:** There are two extra credit tests in the autograder, called "test_extracredit_dbpedia" and "test_extracredit_agn".  You don't need to pass those to get a 100/100 score.  If you pass all tests including those ones, you'll get 110/100, which is about 0.7 total points of extra credit.

**ALSO IMPORTANT:** Given the large number of students in the course, your code will be limited by default to a total of 30 minutes of runtime on our server when we run the autograder.  If you opt to use a computationally expensive model to get the highest possible accuracy please let us know so we can ensure you get enough runtime.

You should make a conda environment for this homework just like you did for previous homeworks. We have included a requirements.txt.

# Free-response questions (3 points)

To answer the free-response questions, you will have to write extra code (that is not covered by the test cases). You may include your experiments in new files in the `experiments` directory. See `experiments/example.py` for an example. You can run any experiments you create within this directory with `python -m experiments.<experiment_name>`. For example, `python -m experiments.example` runs the example experiment. You must hand in whatever code you write for experiments by pushing to github (as you did for all previous assignments). 

**NOTE: if we have any doubts about your experiments we reserve the right to check this code to see if your results could have been generated using this code. If we don't believe it, or if there is no code at all, then you may receive a 0 for any free-response answer that would have depended on running code.**


## Convolutional layers (1 point)

Convolutional layers are layers that sweep over and subsample their input in order to represent complex structures in the input layers. For more information about how they work, [see this blog post](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/). Don't forget to read the PyTorch documentation about Convolutional Layers (linked above).

10. (0.5 points) Convolutional layers produce outputs that are of different size than their input by representing more than one input pixel with each node. If a 2D convolutional layer has `3` channels, batch size `16`, input size `(32, 32, 1)`, padding `(4, 8)`, dilation `(1, 1)`, kernel size `(8, 4)`, and stride `(2, 2)`, what is the output size of the layer (1 input layer, 3 output layers)?

If you're unsure about the answer, explain why you came up with the specific output size and we can give you points for your derivation, even it's wrong.

11. (0.5 point) Combining convolutional layers with fully connected layers can provide a boon in scenarios involving learning from images. Using a similar architecture to the one used in hw7 (neural networks) question 8, replace each of your first two hidden layers with a convolutional layer, and add a fully connected layer to output predictions as before. The number of filters (out_channels) should be 16 for the first convolutional layer and 32 for the second convolutional layer. When you call the PyTorch convolutional layer function, leave all of the arguments to their default settings except for kernel size and stride. Determine reasonable values of kernel size and stride for each layer and report what you chose. Tell us how many connections (weights) this network has.


12. (1 point) Train your convolutional model on DogSet. After every epoch, record four things: the loss of your model on the training set, the loss of your model on the validation set, and the accuracy of your model on both training and validation sets. (Use the same batch size, max epochs, learning rate)

    * Report the number of epochs your model trained, before terminating.
  
    * Make a graph that has both training and validation loss on the y-axis and epoch on the x-axis.
  
    * Make a graph that has both training and validation accuracy on the y-axis and epoch on the x-axis. 

    * Report the accuracy of your model on the testing set.


## Digging more deeply into convolutional networks (1 point) ##

The most important property of convolutional networks is their capability in capturing **shift invariant** patterns. You will investigate this property by training a convolutional network to classify simple synthesized images and visualizing the learned kernels. 

**Exploring the synthesized dataset:** Download the [synth_data file](https://nucs349.github.io/data/synth_data.zip), unzip it, and put it in `/data` directory. `synth_data` contains 10000 images of simple patterns, divided into 2 classes (5000 images per class). Use the `load_synth_data` function in `data/load_data.py` to load the training features (images) and labels. 

13. (1 point) Go through a few images and plot two examples (1 from each class). What is the common feature among the samples included in each class? What is different from one sample to the next in each class? What information must a classifier rely on to be able to tell these classes apart?


**Build the classifier:** Create a convolutional neural network including three convolutional layers and a linear output layer. The numbers and sizes of filters should be as follows:

* First layer: 2 filters of size (5,5)

* Second layer: 4 filters of size (3,3)

* Third layer: 8 filters of size (3,3)

Use strides of size (1,1) and ReLU activation functions in all convolutional layers. Each convolutional layer should be followed by max-pooling with a kernel size of 2. Use an output linear layer with two nodes, one for each class (note that for binary classification you can use a single node with a sigmoid activation function and binary cross entropy loss, but using softmax and cross entropy keeps the code simpler in this homework).

**Training parameters:** Use a cross entropy loss and the SGD optimizer. Set the batch size to 50 and learning rate to 1e-4. Train the network for 50 epochs.   

14. (1 point) Once the network is trained extract and plot the weights of the two kernels in the first layer. Do these kernels present any particular patterns? If so, what are those patterns and how are they related to the classification task at hand and the classifier performance? Note that since the model is randomly initialized (by default in PyTorch), the shape of kernels might be different across different training sessions. Repeat the experiment a few times and give a brief description of your observations.


## Text Classification (1 point)

Write up a short description of the thought process and steps you took to complete the text classification coding part of the assignment.  Mention any methods (featurization+models) you tried that didn't give the desired results as well as a short description (sentence or two at least, but longer if you want to give details) of the method you arrived at in the end.  Write down the final accuracy you got on the test set for each of the two tasks (dbpedia and agn) just in case we aren't able to reproduce your results.

Also let us know how long you spent working on this part of the assignment and any feedback you have for it (this will help us judge the difficulty for future courses).
