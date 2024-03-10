# Traffic_Prediction

Traffic Prediction using deep Learning

This model is developed based on MLPRegressor

MLPRegressor stands for Multi-layer Perceptron Regressor, which is a type of artificial neural network used for regression tasks within the field of machine learning. It's implemented in the sklearn.neural_network module of the scikit-learn library in Python. The MLPRegressor is designed to predict a continuous value given a set of input features, making it suitable for regression problems

Architecture: The MLP consists of at least three layers of nodes: an input layer, one or more hidden layers, and an output layer. Each layer is fully connected to the next one. The nodes in the hidden layers typically use a non-linear activation function to enable the network to capture complex patterns in the data.

Training: The MLPRegressor uses backpropagation for training. This involves the forward propagation of inputs through the network to generate predictions, the calculation of the loss (error) by comparing the predictions with the actual target values, and the backward propagation of the error to adjust the weights of the connections between the nodes

Activation Functions: It supports different activation functions for the hidden layers, such as ReLU (Rectified Linear Unit), sigmoid, and tanh, which introduce non-linear properties to the network

Regularization: To prevent overfitting, MLPRegressor includes options for L2 regularization (also known as weight decay) and can also use dropout.

Solver: It provides different options for the optimization algorithm used to minimize the loss function, including stochastic gradient descent (SGD), Adam, and L-BFGS.
