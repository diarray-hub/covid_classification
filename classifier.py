"""
Classifier.py
~~~~~~~~~~
A module to implement two of the gradient descent learning (Adam / SGD)
algorithms for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features, along which features I don't mastered yet.
Part of this code also comes from the git repository of Michael Nielsen's book: "neuralnetworksanddeeplearning"
A global understanding of gradient descent and backpropagation algorithms help understanging this code
"""

#### Libraries
import random
import numpy as np
import pickle

## Importing Cost and activation classes
from costs import CrossEntropyCost
from activations import Sigmoid

class TrainingArguments():
    """
    A TrainingArguments object for collecting and managing training arguments required such as 
    -training_data
    -mini_batch_size and 
    -learning_rate
    And optionnals:
    The other non-optional parameters are self-explanatory, as is the 
    regularization parameter ``lmbda``.  The method also accepts
    ``evaluation_data``, usually either the validation or test
    data.  We can monitor the cost and accuracy on either the
    evaluation data or the training data, by setting the
    appropriate flags.  The method progressingly records four
    lists: the (per-epoch) costs on the evaluation data, the
    accuracies on the evaluation data, the costs on the training
    data, and the accuracies on the training data. All values are
    evaluated at the end of each training epoch.  So, for example,
    if we train for 30 epochs, then the lists will be a 30-element 
    list containing the cost on the evaluation data at the end of each epoch. Note that the lists
    are empty if the corresponding flag is not set.
    """   
    def __init__(self, training_data, 
                 epochs, 
                 mini_batch_size, 
                 learning_rate,
                 lmbda = 0.0,
                 optimizer = "adam", 
                 beta1=0.9, beta2=0.999, 
                 epsilon=1e-8, test_data=None,
                 monitor_evaluation_cost=False,
                 monitor_evaluation_accuracy=False,
                 monitor_training_cost=False,
                 monitor_training_accuracy=False):
        self.training_data = training_data
        self.training_epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.learning_rate = learning_rate
        self.lmbda = lmbda
        self.optimizer = optimizer.lower()
        if self.optimizer == "adam":
            self.Adam_beta1 = beta1
            self.Adam_beta2 = beta2
            self.epsilon = epsilon
        self.monitor_evaluation_cost = monitor_evaluation_cost
        self.monitor_evaluation_acc = monitor_evaluation_accuracy
        self.monitor_training_cost = monitor_training_cost
        self.monitor_training_acc = monitor_training_accuracy
        self.test_data = test_data

class Classifier(object):
    def __init__(self, sizes, activation=Sigmoid, cost=CrossEntropyCost, with_biases = True):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron. Note that in our purpose 
        The first and the last layer of our network must be respectively of size 16 and 3
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.with_biases = with_biases
        self.cost = cost
        self.activation = activation
        # Initialize Weights and Biases from a Gaussian distribution
        self.default_weight_initializer()
        
    def default_weight_initializer(self):
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        """
        if self.with_biases: self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        """Initialize the weights using a Gaussian distribution with mean 0
        and standard deviation 1.  Initialize the biases using a
        Gaussian distribution with mean 0 and standard deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        This weight and bias initializer uses the same approach as in
        Chapter 1, and is included for purposes of comparison.  It
        will usually be better to use the default weight initializer
        instead.

        """
        if self.with_biases: self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        if self.with_biases:
            for b, w in zip(self.biases, self.weights):
                a = self.activation.function(np.dot(w, a) + b)
        else:
            for w in self.weights:
                a = self.activation.function(np.dot(w, a))
        return a

    def train(self, args : TrainingArguments):
        """ Default: train the neural network using the Adam optimizer.
        Another version of gradient descent that is the mix of two learning algorithm: 
        the Momentum GD and the Root mean square propagation.
        if optimizer is not adam then Train the neural network using mini-batch stochastic gradient descent
        If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        
        self.evaluation_cost, self.evaluation_accuracy = [], []
        self.training_cost, self.training_accuracy = [], []
        assert args.mini_batch_size >= 1, "mini_batch_size must be >= 1"
        n = len(args.training_data)
        if args.test_data: n_test = len(args.test_data)
        for j in range(args.training_epochs):
            random.shuffle(args.training_data)
            mini_batches = [args.training_data[k : k + args.mini_batch_size] for k in range(0, n, args.mini_batch_size)]
            if args.optimizer == "adam":
                for mini_batch in mini_batches:
                    self.update_mini_batch_Adam(mini_batch, args.learning_rate, args.Adam_beta1, args.Adam_beta2, args.epsilon, args.lmbda, n)
            else:
                for mini_batch in mini_batches:
                    self.update_mini_batch_SGD(mini_batch, args.learning_rate, args.lmbda, n)
            print (f"Epoch {j} training complete \n----------------------------------------")
            if args.monitor_training_cost:
                cost = self.total_cost(args.training_data, args.lmbda)
                self.training_cost.append(cost)
                print (f"Loss on training data: {round(cost, 2)}")
            if args.monitor_training_acc:
                accuracy = self.accuracy(args.training_data, convert=True)
                percentage = round((accuracy / n) * 100, 2)
                self.training_accuracy.append(percentage)
                print (f"Accuracy on training data: {accuracy} / {n} ....................... ({percentage}%) accuracy")
            if args.monitor_evaluation_cost:
                cost = self.total_cost(args.test_data, args.lmbda, convert=True)
                self.evaluation_cost.append(cost)
                print (f"Loss on evaluation data: {round(cost, 2)}")
            if args.monitor_evaluation_acc:
                accuracy = self.accuracy(args.test_data)
                percentage = round((accuracy / n_test) * 100, 2)
                self.evaluation_accuracy.append(percentage)
                print (f"Accuracy on evaluation data: {accuracy} / {n_test} ....................... ({percentage}%) accuracy")
            print ("----------------------------------------")
            
    # The update method for ADAM
    def update_mini_batch_Adam(self, mini_batch, eta, beta1, beta2, epsilon, lmbda, len_traing_data):
        """Update the network's weights and biases by applying the Adam optimizer using
        backpropagation to a single mini batch sample. The `beta1`, `beta2`, and `epsilon` 
        hyperparameters are used for computing the biased-corrected moving averages.
        ``lmbda`` is the regularization parameter.
        Regularization is a machine learning technique that (Todo)"""
        if self.with_biases: nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        m = [np.zeros(w.shape) for w in self.weights]
        v = [np.zeros(w.shape) for w in self.weights]
        delta_nabla_b, delta_nabla_w = self.FLMB_backprop(mini_batch=mini_batch)
        if self.with_biases: nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        ### Momentum gradient descent
        #This algorithm is used to accelerate the gradient descent algorithm by taking into consideration 
        #the ‘exponentially weighted average’ of the gradients. Using averages makes the algorithm 
        #converge towards the minima in a faster pace. 
        moving_average = [beta1*mw + (1-beta1)*nw for mw, nw in zip(m, nabla_w)]
        ### Root mean square propagation or RMSprop is an adaptive learning algorithm that tries to improve AdaGrad. 
        # Instead of taking the cumulative sum of squared gradients like in AdaGrad, it takes the ‘exponential moving average’.
        squared_moving_average = [beta2*vw + (1-beta2)*(nw**2) for vw, nw in zip(v, nabla_w)]
        # These moving averages are used to estimate the mean and variance of the gradient distribution, 
        # which can help the optimizer to adapt the learning rate to the different parameters of the model.
        # Since m and v are initialized to 0 and beta1, beta2 ≈ 1. moving_average and squared_moving_average gain a tendency to be ‘biased towards 0’
        # This Optimizer fixes this problem by computing ‘bias-corrected’ m_hat and v_hat as below
        m_hat = [mw/(1-beta1) for mw in moving_average]
        v_hat = [vw/(1-beta2) for vw in squared_moving_average]
        self.weights = [(1-eta*(lmbda/len_traing_data))*w-(eta/np.sqrt(vh+epsilon))*mh for w, mh, vh in zip(self.weights, m_hat, v_hat)]
        # Note that the version of adam I learnt does not in account biases so this line is not part of the adam algorithm
        if self.with_biases: self.biases = [b-(eta/(len(mini_batch) + epsilon))*nb for b, nb in zip(self.biases, nabla_b)]
    
    # The update method for SGD   
    def update_mini_batch_SGD(self, mini_batch, eta, lmbda, len_traing_data):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate. ``lmbda`` is the regularization parameter.
        Regularization is a machine learning technique that (Todo)"""
        if self.with_biases: nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        delta_nabla_b, delta_nabla_w = self.FLMB_backprop(mini_batch=mini_batch)
        if self.with_biases : nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # With SGD stuffs are a bit simpler parameters are simply updated with respect to gradients and learning rate
        self.weights = [(1-eta*(lmbda/len_traing_data))*w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        if self.with_biases: self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
    
    # Fully matrix-based approach to backpropagation over all examples in the mini-batch
    def FLMB_backprop(self, mini_batch):
        """
            Fully matrix-based approach to backpropagation over a mini-batch
            instead of the training example-wise approach. This permit to
            remarkably speed up the training. But if you need to review the
            simpler approach for a better understanding. Please refers to the backprop.py file
            Return also a tuple ``(nabla_b, nabla_w)`` representing the
            gradient for the cost function C_x.  ``nabla_b`` and
            ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
            to ``self.biases`` and ``self.weights``
        """
        if len(mini_batch) > 1:
            xs = mini_batch[0][0]
            ys = mini_batch[0][1]
            for i in range(1, len(mini_batch)):
                xs = np.hstack((xs, mini_batch[i][0]))
                ys = np.hstack((ys, mini_batch[i][1]))
        else: 
            xs = mini_batch[0][0]
            ys = mini_batch[0][1]
        #feedforward
        activation = xs
        activations = [xs] # list to store all the activations matrices, layer by layer
        zs = [] # list to store all the z matrices, layer by layer
        if self.with_biases:
            nabla_b = [np.zeros(b.shape) for b in self.biases]
            nabla_w = [np.zeros(w.shape) for w in self.weights]
            for weight, bias in zip(self.weights, self.biases):
                z = np.dot(weight, activation) + bias
                zs.append(z)
                activation = self.activation.function(z)
                activations.append(activation)
            ### backward pass
            # The last layers
            delta = self.cost.delta(z=zs[-1], a=activations[-1], y=ys)
            nabla_b[-1] = np.reshape(np.sum(delta, 1) / delta.shape[1], (delta.shape[0], 1))
            nabla_w[-1] = np.dot(delta, activations[-2].transpose())
            # The other layers
            # Note that the variable l in the loop below is used a little
            # differently to the notation in Chapter 2 of the book.  Here,
            # l = 1 means the last layer of neurons, l = 2 is the
            # second-last layer, and so on.  It's a renumbering of the
            # scheme in the book, used here to take advantage of the fact
            # that Python can use negative indices in lists.
            for l in range(2, self.num_layers):
                z = zs[-l]
                activation_derivative = self.activation.derivative(z=z)
                delta = np.dot(self.weights[-l+1].transpose(), delta) * activation_derivative
                nabla_b[-l] = np.reshape(np.sum(delta, 1) / delta.shape[1], (delta.shape[0], 1))
                nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        else:
            nabla_b = None
            nabla_w = [np.zeros(w.shape) for w in self.weights]
            for weight in self.weights:
                z = np.dot(weight, activation)
                zs.append(z)
                activation = self.activation.function(z)
                activations.append(activation)
            ### backward passes
            delta = self.cost.delta(z=zs[-1], a=activations[-1], y=ys)
            nabla_w[-1] = np.dot(delta, activations[-2].transpose())
            for l in range(2, self.num_layers):
                z = zs[-l]
                activation_derivative = self.activation.derivative(z)
                delta = np.dot(self.weights[-l+1].transpose(), delta) * activation_derivative
                nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    
    def accuracy(self, data, convert=False):
        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.

        The flag ``convert`` should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data. The need for this flag arises
        due to differences in the way the results ``y`` are
        represented in the different data sets.  In particular, it
        flags whether we need to convert between the different
        representations.  It may seem strange to use different
        representations for the different data sets.  Why not use the
        same representation for all three data sets?  It's done for
        efficiency reasons -- the program usually evaluates the cost
        on the training data and the accuracy on other data sets.
        These are different types of computations, and using different
        representations speeds things up. 
        """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data, lmbda, convert=False):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorize(y, train_out=True)
            cost += self.cost.function(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)
        return cost
    
    # The method for the prediction of our Classifier
    def predict(self, predict_data):
        content = f"""ID, LABEL\n"""
        for data in predict_data:
            output = np.argmax(self.feedforward(data[0]))
            # Note that lbl_encoder take an array and will return an array 
            output = lbl_encoder.inverse_transform([output])[0]
            content += f"""{data[1]}, {output}\n"""
        with open("data/prediction.csv", "w") as pred:
            pred.write(content)
        print("Predictions done!! Check the prediction.csv file")
    
    ## You can save and load the model using pickle in the same way we did for the label_encoder object created in processing.py
    def save(self, model_name : str):
        # Save the encoder instance to a file
        with open(f'models/{model_name}.log', 'wb') as file:
            pickle.dump(self, file)
        print(f"Model saved as {model_name}")

## Miscellaneous
# The function in processing.py for vectorization
from processing import vectorize
# The LabelEncoder object in processing.py
with open('encoder.pkl', 'rb') as f:
    lbl_encoder = pickle.load(f)