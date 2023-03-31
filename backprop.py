import numpy as np

def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        if self.with_biases:
            nabla_b = [np.zeros(b.shape) for b in self.biases]
            nabla_w = [np.zeros(w.shape) for w in self.weights]
            # feedforward
            activation = x
            activations = [x] # list to store all the activations, layer by layer
            zs = [] # list to store all the z vectors, layer by layer
            for b, w in zip(self.biases, self.weights):
                z = np.dot(w, activation) + b
                zs.append(z)
                activation = self.activation.function(z)
                activations.append(activation)
            # backward pass
            delta = (self.cost).delta(zs[-1], activations[-1], y)
            nabla_b[-1] = delta
            nabla_w[-1] = np.dot(delta, activations[-2].transpose())
            # Note that the variable l in the loop below is used a little
            # differently to the notation in Chapter 2 of the book.  Here,
            # l = 1 means the last layer of neurons, l = 2 is the
            # second-last layer, and so on.  It's a renumbering of the
            # scheme in the book, used here to take advantage of the fact
            # that Python can use negative indices in lists.
            for l in range(2, self.num_layers):
                z = zs[-l]
                sp = self.activation.derivative(z)
                delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
                nabla_b[-l] = delta
                nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        
        else:
            nabla_b = None
            nabla_w = [np.zeros(w.shape) for w in self.weights]
            # feedforward
            activation = x
            activations = [x] # list to store all the activations, layer by layer
            zs = [] # list to store all the z vectors, layer by layer
            for w in self.weights:
                z = np.dot(w, activation)
                zs.append(z)
                activation = self.activation.function(z)
                activations.append(activation)
            # backward pass
            delta = (self.cost).delta(zs[-1], activations[-1], y)
            nabla_w[-1] = np.dot(delta, activations[-2].transpose())
            for l in range(2, self.num_layers):
                z = zs[-l]
                sp = self.activation.derivative(z)
                delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
                nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)