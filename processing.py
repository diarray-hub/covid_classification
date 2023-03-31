import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle

"""
Encoding data that are not numerical is a common stuff in machine learning field
Since our models work with numericals data, we need to encode data_types in order to use them 
In our case the labels we are using are strings (SOFT_COVID, STRONG_COVID and DEAD)
There is number of means to encode strings. You can even write your own script to do that
But since I have not many time and I want to keep these codes simples and easily readable
I will use 'scikit-learn' LabelEncoder encoder object to do this task
Scikit-learn is a free software machine learning library for the Python programming language.
There is countless frameworks and libraries like scikit-learn nowadays (Keras, scipy, spacy, Theano...) or the well knowns Tensorflow and Pytorch
We can even use it to do this classication task. But Our purpose is to understand what is genearlly going in those frameworks
This is the reason why our main Classifier code will be written only with numpy
Numpy is a library to perform fast linear algebra
"""
lbl_encoder = LabelEncoder()

# A function making one-hot encoding for my categorical enties
def one_hot(dataset):
    """
    Since we have lot of categorical data, we will try one hot encoding for this classifier
    One-hot encoding is a technique used to represent categorical features as binary vectors. 
    Each category is represented by a binary vector where all elements are set to 0 except for the 
    element corresponding to the category, which is set to 1. You can think of it like dividing a feature
    in simpler little features with binary values. So we'll get new columns in our dataset from 17 to 43 including labels.
    This often help the algorithm managing categorical features like independents fields and thus converge faster to the minima
    """
    for column in dataset.columns:
        dataset[column] = pd.Categorical(dataset[column])
    dataset = pd.get_dummies(dataset, prefix=['usmer', 'medical_unit', 'sex', 'pneumonia', 'pregnant', 'diabetes', 'copd', 'asthma', 'inmsupr', 
                       'hipertension', 'other_disease', 'cardiovascular', 'obesity', 'renal_chronic', 'tobacco'])
    return dataset
    
def preparation(one__hot : bool):
    """
    Since we already have clean data in .csv files. 
    Most of the preprocessing job is done such as denoising or data types stuffs
    Here, we will just do a little bit coding in order to get these data usable by our Classifier
    """
    dataset = pd.read_csv("data/train.csv")
    predict_data = pd.read_csv("data/prev.csv")
    #Renaming my columns
    dataset.columns = ['usmer', 'medical_unit', 'sex', 'pneumonia', 'age', 'pregnant', 'diabetes', 'copd', 'asthma', 'inmsupr', 
                       'hipertension', 'other_disease', 'cardiovascular', 'obesity', 'renal_chronic', 'tobacco', 'label']
    
    predict_data.columns = ['id', 'usmer', 'medical_unit', 'sex', 'pneumonia', 'age', 'pregnant', 'diabetes', 'copd', 'asthma', 'inmsupr', 
                            'hipertension', 'other_disease', 'cardiovascular', 'obesity', 'renal_chronic', 'tobacco']
    # dropping ids in pred data since it they will not be used in our classifier
    predict_id = np.array(predict_data.pop('id'))
    if one__hot:
        # Labels will not be encoded by one_hot
        labels = dataset.pop('label')
        # Since age is a continuous features we will scale it by Min-max scaling to get values between 0 and 1 
        dataset_age = dataset.pop('age')
        pred_age = predict_data.pop('age')
        min_age = dataset_age.min()
        max_age = dataset_age.max()
        dataset_age = (dataset_age - min_age) / (max_age - min_age)
        pred_age = (pred_age - min_age) / (max_age - min_age)
        # Applying one-hot encoding to our data
        dataset = one_hot(dataset=dataset)
        predict_data = one_hot(dataset=predict_data)
        # Getting 'age' column and labels back before sampling
        dataset['age'] = dataset_age
        predict_data['age'] = pred_age
        dataset['label'] = labels
    # We will take a little part of training data to use it as test data
    # Here we'll randomly pick 5% the training data from train.csv
    # This test data will be used to track our model performance after the training and before the prediction
    # A random state is set for reproducibility
    training_data = dataset.sample(frac=0.95, random_state=123)
    test_data = dataset.drop(training_data.index)
    # dropping labels that will be encoded in another way
    training_outputs = np.array(training_data.pop('label'))
    test_outputs = np.array(test_data.pop('label'))
    training_inputs = np.array(training_data)
    test_inputs = np.array(test_data)
    predict_inputs = np.array(predict_data)
    # This method will permit to the Encoder to identify all classes (SOFT_COVID, STRONG_COVID and DEAD)
    lbl_encoder.fit(['SOFT_COVID', 'STRONG_COVID', 'DEAD'])
    # This method is used to encode our labels
    training_outputs = lbl_encoder.transform(training_outputs)
    test_outputs = lbl_encoder.transform(test_outputs)
    training_data = [(inp, out) for inp, out in zip(training_inputs, training_outputs)]
    test_data = [(inp, out) for inp, out in zip(test_inputs, test_outputs)]
    predict_data = [(inp, id) for inp, id in zip(predict_inputs, predict_id)]

    # Save the encoder instance to a file
    with open('encoder.pkl', 'wb') as file:
        pickle.dump(lbl_encoder, file)

    return (training_data, test_data, predict_data)

def preprocess(one__hot : bool = True):
    """
    The Classifier we are creating works with vectors, since It will take one input vector and output another one of dim 3
    So we will slightly modify our training data to much this format in order to apply Gradient descent algorithm in
    for test_data and predict_data we will just modify inputs to get vectors
    As result we will get data like below:
    Training_data: A list of tuples (inputs, outputs) where inputs are 16-dimensionnal vectors and outputs are 3-dim vectors
    Test_data: A list of tuples (inputs, outputs) where inputs are 16-dimensionnal vectors and outputs are digits from 0 to 2
    Predict_data: A list of tuples (inputs, id) where inputs are 16-dimensionnal vectors and outputs are digits from 0 to 7515 (numbers of samples)
    Note that since LabelEncoder from Scikit-learn respect an alphabetic order, labels are the following
    0: DEAD
    1: SOFT_COVID
    2:STRONG_COVID
    """
    training_data, test_data, predict_data = preparation(one__hot=one__hot)
    training_inputs = [vectorize(array=tup[0]) for tup in training_data]
    training_outputs = [vectorize(array=tup[1], train_out=True) for tup in training_data]
    training_data = list(zip(training_inputs, training_outputs))
    test_inputs = [vectorize(array=tup[0]) for tup in test_data]
    # I decided to keep test output in there original state. You'll understand with the evaluate method of the classifier
    test_outputs = [tup[1] for tup in test_data]
    test_data = list(zip(test_inputs, test_outputs))
    predict_inputs = [vectorize(array=tup[0]) for tup in predict_data]
    predict_id = [tup[1] for tup in predict_data]
    predict_data = list(zip(predict_inputs, predict_id))
    return (training_data, test_data, predict_data)

def vectorize(array, train_out=False):
    """
    Return a vectorized form of a given array which represent inputs of the model.
    The shape is depending on either the one__hot parameter in preprocess function 
    is True or False if True shape would be (16,1) and (42,1) else.
    If train_out is true return a 3-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    0, 1, 2 into a corresponding desired output from the neural network. (Classifier)
    Note that if train_out is true array must be a single integer representing the label value"""
    if train_out:
        vector = np.zeros((3, 1))
        vector[array] = 1.0
    else:
        vector = np.reshape(array, (len(array), 1))
    return vector
