from string import punctuation, digits
import numpy as np
import random

# Part I

def get_order(n_samples):
    try:
        # Try to open a file with the name 'n_samples.txt'
        with open(str(n_samples) + '.txt') as fp:
            # Read the first line from the file
            line = fp.readline()
            # Split the line by commas and convert each element to an integer
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        # If the file is not found, generate a random order
        # Set a seed for reproducibility
        random.seed(1)
        # Generate a list of indices from 0 to n_samples - 1
        indices = list(range(n_samples))
        # Shuffle the list randomly
        random.shuffle(indices)
        # Return the shuffled list
        return indices



def hinge_loss_single(feature_vector, label, theta, theta_0):
    # Calculate the dot product of feature_vector and theta, then add theta_0
    h = np.sum(np.multiply(feature_vector, theta)) + theta_0
    
    # Calculate the hinge loss using the formula: max(0, 1 - h * label)
    ret = max(0, 1 - h * label)
    
    # Return the calculated hinge loss
    return ret




def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    
    # Calculate the raw hinge losses for each data point
    raw_losses = np.maximum(0, 1 - labels * (np.sum(feature_matrix * theta, axis=1) + theta_0))
    # Calculate the average hinge loss
    avg_loss = np.mean(raw_losses)
    return avg_loss

   


def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):

    # Check if the current prediction is incorrect
    if label * (np.dot(current_theta, feature_vector) + current_theta_0) <= 0:
        # If prediction is incorrect, update the weight vector and bias term
        current_theta += label * feature_vector
        current_theta_0 += label
    return (current_theta, current_theta_0)


def perceptron(feature_matrix, labels, T):

    # Initialize weight vector and bias term to zeros
    theta = np.zeros(feature_matrix.shape[1])
    theta_0 = 0
    
    # Iterate over T epochs
    for t in range(T):
        # Iterate over data points in a random order
        for i in get_order(feature_matrix.shape[0]):
            # Perform a single step update of the Perceptron algorithm
            theta, theta_0 = perceptron_single_step_update(feature_matrix[i, :], labels[i], theta, theta_0)
    
    # Return the learned weight vector and bias term
    return (theta, theta_0)


def average_perceptron(feature_matrix, labels, T):

    num_data_points = feature_matrix.shape[0]
    theta = np.zeros(feature_matrix.shape[1])
    theta_0 = 0

    theta_cache = np.zeros(feature_matrix.shape[1])
    theta_0_cache = 0
    for _ in range(T):
        for i in get_order(num_data_points):
            # Check if the prediction is incorrect
            if labels[i] * (np.dot(theta, feature_matrix[i, :]) + theta_0) <= 0:
                # Update parameters using single step update function
                theta, theta_0 = perceptron_single_step_update(feature_matrix[i, :], labels[i], theta, theta_0)
            # Accumulate parameters for averaging
            theta_cache += theta
            theta_0_cache += theta_0
    
    # Calculate the average parameters
    theta_final = theta_cache / (num_data_points * T)
    theta_0_final = theta_0_cache / (num_data_points * T)
    
    return (theta_final, theta_0_final)

def pegasos_single_step_update(
    feature_vector,  # Input feature vector for the data point.
    label,  # Label of the data point.
    L,  # Regularization parameter.
    eta,  # Learning rate.
    current_theta,  # Current weight vector.
    current_theta_0):  # Current bias term.

    # Check if the data point is correctly classified or not.
    if label * (np.sum(feature_vector * current_theta) + current_theta_0) <= 1:
        # Update the weight vector and bias term if the data point is correctly classified.
        new_theta = (1 - eta * L) * current_theta + eta * label * feature_vector
        new_theta_0 = current_theta_0 + eta * label
    else:
        # Update only the weight vector if the data point is misclassified.
        new_theta = (1 - eta * L) * current_theta
        new_theta_0 = current_theta_0
    
    # Return the updated weight vector and bias term.
    return (new_theta, new_theta_0)

def pegasos(feature_matrix, labels, T, L):

    # Get the shape of the feature matrix.
    (nsamples, nfeatures) = feature_matrix.shape

    # Initialize the weight vector and bias term.
    theta = np.zeros(nfeatures)
    theta_0 = 0
    
    # Initialize counter for the number of updates.
    count = 0
    
    # Iterate over the specified number of iterations.
    for t in range(T):
        # Iterate over the data points in a random order.
        for i in get_order(nsamples):
            # Increment the update count.
            count += 1
            
            # Calculate the learning rate (eta) for the current iteration.
            eta = 1.0 / np.sqrt(count)
            
            # Perform a single step update of the Pegasos algorithm.
            (theta, theta_0) = pegasos_single_step_update(
                feature_matrix[i], labels[i], L, eta, theta, theta_0)
    
    # Return the trained weight vector and bias term.
    return (theta, theta_0)


# Part II


def classify(feature_matrix, theta, theta_0):
    """
    A classification function that uses theta and theta_0 to classify a set of
    data points.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
                theta - A numpy array describing the linear classifier.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.

    Returns: A numpy array of 1s and -1s where the kth element of the array is
    the predicted classification of the kth row of the feature matrix using the
    given theta and theta_0. If a prediction is GREATER THAN zero, it should
    be considered a positive classification.
    """
    # Your code here
    raise NotImplementedError


def classifier_accuracy(
        classifier,
        train_feature_matrix,
        val_feature_matrix,
        train_labels,
        val_labels,
        **kwargs):
    """
    Trains a linear classifier and computes accuracy.
    The classifier is trained on the train data. The classifier's
    accuracy on the train and validation data is then returned.

    Args:
        classifier - A classifier function that takes arguments
            (feature matrix, labels, **kwargs) and returns (theta, theta_0)
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the validation
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        **kwargs - Additional named arguments to pass to the classifier
            (e.g. T or L)

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the
    accuracy of the trained classifier on the validation data.
    """
    # Your code here
    raise NotImplementedError


def extract_words(input_string):
    """
    Helper function for bag_of_words()
    Inputs a text string
    Returns a list of lowercase words in the string.
    Punctuation and digits are separated out into their own words.
    """
    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')

    return input_string.lower().split()


def bag_of_words(texts):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input

    Feel free to change this code as guided by Problem 9
    """
    # Your code here
    dictionary = {} # maps word to unique index
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word not in dictionary:
                dictionary[word] = len(dictionary)
    return dictionary


def extract_bow_feature_vectors(reviews, dictionary):
    """
    Inputs a list of string reviews
    Inputs the dictionary of words as given by bag_of_words
    Returns the bag-of-words feature matrix representation of the data.
    The returned matrix is of shape (n, m), where n is the number of reviews
    and m the total number of entries in the dictionary.

    Feel free to change this code as guided by Problem 9
    """
    # Your code here

    num_reviews = len(reviews)
    feature_matrix = np.zeros([num_reviews, len(dictionary)])

    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word in dictionary:
                feature_matrix[i, dictionary[word]] = 1
    return feature_matrix


def accuracy(preds, targets):
    """
    Given length-N vectors containing predicted and target labels,
    returns the percentage and number of correct predictions.
    """
    return (preds == targets).mean()
