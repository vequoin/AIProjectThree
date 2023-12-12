import numpy as np
from generateDiagrams import diagram

# ------------------------------
# Data Preparation Functions
# ------------------------------

def filter_and_prepare_dangerous_diagrams(dataset, feature_engineering_function, label_encoding):
    dangerous_diagrams = [item for item in dataset if item[1] == 'Dangerous']
    features, labels = [], []
    for diagram, _, wire_color in dangerous_diagrams:
        features.append(feature_engineering_function(diagram))
        labels.append(label_encoding[wire_color])
    return np.array(features), np.array(labels)


def extract_features(diagram):
    """
    Extract features from a given diagram.

    :param diagram: A list representing the sequence of wires laid down in the diagram.
    :return: Extracted features as a list.
    """
    color_encoding = {'Red': 0, 'Blue': 1, 'Yellow': 2, 'Green': 3}
    features = []

    # Feature 1: Identify the third wire laid down
    third_wire = color_encoding[diagram[2]]
    features.append(third_wire)

    # Add other features based on your analysis and requirements
    # For example, count intersections, color density, etc.

    return features

def prepare_data(diagram):
    """
    Prepare the data by flattening the diagram and adding the red-before-yellow feature.
    """
    # Flatten the diagram
    flattened = diagram.reshape(-1)

    pass

# ------------------------------
# Softmax Regression Functions
# ------------------------------

def softmax(scores):
    exp_scores = np.exp(scores - np.max(scores))
    return exp_scores / np.sum(exp_scores, axis=0)

def compute_logits(features, weights, biases):
    logits = np.array([np.dot(weights[color], features) + biases[color] for color in ['R', 'G', 'Y', 'B']])
    return logits

def predict(features, weights, biases):
    logits = compute_logits(features, weights, biases)
    return softmax(logits)

def cross_entropy_loss(y_true, y_pred):
    y_true_one_hot = np.zeros_like(y_pred)
    y_true_one_hot[y_true] = 1
    loss = -np.sum(y_true_one_hot * np.log(y_pred + 1e-7))  # Adding epsilon to prevent log(0)
    return loss


def compute_gradients(x, y_true, probabilities, weights, biases):
    # x: Input feature vector
    # y_true: Integer representing the true class label
    # probabilities: Predicted probabilities for each class
    # weights, biases: Current model parameters

    gradients = {'weights': {}, 'biases': {}}

    # Convert y_true to one-hot encoded vector
    y_true_one_hot = np.zeros_like(probabilities)
    y_true_one_hot[y_true] = 1

    # Calculate the gradient of the loss w.r.t. each weight and bias
    for color in ['R', 'G', 'Y', 'B']:
        error = probabilities - y_true_one_hot
        gradients['weights'][color] = np.outer(error, x)  # Gradient w.r.t. weights
        gradients['biases'][color] = error  # Gradient w.r.t. biases

    return gradients


def update_parameters(weights, biases, gradients, learning_rate):
    for color in ['R', 'G', 'Y', 'B']:
        weights[color] -= learning_rate * gradients['weights'][color]
        biases[color] -= learning_rate * gradients['biases'][color]
    return weights, biases

# ------------------------------
# Training and Evaluation
# ------------------------------

def train_model(X_train, y_train, num_epochs, learning_rate):
    # Initialize weights and biases
    weights = {'R': np.random.rand(X_train.shape[1]), 'G': np.random.rand(X_train.shape[1]),
               'Y': np.random.rand(X_train.shape[1]), 'B': np.random.rand(X_train.shape[1])}
    biases = {'R': 0, 'G': 0, 'Y': 0, 'B': 0}

    for epoch in range(num_epochs):
        epoch_losses = []
        for x, y in zip(X_train, y_train):
            probabilities = predict(x, weights, biases)
            loss = cross_entropy_loss(y, probabilities)
            epoch_losses.append(loss)
            gradients = compute_gradients(x, y, probabilities, weights, biases)
            weights, biases = update_parameters(weights, biases, gradients, learning_rate)
            
        average_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch}: Average Loss = {average_loss}")
        # Optional: Print loss every N epochs or evaluate on a validation set

    return weights, biases

def evaluate_model(X_test, y_test, weights, biases):
    correct_predictions = 0
    for x, y in zip(X_test, y_test):
        probabilities = predict(x, weights, biases)
        predicted_class = np.argmax(probabilities)
        if predicted_class == y:
            correct_predictions += 1
    accuracy = correct_predictions / len(X_test)
    return accuracy

# ------------------------------
# Main Execution
# ------------------------------

# Load dataset
dataset = ... # Load your dataset here

# Encode labels
label_encoding = {'Red': 0, 'Blue': 1, 'Yellow': 2, 'Green': 3}

# Prepare data
X, y = filter_and_prepare_dangerous_diagrams(dataset, prepare_data, label_encoding)

# Split into training and testing sets
# (Assuming you have a function to split the dataset)
X_train, X_test, y_train, y_test = split_dataset(X, y)

# Train the model
weights, biases = train_model(X_train, y_train, num_epochs=1000, learning_rate=0.01)

# Evaluate the model
accuracy = evaluate_model(X_test, y_test, weights, biases)
print(f"Model Accuracy: {accuracy}")
