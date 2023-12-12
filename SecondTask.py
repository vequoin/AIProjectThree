import numpy as np
from generateDiagrams import diagram
from generateDiagrams import generate_dangerous
import random

# ------------------------------
# Data Preparation Functions
# ------------------------------

from scipy.stats import entropy  # For entropy calculation

def compute_statistical_aggregations(diagram):
    # Initialize lists to store computed statistics for each row
    row_mean = []
    row_variance = []
    row_entropy = []

    # Iterate through each row
    for row in diagram:
        # Count color occurrences in the row
        color_counts = [np.sum(row == color) for color in [0, 1, 2, 3]]  # Assuming 0 for 'Red', 1 for 'Blue', etc.

        # Calculate statistics
        mean = np.mean(color_counts)
        variance = np.var(color_counts)
        ent = entropy(color_counts)  # You might need to adjust this based on your data format

        # Append statistics to the respective lists
        row_mean.append(mean)
        row_variance.append(variance)
        row_entropy.append(ent)

    # Return the computed statistics as a list
    return row_mean, row_variance, row_entropy



rgb_to_color_name = {
    (255, 0, 0): 'Red',
    (0, 255, 0): 'Green',
    (255, 255, 0): 'Yellow',
     (0, 0, 255): 'Blue',
}


def get_third_wire_position(diagram, color_sequence):
    third_color = color_sequence[2]
    row, col = None, None
    # Check if the third wire is laid in a row or a column
    for i in range(20):  # Assuming a 20x20 grid
        if np.all(diagram[i, :, :] == third_color, axis=1).all():  # Third wire is in this row
            row = i
            break
        if np.all(diagram[:, i, :] == third_color, axis=0).all():  # Third wire is in this column
            col = i
            break
    
    return row, col


def count_color_frequencies(diagram):
    # Initialize lists to store color frequencies for each color
    red_count = []
    blue_count = []
    yellow_count = []
    green_count = []

    # Define numeric values for colors (0 for 'Red', 1 for 'Green', etc.)
    color_values = [0, 1, 2, 3]

    # Iterate through each color value
    for color_value in color_values:
        # Count occurrences of the color value in each row
        color_counts = np.sum(diagram == color_value, axis=1)

        # Append color frequencies to the respective lists
        if color_value == 0:
            red_count.extend(color_counts)
        elif color_value == 1:
            green_count.extend(color_counts)
        elif color_value == 2:
            yellow_count.extend(color_counts)
        elif color_value == 3:
            blue_count.extend(color_counts)

    # Return the computed color frequencies as lists
    return red_count, green_count, yellow_count, blue_count


def extract_features(diagram, color_sequence):
    """
    Extract features from a given diagram.

    :param diagram: A list representing the sequence of wires laid down in the diagram.
    :return: Extracted features as a list.
    """
    color_encoding = {'Red': 0, 'Green': 1, 'Yellow': 2, 'Blue': 3}
    color_arr = ['Red', 'Green', 'Yellow', 'Blue']
    features = []

    # Feature 1: Identify the third wire laid down
    third_wire_color_name = rgb_to_color_name.get(color_sequence[2])
    third_wire = color_encoding.get(third_wire_color_name)
    #features.append(third_wire)
    
    row_mean, row_variance, row_entropy = compute_statistical_aggregations(diagram)
    red_count, blue_count, yellow_count, green_count = count_color_frequencies(diagram)
    
    features.extend([np.mean(row_mean), np.mean(row_variance), np.mean(row_entropy),
                     np.mean(red_count), np.mean(blue_count), np.mean(yellow_count), np.mean(green_count)])
    
    first_wire = rgb_to_color_name.get(color_sequence[0])
    second_wire = rgb_to_color_name.get(color_sequence[1])
    third_wire_seq = rgb_to_color_name.get(color_sequence[2])
    fourth_wire = rgb_to_color_name.get(color_sequence[3])
    first_wire_num = color_encoding.get(first_wire)
    second_wire_num = color_encoding.get(second_wire)
    third_wire_num = color_encoding.get(third_wire_seq)
    fourth_wire_num = color_encoding.get(fourth_wire)
    wire_laid_sequence = [first_wire_num, second_wire_num, third_wire_num, fourth_wire_num]
    features.extend(wire_laid_sequence)
    
    
    color_hot_one = []
    for color in color_arr:
        if third_wire_color_name == color:
            color_hot_one.append(1)
        else:
            color_hot_one.append(0)
            
    #print(color_hot_one)
    features.extend(color_hot_one)
    
    #Feature 2: Intersection count for each wire
    intersection_counts = [2, 1, 1, 0]  # Fixed pattern based on the laying order
    #features.extend(intersection_counts)
    
    for color in ['Red', 'Green', 'Yellow', 'Blue']:
       pos = next((i for i, x in enumerate(color_sequence[:3]) if x == color), -1)
       #features.append(pos)

    #Spatial Features
    row, col = get_third_wire_position(diagram, color_sequence)
    features.extend([row if row is not None else -1, col if col is not None else -1])
    
    third_wire_color = color_sequence[2]
    
    third_wire_intersections = count_intersections_for_wire(diagram, third_wire_color_name)
    features.append(third_wire_intersections)

    return features

def count_intersections_for_wire(diagram, wire_color):
    color_encoding = {'Red': 0, 'Green': 1, 'Yellow': 2, 'Blue': 3}
    wire_code = color_encoding[wire_color]

    intersection_count = 0
    for row in diagram:
        for cell in row:
            if (cell == wire_code).all():
                intersection_count += 1

    # Subtract the length of the wire itself to get only intersections
    return intersection_count - len(diagram)


def encode_wire_laying_order(color_sequence):
    return [rgb_to_color_name[color] for color in color_sequence]


def prepare_data(diagram, color_sequence):
    """
    Prepare the data by flattening the diagram and adding the red-before-yellow feature.
    """
    # Flatten the diagram
    flattened = diagram.reshape(-1)
    
    features = extract_features(diagram, color_sequence)
    
    features_array = np.array(features)
    
    combined_features = np.concatenate([flattened, features_array])
    
    return combined_features
    

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


def cross_entropy_loss(y_true, y_pred, weights, lambda_reg):
    y_true_one_hot = np.zeros_like(y_pred)
    y_true_one_hot[y_true] = 1
    cross_entropy = -np.sum(y_true_one_hot * np.log(y_pred + 1e-7))  # Adding epsilon to prevent log(0)
    
    l2_penalty = sum(np.sum(w**2) for w in weights.values())
    regularization_loss = lambda_reg / 2 * l2_penalty

    return cross_entropy + regularization_loss



'''def compute_gradients(x, y_true, probabilities, weights, biases):
    # x: Input feature vectors
    # y_true: Integer representing the true class label
    # probabilities: Predicted probabilities for each class
    # weights, biases: Current model parameters

    gradients = {'weights': {}, 'biases': {}}

    # Convert y_true to one-hot encoded vector
    y_true_one_hot = np.zeros_like(probabilities)
    y_true_one_hot[y_true] = 1
    #print(gradients)
    # Calculate the gradient of the loss w.r.t. each weight and bias
    error = probabilities - y_true_one_hot
    #print(error)
    colors = ['R', 'G', 'Y', 'B']
    for i in  range(len(colors)):
        gradients['weights'][colors[i]] = np.outer(error[colors[i]], x)  # Gradient w.r.t. weights
        gradients['biases'][colors[i]] = error[colors[i]]  # Gradient w.r.t. biases
        
    for i, color in enumerate(['R', 'G', 'Y', 'B']):
        gradients['weights'][color] = error[i]* x  # Gradient w.r.t. weights
        gradients['biases'][color] = error[i]

    return gradients


    def update_parameters(weights, biases, gradients, learning_rate):
        for color in ['R', 'G', 'Y', 'B']:
            print("Weights shape:", weights[color].shape)
            print("Gradient shape:", gradients['weights'][color].shape)
            weights[color] -= learning_rate * gradients['weights'][color]
            biases[color] -= learning_rate * gradients['biases'][color]
        return weights, biases'''
    

def compute_gradients(x, y_true, probabilities, weights, biases, lambda_reg):
    gradients = {'weights': {}, 'biases': {}}
    y_true_one_hot = np.zeros_like(probabilities)
    y_true_one_hot[y_true] = 1
    error = probabilities - y_true_one_hot

    for i, color in enumerate(['R', 'G', 'Y', 'B']):
        gradients['weights'][color] = np.outer(error[i], x) + lambda_reg * weights[color]  # Adding L2 regularization term
        gradients['biases'][color] = error[i]

    return gradients



def update_parameters(weights, biases, gradients, learning_rate):
    for color in ['R', 'G', 'Y', 'B']:
        # Ensure the gradient is reshaped or sliced correctly to match the weights
        gradient = gradients['weights'][color].reshape(weights[color].shape)
        #print("Weights shape:", weights[color].shape)
        #print("Gradient shape:", gradients['weights'][color].shape)
        weights[color] -= learning_rate * gradient
        biases[color] -= learning_rate * gradients['biases'][color]
    return weights, biases


# ------------------------------
# Training and Evaluation
# ------------------------------

'''def train_model(X_train, y_train, num_epochs, learning_rate):
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

    return weights, biases'''
    
    
def train_model(X_train, y_train, num_epochs, learning_rate, lambda_reg):
    # Initialize weights and biases
    weights = {color: np.random.rand(X_train.shape[1]) * 0.01 for color in ['R', 'G', 'Y', 'B']}
    biases = {color: 0 for color in ['R', 'G', 'Y', 'B']}

    for epoch in range(num_epochs):
        epoch_losses = []
        for x, y in zip(X_train, y_train):
            probabilities = predict(x, weights, biases)
            loss = cross_entropy_loss(y, probabilities, weights, lambda_reg)
            epoch_losses.append(loss)
            gradients = compute_gradients(x, y, probabilities, weights, biases, lambda_reg)
            weights, biases = update_parameters(weights, biases, gradients, learning_rate)

        average_loss = np.mean(epoch_losses)
        #print(f"Epoch {epoch}: Average Loss = {average_loss}")

    return weights, biases


'''def train_model(X_train, y_train, X_val, y_val, num_epochs, learning_rate, early_stopping_rounds=None):
    # Initialize weights and biases with small random values
    weights = {color: np.random.randn(X_train.shape[1]) * 0.01 for color in ['R', 'G', 'Y', 'B']}
    biases = {color: 0 for color in ['R', 'G', 'Y', 'B']}

    best_val_loss = float('inf')
    no_improvement_count = 0

    for epoch in range(num_epochs):
        epoch_losses = []
        for x, y in zip(X_train, y_train):
            probabilities = predict(x, weights, biases)
            loss = cross_entropy_loss(y, probabilities)
            epoch_losses.append(loss)
            gradients = compute_gradients(x, y, probabilities, weights, biases)
            weights, biases = update_parameters(weights, biases, gradients, learning_rate)

        average_loss = np.mean(epoch_losses)
        val_loss = evaluate_model(X_val, y_val, weights, biases, loss_only=True)
        print(f"Epoch {epoch}: Average Loss = {average_loss}, Validation Loss = {val_loss}")

        if early_stopping_rounds:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count >= early_stopping_rounds:
                    print("Early stopping triggered.")
                    break

    return weights, biases'''

'''def evaluate_model(X_test, y_test, weights, biases):
    correct_predictions = 0
    for x, y in zip(X_test, y_test):
        probabilities = predict(x, weights, biases)
        predicted_class = np.argmax(probabilities)
        if predicted_class == y:
            correct_predictions += 1
    accuracy = correct_predictions / len(X_test)
    return accuracy'''

def evaluate_model(X_test, y_test, weights, biases, loss_only=False):
    correct_predictions = 0
    total_loss = 0
    for x, y in zip(X_test, y_test):
        probabilities = predict(x, weights, biases)
        if not loss_only:
            predicted_class = np.argmax(probabilities)
            if predicted_class == y:
                correct_predictions += 1
        loss = cross_entropy_loss(y, probabilities,weights, lambda_reg=0.00001)
        total_loss += loss

    if loss_only:
        return total_loss / len(X_test)
    else:
        accuracy = correct_predictions / len(X_test)
        return accuracy


# ------------------------------
# Main Execution
# ------------------------------

total = 0


def split_dataset(X,y, train_ratio=.8):
    """
    Splits the dataset into training and testing sets based on the specified ratio.

    :param X: Array of features.
    :param y: Array of labels.
    :param train_ratio: Ratio of the dataset to be used for training (default is 0.8).
    :return: X_train, X_test, y_train, y_test
    """
    # Calculate the number of training examples
    num_train_examples = int(len(X) * train_ratio)

    # Shuffle the dataset
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    # Split the dataset
    X_train = X_shuffled[:num_train_examples]
    X_test = X_shuffled[num_train_examples:]
    y_train = y_shuffled[:num_train_examples]
    y_test = y_shuffled[num_train_examples:]

    return X_train, X_test, y_train, y_test


'''def split_dataset(X, y, train_ratio=0.8, val_ratio=0.1):
    """
    Splits the dataset into training, validation, and testing sets based on the specified ratios.

    :param X: Array of features.
    :param y: Array of labels.
    :param train_ratio: Ratio of the dataset to be used for training.
    :param val_ratio: Ratio of the dataset to be used for validation.
    :return: X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Calculate the number of examples for each set
    num_train_examples = int(len(X) * train_ratio)
    num_val_examples = int(len(X) * val_ratio)

    # Shuffle the dataset
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    # Split the dataset
    X_train = X_shuffled[:num_train_examples]
    y_train = y_shuffled[:num_train_examples]

    X_val = X_shuffled[num_train_examples:num_train_examples + num_val_examples]
    y_val = y_shuffled[num_train_examples:num_train_examples + num_val_examples]

    X_test = X_shuffled[num_train_examples + num_val_examples:]
    y_test = y_shuffled[num_train_examples + num_val_examples:]

    return X_train, X_val, X_test, y_train, y_val, y_test'''

for i in range(5):
    # Load dataset
    dataset_size = 500
    dataset = generate_dangerous(dataset_size) # Load your dataset here

    # Encode labels
    label_encoding = {'Red': 0, 'Green': 1, 'Yellow': 2, 'Blue': 3}

    # Prepare data
    features, labels = [], []
    for diagram_data in dataset:
        diaagram, is_dangerous, color_sequence = diagram_data
        if is_dangerous:  # Ensure that the diagram is dangerous
            prepared_data = prepare_data(diaagram, color_sequence)
            color_to_cut = color_sequence[2]  # Third wire to be cut
            wire_to_cut = rgb_to_color_name[color_to_cut]
            features.append(prepared_data)
            labels.append(label_encoding[wire_to_cut])

    X = np.array(features)
    y = np.array(labels)


    # Split into training and testing sets
    # (Assuming you have a function to split the dataset)
    '''X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y, train_ratio=0.7, val_ratio=0.1)
    weights, biases = train_model(X_train, y_train, X_val, y_val, num_epochs=100, learning_rate=0.01)'''

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # Train the model
    weights, biases = train_model(X_train, y_train, num_epochs=80, learning_rate=0.001, lambda_reg=0.000001)

    # Evaluate the model
    accuracy = evaluate_model(X_test, y_test, weights, biases)
    total += accuracy
    
    print(f"Model Accuracy: {accuracy}")

print(f"Totak Accuracy: {total/5}")
