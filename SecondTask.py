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
    (255, 0, 0): 'R',
    (0, 255, 0): 'G',
    (255, 255, 0): 'Y',
     (0, 0, 255): 'B',
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

grid_width = 20  # Width of the grid
num_channels = 3  # Number of channels (RGB)

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


def get_third_wire_row_col(diagram, third_wire_color):
    
    i_cols = []
    i_rows = []
    def majority_color(line):
        # Count the occurrence of each color in the line and return the most common one
        colors, count = np.unique(line.reshape(-1, 3), axis=0, return_counts=True)
        return colors[count.argmax()]

    for i in range(diagram.shape[0]):  # Loop through rows
        row = diagram[i, :]
        if np.array_equal(majority_color(row), third_wire_color):
            return row

    for j in range(diagram.shape[1]):  # Loop through columns
        col = diagram[:, j]
        if np.array_equal(majority_color(col), third_wire_color):
            return col

    return None

def get_third_wire_positions(diagram, third_wire_color):
    def majority_color(line):
        colors, count = np.unique(line.reshape(-1, 3), axis=0, return_counts=True)
        return colors[count.argmax()]

    third_wire_positions = []

    # Check rows
    for i in range(diagram.shape[0]):  # Loop through rows
        row = diagram[i, :]
        if np.array_equal(majority_color(row), third_wire_color):
            third_wire_positions.extend([(i, j) for j in range(diagram.shape[1])])
            
    # Check columns
    for j in range(diagram.shape[1]):  # Loop through columns
        col = diagram[:, j]
        if np.array_equal(majority_color(col), third_wire_color):
            third_wire_positions.extend([(i, j) for i in range(diagram.shape[0])])

    return third_wire_positions


def encode_wire_positions(color_sequence, color_encoding):
    return [color_encoding[color] for color in color_sequence]



def calculate_relative_positions(diagram, color_sequence, color_encoding):
    third_wire_color = color_sequence[2]
    third_wire_code = color_encoding[third_wire_color]
    third_color_list = [color for color in third_wire_color]
    
    relative_positions = []
    for row in range(diagram.shape[0]):
        for col in range(diagram.shape[1]):
            # Compare each pixel's color to the third wire's color
            if np.all(diagram[row, col] == third_color_list):
                relative_positions.append((row, col))

    #print(relative_positions)
    flattened_relative_positions = [coord for pos in relative_positions for coord in pos]
    return flattened_relative_positions


def get_third_wire_intersections(diagram, third_wire_color):
    def majority_color(line):
        colors, count = np.unique(line.reshape(-1, 3), axis=0, return_counts=True)
        return colors[count.argmax()]

    third_wire_intersections = []

    # Check rows for intersections
    for i in range(diagram.shape[0]):  # Loop through rows
        row = diagram[i, :]
        if np.array_equal(majority_color(row), third_wire_color):
            for j in range(diagram.shape[1]):  # Check each cell in the row
                if not np.array_equal(row[j], third_wire_color):  # Intersection if the cell color is different
                    third_wire_intersections.append((i, j))
            
    # Check columns for intersections
    for j in range(diagram.shape[1]):  # Loop through columns
        col = diagram[:, j]
        if np.array_equal(majority_color(col), third_wire_color):
            for i in range(diagram.shape[0]):  # Check each cell in the column
                if not np.array_equal(col[i], third_wire_color):  # Intersection if the cell color is different
                    third_wire_intersections.append((i, j))

    return third_wire_intersections


def get_third_wire_overlaps(diagram, color_sequence):
    second_wire_color = color_sequence[1]
    third_wire_color = color_sequence[2]

    # Function to find the majority color in a row/column
    def majority_color(line):
        colors, count = np.unique(line.reshape(-1, 3), axis=0, return_counts=True)
        return colors[count.argmax()]

    # Determine if the second wire is in a row or column
    for i in range(diagram.shape[0]):
        if np.array_equal(majority_color(diagram[i, :]), second_wire_color):
            # The second wire is in this row, check for third wire color in this row
            for j in range(diagram.shape[1]):
                if np.array_equal(diagram[i, j], third_wire_color):
                    return (i, j)
            break

        if np.array_equal(majority_color(diagram[:, i]), second_wire_color):
            # The second wire is in this column, check for third wire color in this column
            for j in range(diagram.shape[0]):
                if np.array_equal(diagram[j, i], third_wire_color):
                    return (j, i)
            break

    return None



def extract_features(diagram, color_sequence):
    """
    Extract features from a given diagram.

    :param diagram: A list representing the sequence of wires laid down in the diagram.
    :return: Extracted features as a list.
    """
    color_encoding = {'R': 0, 'G': 1, 'Y': 2, 'B': 3}
    color_arr = ['R', 'G', 'Y', 'B']
    features = []
    
    color_encodingg = {
    (255, 0, 0): 'R',      # Red
    (0, 255, 0): 'G',    # Green
    (0, 0, 255): 'Y',     # Blue
    (255, 255, 0): 'B', # Yellow
    # Add any other colors that are used in your diagrams
    }

    # Get relative positions and patterns
    #relative_positions = calculate_relative_positions(diagram, color_sequence, color_encodingg)
    #print(relative_positions)
    

    # Combine all features
    #features.extend(relative_positions)

    # Feature 1: Identify the third wire laid down
    third_wire_color_name = rgb_to_color_name.get(color_sequence[2])
    third_wire = color_encoding.get(third_wire_color_name)
    color_wire = [item for item in color_sequence[2]]
    features.append(third_wire)
    # features.extend(color_wire)
    
    third_wire_col = get_third_wire_row_col(diagram, color_sequence[2])
    if third_wire_col is not None:
        flattened_col = third_wire_col.flatten()
        #features.extend(flattened_col)
    else:
        pass
        # Handle the None case
        #features.extend([0] * 3)
        
    
    third_wire_pos = get_third_wire_positions(diagram, color_sequence[2])
    #print(third_wire_pos)
    #third_pos = third_wire_pos.flatten()
    flattened_third_wire_pos = [item for sublist in third_wire_pos for item in sublist]
    #features.extend(flattened_third_wire_pos)
    flattened_indices = []
    for i, j in third_wire_pos:
        base_index = i * 20 * 3 + j * 3  # Calculate the base index for the flattened array
        flattened_indices.extend([base_index, base_index + 1, base_index + 2])  # Add indices for R, G, B channels
    #print(flattened_indices)
    features.extend(flattened_indices)

   
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
    flatten_seq_color = []
    for color in color_sequence:
        flatten_seq_color.extend(color)  # Assuming each 'color' is a tuple like (255, 0, 0)
    features.extend(flatten_seq_color)
    
    overlap = get_third_wire_overlaps(diagram, color_sequence)
    #print(overlap)
    flatten_overlap = flatten_coordinates(overlap)
    #print(overlap)
    features.extend(flatten_overlap)
    
    color_hot_one = []
    for color in color_arr:
        if third_wire_color_name == color:
            color_hot_one.append(1)
        else:
            color_hot_one.append(0)
            
    #print(color_hot_one)
    features.extend(color_hot_one)
    
    Intersections_arr = [2,1,1,0]
    
    features.extend(Intersections_arr)

    
    third_wire_color = color_sequence[2]
    
    third_wire_intersections = get_third_wire_intersections(diagram, third_wire_color)
    flatten_intersect = flatten_coordinates(third_wire_intersections[0])
    features.extend(flatten_intersect)
    # print(f"intersect is {flatten_intersect}")
    # print(f"overlap is {flatten_overlap}")
    #print(features)
    third_wire_intersections_count = count_intersections_for_wire(diagram, third_wire_color_name)
    #print(third_wire_intersections)
    #bb = input("...")

    features.append(third_wire_intersections_count)
    # print(features)
    # bb = input("...")
    return features


def flatten_coordinates(coordinates):
    flattened_indices = []
    i = coordinates[0]
    j = coordinates[1]
    index = (i * grid_width + j) * num_channels
    flattened_indices.extend([index, index + 1, index + 2])  # Append indices for R, G, B channels
    return flattened_indices


def count_intersections_for_wire(diagram, wire_color):
    color_encoding = {'R': 0, 'G': 1, 'Y': 2, 'B': 3}
    wire_code = color_encoding[wire_color]

    intersection_count = 0
    for row in diagram:
        for cell in row:
            #print(f"cell is {cell}")
            #print(f"wire code is: {wire_color}")
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
    #print(combined_features)
    
    return combined_features
    

# ------------------------------
# Softmax Regression Functions
# ------------------------------

def softmax(scores):
    exp_scores = np.exp(scores - np.max(scores))
    #print(exp_scores)
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
        print(f"Epoch {epoch}: Average Loss = {average_loss}")

    return weights, biases


def evaluate_model(X_test, y_test, weights, biases, loss_only=False):
    correct_predictions = 0
    total_loss = 0
    for x, y in zip(X_test, y_test):
        probabilities = predict(x, weights, biases)
        #print(probabilities)
        if not loss_only:
            predicted_class = np.argmax(probabilities)
            #print(predicted_class)
            if predicted_class == y:
                correct_predictions += 1
            #print(y)
            #taking_break = input("Enter... ")
        loss = cross_entropy_loss(y, probabilities,weights, lambda_reg=0.00001)
        total_loss += loss
        # print(loss)
        # print(total_loss)

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

num_runs = 5
for i in range(num_runs):
    # Load dataset
    dataset_size = 500
    dataset = generate_dangerous(dataset_size) # Load your dataset here

    # Encode labels
    label_encoding = {'R': 0, 'G': 1, 'Y': 2, 'B': 3}

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
   

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # Train the model
    weights, biases = train_model(X_train, y_train, num_epochs=100, learning_rate=0.001, lambda_reg=0.00001)

    # Evaluate the model
    accuracy = evaluate_model(X_test, y_test, weights, biases)
    total += accuracy
    
    print(f"Model Accuracy: {accuracy}")

print(f"Totak Accuracy: {total/num_runs}")
