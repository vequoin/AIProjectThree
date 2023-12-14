import numpy as np
from generateDiagrams import diagram
from generateDiagrams import generate_dangerous
import random
import time

# ------------------------------
# Data Preparation Functions
# ------------------------------

from scipy.stats import entropy  # For entropy calculation


def compute_statistical_aggregations(diagram):
    row_mean = []
    row_variance = []
    row_entropy = []

    color_values = [(255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 0, 255)]  # RGB values for Red, Green, Yellow, Blue

    for row in diagram:
        color_counts = [np.sum(np.all(row == color, axis=-1)) for color in color_values]

        mean = np.mean(color_counts)
        variance = np.var(color_counts)
        ent = entropy(color_counts, base=2)  # Using base 2 for entropy calculation

        row_mean.append(mean)
        row_variance.append(variance)
        row_entropy.append(ent)

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
    for i in range(20):  
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
    # Initialize counts for each color
    red_count, green_count, yellow_count, blue_count = 0, 0, 0, 0

    # Define the RGB values for colors
    color_values = {'Red': [255, 0, 0], 'Green': [0, 255, 0], 'Yellow': [255, 255, 0], 'Blue': [0, 0, 255]}

    # Count occurrences of each color
    for i in range(diagram.shape[0]):
        for j in range(diagram.shape[1]):
            pixel = diagram[i, j]
            if np.array_equal(pixel, color_values['Red']):
                red_count += 1
            elif np.array_equal(pixel, color_values['Green']):
                green_count += 1
            elif np.array_equal(pixel, color_values['Yellow']):
                yellow_count += 1
            elif np.array_equal(pixel, color_values['Blue']):
                blue_count += 1

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



'''def pixel_color_relationships(diagram, seq):
    grid_size = diagram.shape[0]
    color_relationships = []

    colors_of_interest = colors_of_interest = {
    'Red': (255, 0, 0),
    'Green': (0, 255, 0),
    'Yellow': (255, 255, 0),
    'Blue': (0, 0, 255)
    }


    default_value = [0, 0, 0]  # RGB representation for default value

    for i in range(grid_size):
        for j in range(grid_size):
            current_pixel = diagram[i, j]

            if any(np.array_equal(current_pixel, color) for color in colors_of_interest.values()):
                # Initialize with default values
                up_neighbor = left_neighbor = default_value

                if i > 0:  # Up
                    up_neighbor = diagram[i-1, j].tolist()
                if j > 0:  # Left
                    left_neighbor = diagram[i, j-1].tolist()

                # Add to color_relationships
                color_relationships.extend(up_neighbor + left_neighbor)
                
    #print(color_relationships)

    return color_relationships'''



def calculate_relative_positions(diagram, color_sequence):
    third_wire_color = color_sequence[2]
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
    for i in range(diagram.shape[0]): 
        row = diagram[i, :]
        if np.array_equal(majority_color(row), third_wire_color):
            for j in range(diagram.shape[1]):  # Check each cell in the row
                if not np.array_equal(row[j], third_wire_color):  # Intersection if the cell color is different
                    third_wire_intersections.append((i, j))
            
    # Check columns for intersections
    for j in range(diagram.shape[1]):  
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


def get_wire_order(diagram):
    grid_size = diagram.shape[0]
    wire_order = []
    visited_colors = set()
    
    for i in range(grid_size):
        # Check row
        row_color, row_count = dominant_color_and_count(diagram[i, :, :])
        if row_color is not None:
            wire_order.append((tuple(row_color), row_count))

        # Check column
        col_color, col_count = dominant_color_and_count(diagram[:, i, :])
        if col_color is not None:
            wire_order.append((tuple(col_color), col_count))


    return wire_order


def dominant_color_and_count(line):
    grid_size = 20
    colors, counts = np.unique(line.reshape(-1, 3), axis=0, return_counts=True)
    # Filter out the background color (0,0,0)
    filtered_indices = [index for index, color in enumerate(colors) if not np.array_equal(color, [0, 0, 0])]
    if filtered_indices:
        max_index = filtered_indices[np.argmax(counts[filtered_indices])]
        return colors[max_index], counts[max_index]
    return None, 0


def calculate_color_distribution(diagram):
    grid_size = diagram.shape[0] * diagram.shape[1]  # Total number of cells in the diagram
    color_encoding = {'Red': [255, 0, 0], 'Green': [0, 255, 0], 'Yellow': [255, 255, 0], 'Blue': [0, 0, 255]}
    color_distribution = {'Red': 0, 'Green': 0, 'Yellow': 0, 'Blue': 0}

    # Count each color
    for color, rgb in color_encoding.items():
        color_distribution[color] = np.sum(np.all(diagram == rgb, axis=-1))

    # Normalize the counts
    for color in color_distribution:
        color_distribution[color] = color_distribution[color] / grid_size

    return color_distribution


def pixel_color_relationships(diagram, seq):
    grid_size = diagram.shape[0]
    color_relationships = []

    # Define the RGB values for Red, Green, Yellow, and Blue
    colors_of_interest = {
        'Red': (255, 0, 0),
        'Green': (0, 255, 0),
        'Yellow': (255, 255, 0),
        'Blue': (0, 0, 255)
    }

    # Define a default value for pixels not of interest or without neighbors
    default_value = [0, 0, 0]  # RGB representation for default value

    for i in range(grid_size):
        for j in range(grid_size):
            current_pixel = diagram[i, j]
            
            # Initialize neighbor colors with default values
            neighbors = [default_value] * 4  # Up, Down, Left, Right

            # Check if current pixel is one of the colors of interest
            if any(np.array_equal(current_pixel, color) for color in colors_of_interest.values()):
                # Update neighbors based on actual neighbor pixels
                if i > 0:  # Up
                    neighbors[0] = diagram[i-1, j].tolist()
                if i < grid_size - 1:  # Down
                    neighbors[1] = diagram[i+1, j].tolist()
                if j > 0:  # Left
                    neighbors[2] = diagram[i, j-1].tolist()
                if j < grid_size - 1:  # Right
                    neighbors[3] = diagram[i, j+1].tolist()

            # Flatten the list of neighbors and add to color_relationships
            color_relationships.extend(sum(neighbors, []))

    return color_relationships



def count_intersections(diagram):
    grid_size = diagram.shape[0]
    color_encoding = {'Red': [255, 0, 0], 'Green': [0, 255, 0], 'Yellow': [255, 255, 0], 'Blue': [0, 0, 255]}
    intersections = {'Red': 0, 'Green': 0, 'Yellow': 0, 'Blue': 0}

    # Convert diagram to color names for easier processing
    color_diagram = np.empty((grid_size, grid_size), dtype=object)
    for color, rgb in color_encoding.items():
        color_diagram[np.all(diagram == rgb, axis=-1)] = color

    # Count intersections
    for i in range(grid_size):
        for j in range(grid_size):
            cell_color = color_diagram[i, j]
            if cell_color:
                # Check horizontal and vertical neighbors for different color
                if i < grid_size - 1 and color_diagram[i+1, j] and color_diagram[i+1, j] != cell_color:
                    intersections[cell_color] += 1
                if j < grid_size - 1 and color_diagram[i, j+1] and color_diagram[i, j+1] != cell_color:
                    intersections[cell_color] += 1

    return intersections



def flatten_intersection_features(intersections):
    return [intersections['Red'], intersections['Green'], intersections['Yellow'], intersections['Blue']]

def flatten_color_features(color_distribution):
    return [color_distribution['Red'], color_distribution['Green'], color_distribution['Yellow'], color_distribution['Blue']]


def count_color_matches(diagram):
    grid_size = diagram.shape[0]
    match_count = 0

    for i in range(grid_size):
        for j in range(grid_size):
            if i < grid_size - 1:  # Check vertical match
                if np.array_equal(diagram[i, j], diagram[i + 1, j]):
                    match_count += 1
            if j < grid_size - 1:  # Check horizontal match
                if np.array_equal(diagram[i, j], diagram[i, j + 1]):
                    match_count += 1

    return match_count


def extract_features(diagram, color_sequence):
    """
    Extract features from a given diagram.

    :param diagram: A list representing the sequence of wires laid down in the diagram.
    :return: Extracted features as a list.
    """
    color_encoding = {'R': 0, 'G': 1, 'Y': 2, 'B': 3}
    features = []
    
    color_encodingg = {
    (255, 0, 0): 'R',      # Red
    (0, 255, 0): 'G',    # Green
    (0, 0, 255): 'Y',     # Yellow
    (255, 255, 0): 'B', # Blue
    # Add any other colors that are used in your diagrams
    }

    color_distributions = calculate_color_distribution(diagram)
    color_distributions_flat = flatten_color_features(color_distributions)
    features.extend(color_distributions_flat)
    intersections = count_intersections(diagram)
    intersections_all = flatten_intersection_features(intersections)
    features.extend(intersections_all)
    
    match_colors = count_color_matches(diagram)
    features.append(match_colors)
    
            
    pixel_relationships = pixel_color_relationships(diagram, color_sequence)
    #print(pizel_relationships)
    features.extend(pixel_relationships)
    #print(features)

   
    row_mean, row_variance, row_entropy = compute_statistical_aggregations(diagram)
    red_count, blue_count, yellow_count, green_count = count_color_frequencies(diagram)
    #print(red_count, blue_count, yellow_count, green_count)
    
    features.extend([np.mean(row_mean), np.mean(row_variance), np.mean(row_entropy),
                     np.mean(red_count), np.mean(blue_count), np.mean(yellow_count), np.mean(green_count)])
    
    
    overlap = get_third_wire_overlaps(diagram, color_sequence)
    #print(overlap)
    flatten_overlap = flatten_coordinates(overlap)
    #print(overlap)
    features.extend(flatten_overlap)

    
    third_wire_color = color_sequence[2]
    
    third_wire_intersections = get_third_wire_intersections(diagram, third_wire_color)
    flatten_intersect = flatten_coordinates(third_wire_intersections[0])
    features.extend(flatten_intersect)
    # print(f"intersect is {flatten_intersect}")
    # print(f"overlap is {flatten_overlap}")
    # print(features)
    third_wire_color_name = rgb_to_color_name.get(color_sequence[2])
    third_wire_intersections_ex = count_intersections_for_wire_verus_length(diagram, third_wire_color_name)
    # #print(third_wire_intersections_count)
    # #bb = input("...")

    features.append(third_wire_intersections_ex)
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



def count_intersections_for_wire_verus_length(diagram, wire_color):
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
    #print(f"length of feature array is: {len(combined_features)}")
    #print(combined_features)
    
    return combined_features


def prepare_data_before(diagram):
    flattened = diagram.reshape(-1)
    
    return np.concatenate([flattened])
    

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

num_runs = 1
for i in range(num_runs):
    start_time = time.time()
    # Load dataset
    dataset_size = 500
    dataset = generate_dangerous(dataset_size) # Load your dataset here

    # Encode labels
    label_encoding = {'R': 0, 'G': 1, 'Y': 2, 'B': 3}

    # Prepare data
    features, labels = [], []
    for diagram_data in dataset:
        diaagram, is_dangerous, color_sequence = diagram_data
        if is_dangerous:  
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
    weights, biases = train_model(X_train, y_train, num_epochs=50, learning_rate=0.0001, lambda_reg=0.00001)

    # Evaluate the model
    accuracy = evaluate_model(X_test, y_test, weights, biases)
    total += accuracy
    
    print(f"Model Accuracy: {accuracy}")
    end_time = time.time()
    print(f"Time taken: {end_time-start_time}")

print(f"Totak Accuracy: {total/num_runs}")
