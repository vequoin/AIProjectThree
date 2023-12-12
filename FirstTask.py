from generateDiagrams import diagram
import numpy as np

colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 0)]

def count_color_intersections(diagram, color1, color2):
    """
    Counts intersections between two colors in the diagram.
    :param diagram: The wiring diagram (20x20x3 array).
    :param color1: The RGB tuple of the first color.
    :param color2: The RGB tuple of the second color.
    :return: The count of intersections between color1 and color2.
    """
    mask1 = np.all(diagram == color1, axis=2)
    mask2 = np.all(diagram == color2, axis=2)
    intersections = np.sum(mask1 & mask2)
    return intersections


def create_sequence_based_features(color_sequence):

    last_wire_yellow = int(color_sequence[-1] == (255, 255, 0))
    first_wire_red = int(color_sequence[0] == (255, 0, 0))
    first_wire_yellow = int(color_sequence[0] == (255,255,0))
    last_wire_red = int(color_sequence[0] == (255,0,0))

    # Other sequence-based features can be added here

    return [last_wire_yellow, first_wire_red, first_wire_yellow, last_wire_red]  


def create_color_order_features(color_sequence, colors):
    # Initialize features
    features = []
    for color in colors:
        # Check if the color is last in the sequence
        is_last = int(color_sequence[-1] == color)
        features.append(is_last)
    
    # Add other color order related features as needed
    # Example: red_before_blue = int(color_sequence.index((255, 0, 0)) < color_sequence.index((0, 0, 255)))
    # features.append(red_before_blue)

    return features


def create_wire_order_feature(diagram):
    """
    Create a feature based on whether a red wire is laid before a yellow wire.
    The feature is 1 if a red wire comes before a yellow wire in the laying sequence, and 0 otherwise.
    """
    # Define the RGB values for red and yellow
    red = (255, 0, 0)
    yellow = (255, 255, 0)

    # Create a flattened sequence of the diagram
    flattened_diagram = diagram.reshape(-1, 3)

    # Find the first occurrence of red and yellow wires
    first_red = np.where(np.all(flattened_diagram == red, axis=1))[0]
    first_yellow = np.where(np.all(flattened_diagram == yellow, axis=1))[0]

    # Check if any red wire comes before the first yellow wire
    red_before_yellow = np.any(first_red < first_yellow.min()) if first_yellow.size > 0 else False

    return int(red_before_yellow)

def prepare_data(diagram, color_sequence):
    """
    Prepare the data by flattening the diagram and adding the red-before-yellow feature.
    """
     # Flatten the diagram
    flattened = diagram.reshape(-1)

    # Create the red-before-yellow feature
    red_before_yellow_feature = create_wire_order_feature(diagram)
    
     # Create sequence-based features
    sequence_features = create_sequence_based_features(color_sequence)

    # Ensure that sequence_features is a numpy array
    sequence_features_array = np.array(sequence_features, dtype=float)
    
    # New intersection feature
    red_yellow_intersections = count_color_intersections(diagram, (255, 0, 0), (255, 255, 0))

    # Combine flattened image and all features into a single flat array
    combined_features = np.concatenate([flattened,[red_before_yellow_feature],sequence_features_array,[red_yellow_intersections]])
    
    return combined_features

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#def compute_loss(y, y_hat):
    #return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


def compute_loss(y, y_hat):
    epsilon = 1e-7
    return -np.mean(y * np.log(y_hat + epsilon) + (1 - y) * np.log(1 - y_hat + epsilon))


def train_logistic_regression(X, y, epochs, learning_rate):
    # Initialize weights and bias
    weights = np.zeros(X.shape[1])
    bias = 0

    for epoch in range(epochs):
        for i in range(len(X)):
            # Compute the model's prediction
            y_hat = sigmoid(np.dot(X[i], weights) + bias)
            #print(f"y_hat is: {y_hat}")

            # Update weights and bias
            weights -= learning_rate * (y_hat - y[i]) * X[i]
            bias -= learning_rate * (y_hat - y[i])

        # Optionally, print loss every N epochs
        if epoch % 100 == 0:
            loss = compute_loss(y, sigmoid(np.dot(X, weights) + bias))
            print(f"Epoch {epoch}, Loss: {loss}")

    return weights, bias


def compute_loss_with_regularization(y, y_hat, weights, lambda_reg):
    # Original loss
    epsilon = 1e-7
    loss = -np.mean(y * np.log(y_hat + epsilon) + (1 - y) * np.log(1 - y_hat + epsilon))
    # L2 Regularization term
    l2_penalty = lambda_reg * np.sum(weights**2)
    return loss + l2_penalty

def train_logistic_regression_with_regularization(X, y, epochs, learning_rate, lambda_reg):
    weights = np.zeros(X.shape[1])
    bias = 0

    for epoch in range(epochs):
        for i in range(len(X)):
            y_hat = sigmoid(np.dot(X[i], weights) + bias)
            # Update weights and bias with regularization
            weights -= learning_rate * ((y_hat - y[i]) * X[i] + 2 * lambda_reg * weights)
            bias -= learning_rate * (y_hat - y[i])

        if epoch % 100 == 0:
            loss = compute_loss_with_regularization(y, sigmoid(np.dot(X, weights) + bias), weights, lambda_reg)
            print(f"Epoch {epoch}, Loss: {loss}")

    return weights, bias



def predict(X, weights, bias):
    return sigmoid(np.dot(X, weights) + bias) >= 0.5

def evaluate_model(X_test, y_test, weights, bias):
    predictions = predict(X_test, weights, bias)
    accuracy = np.mean(predictions == y_test)
    return accuracy



'''def normalize_features(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std'''


def normalize_features(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    # Avoid division by zero
    std_replaced = np.where(std == 0, 1, std)

    return (X - mean) / std_replaced


def predict_with_untrained_model(X, weights, bias):
    return sigmoid(np.dot(X, weights) + bias) >= 0.5


untrainedSet_size = 500
untrainedset = [diagram() for _ in range(untrainedSet_size)]

X_test_before = np.array([prepare_data(diagram[0], diagram[2]) for diagram in untrainedset])
x_test_before_normalized = normalize_features(X_test_before)
y_test_before = np.array([int(diagram[1]) for diagram in untrainedset])
# Initialize weights and bias to zero (or small random numbers)
untrained_weights = np.zeros(X_test_before.shape[1])  # or np.random.rand(X_test.shape[1]) * small_number
untrained_bias = 0  # or a small random number

# Make predictions
untrained_predictions = predict_with_untrained_model(x_test_before_normalized, untrained_weights, untrained_bias)

untrained_accuracy = np.mean(untrained_predictions == y_test_before)
print(f"Model Accuracy before training: {untrained_accuracy}")


# Training
dataset_size = 1500
dataset = [diagram() for _ in range(dataset_size)]

X = np.array([prepare_data(diagram[0], diagram[2]) for diagram in dataset])
y = np.array([int(diagram[1]) for diagram in dataset])
X_normalized = normalize_features(X)

# Shuffle the training dataset
indices = np.arange(len(X))
np.random.shuffle(indices)
X_shuffled = X_normalized[indices]
y_shuffled = y[indices]

# Train the model
weights, bias = train_logistic_regression_with_regularization(X_shuffled, y_shuffled, epochs=1000, learning_rate=0.01,lambda_reg=.00001)

# Testing
test_size = 1000
testset = [diagram() for _ in range(test_size)]

X_test = np.array([prepare_data(diagram[0],diagram[2]) for diagram in testset])
y_test = np.array([int(diagram[1]) for diagram in testset])

x_test_normalized = normalize_features(X_test) 

# Evaluate the model
accuracy = evaluate_model(x_test_normalized, y_test, weights, bias)
print(f"Model Accuracy after training: {accuracy}")

