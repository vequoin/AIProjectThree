import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from generateDiagrams import generate_dangerous
from generateDiagrams import diagram
from SecondTask import split_dataset
from scipy.stats import entropy  # For entropy calculation
import numpy as np

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# ------------------------------
# Data Preparation Functions
# ------------------------------



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
    
dataset_size = 1500   
dataset = generate_dangerous(dataset_size)
features, labels = [], []

 # Encode labels
label_encoding = {'R': 0, 'G': 1, 'Y': 2, 'B': 3}

for diagram_data in dataset:
    diagram, is_dangerous, color_sequence = diagram_data
    if is_dangerous:
        prepared_data = prepare_data(diagram, color_sequence)  
        color_to_cut = color_sequence[2]  
        wire_to_cut = rgb_to_color_name[color_to_cut]
        features.append(prepared_data)
        labels.append(label_encoding[wire_to_cut])

# Convert to PyTorch tensors
features_array = np.array(features)  # Convert list of numpy arrays to a single numpy array
tensor_x = torch.Tensor(features_array)  # Efficient conversion to tensor
tensor_y = torch.LongTensor(labels)

input_size = features_array.shape[1]

# Split into training and testing sets
X_train, X_test, y_train, y_test = split_dataset(tensor_x, tensor_y)
    
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


hidden_size = 50  # Example size
num_classes = 4  # Assuming 4 different classes

model = SimpleNet(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 50

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    for data, labels in train_loader:
        outputs = model(data)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test accuracy calculation
model.eval()  # Set model to evaluation mode
with torch.no_grad():
    correct, total = 0, 0
    for data, labels in test_loader:
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    test_accuracy = 100 * correct / total
    print(f'Test Accuracy: {test_accuracy:.2f}%')


