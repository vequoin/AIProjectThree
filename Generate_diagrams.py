import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt

def generate_diagram():
    """
    Generate a single 20x20 wiring diagram with the specified rules:
    - 4 wires in total, each of a different color.
    - Alternating between rows and columns.
    - The diagram is dangerous if a red wire is laid before a yellow wire.
    """
    grid_size = 20
    layout = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
    colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 0)]  # Red, Blue, Yellow, Green in RGB
    used_rows = set()
    used_columns = set()
    color_sequence = []
    is_dangerous = False

    for i in range(4):
        # Alternate between row and column
        if i % 2 == 0:  # Row
            row = random.choice([r for r in range(grid_size) if r not in used_rows])
            used_rows.add(row)
            layout[row, :, :] = colors[i]
        else:  # Column
            column = random.choice([c for c in range(grid_size) if c not in used_columns])
            used_columns.add(column)
            layout[:, column, :] = colors[i]

        color_sequence.append(colors[i])

        # Check if the diagram becomes dangerous
        if colors[i] == (255, 0, 0):  # Red wire
            is_dangerous = (255, 255, 0) in color_sequence  # Check if Yellow is already placed

    return layout, is_dangerous

def save_diagrams(num_diagrams=1000):
    """
    Generate and save a specified number of diagrams.
    """
    dataset = []
    for _ in range(num_diagrams):
        diagram, is_dangerous = generate_diagram()
        dataset.append((diagram, 'Dangerous' if is_dangerous else 'Safe'))

    return dataset

# Generate the dataset
dataset = save_diagrams(1000)

# Displaying the first few diagrams as an example
for i, (diagram, label) in enumerate(dataset[:3]):
    image = Image.fromarray(diagram, 'RGB')
    plt.figure(figsize=(2, 2))
    plt.imshow(image)
    plt.title(f"Diagram {i+1}: {label}")
    plt.axis('off')
plt.show()
