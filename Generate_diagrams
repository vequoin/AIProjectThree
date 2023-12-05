import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import random

def generate_wire_layout(grid_size=20):
    """
    Generate a random wire layout for a 20x20 grid.
    Each wire is represented by a row or column, randomly chosen.
    Colors are assigned based on specific rules.
    """
    layout = np.zeros((grid_size, grid_size), dtype=int)
    colors = ['Red', 'Blue', 'Yellow', 'Green']
    wire_sequence = []

    # Randomly lay down wires
    for _ in range(random.randint(5, 10)):  # Random number of wires
        wire_color = random.choice(colors)
        wire_type = random.choice(['row', 'column'])
        wire_position = random.randint(0, grid_size - 1)

        if wire_type == 'row':
            layout[wire_position, :] = colors.index(wire_color) + 1
        else:
            layout[:, wire_position] = colors.index(wire_color) + 1
        
        wire_sequence.append((wire_color, wire_type, wire_position))

    return layout, wire_sequence

def determine_safety(wire_sequence):
    """
    Determine if the wiring is safe or dangerous.
    Dangerous if a Red wire is laid before a Yellow wire.
    """
    red_placed = False
    for color, _, _ in wire_sequence:
        if color == 'Red':
            red_placed = True
        elif color == 'Yellow' and red_placed:
            return 'Dangerous'
    return 'Safe'

def convert_to_image(layout, save_path=None):
    """
    Convert the numeric layout to an image with colored wires.
    """
    color_map = {0: (255, 255, 255),  # White for background
                 1: (255, 0, 0),      # Red
                 2: (0, 0, 255),      # Blue
                 3: (255, 255, 0),    # Yellow
                 4: (0, 255, 0)}      # Green

    # Create an RGB image from the layout
    image_data = np.zeros((layout.shape[0], layout.shape[1], 3), dtype=np.uint8)
    for i in range(layout.shape[0]):
        for j in range(layout.shape[1]):
            image_data[i, j] = color_map[layout[i, j]]

    image = Image.fromarray(image_data, 'RGB')
    if save_path:
        image.save(save_path)
    return image

# Example of generating a dataset
num_diagrams = 5  # Number of diagrams to generate
dataset = []

for _ in range(num_diagrams):
    layout, wire_sequence = generate_wire_layout()
    safety_label = determine_safety(wire_sequence)
    image = convert_to_image(layout)
    dataset.append((layout, wire_sequence, safety_label, image))

# Displaying the first few diagrams as an example
for i, (_, _, safety_label, image) in enumerate(dataset[:3]):
    plt.figure(figsize=(2, 2))
    plt.imshow(image)
    plt.title(f"Diagram {i+1} - {safety_label}")
    plt.axis('off')
plt.show()
