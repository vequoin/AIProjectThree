import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt

def diagram():
    grid_size = 20
    layout = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
    # Initialize an ordered list of colors
    colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 0)]  # Red, Blue, Yellow, Green in RGB
    random.shuffle(colors)  # Shuffle the colors to randomize their order
    used_rows = set()
    used_columns = set()
    color_sequence = []
    is_dangerous = False

    for i in range(4):
        color = colors[i]
        # Alternate between row and column
        if i % 2 == 0:  # Row
            row = random.choice([r for r in range(grid_size) if r not in used_rows])
            used_rows.add(row)
            layout[row, :, :] = color
        else:  # Column
            column = random.choice([c for c in range(grid_size) if c not in used_columns])
            used_columns.add(column)
            layout[:, column, :] = color
        
        color_sequence.append(color)
        # Check if a red wire is laid before a yellow wire at any point in the sequence
        if (255, 0, 0) in color_sequence and (255, 255, 0) in color_sequence:
            # And only then, we compare their indices to set is_dangerous
            if color_sequence.index((255, 0, 0)) < color_sequence.index((255, 255, 0)):
                is_dangerous = True
    #print(color_sequence, is_dangerous)
    return layout, is_dangerous,color_sequence


def save_diagrams(num_diagrams=1000):
    """
    Generate and save a specified number of diagrams.
    """
    dataset = []
    for _ in range(num_diagrams):
        diagram, is_dangerous = diagram()
        dataset.append((diagram, 'Dangerous' if is_dangerous else 'Safe'))

    return dataset
