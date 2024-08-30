import numpy as np
import pandas as pd
import colorsys

# Number of samples in the dataset
num_samples = 10000  # Adjust this number for a larger dataset

def rgb_to_hsv(r, g, b):
    return colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)

def hsv_to_rgb(h, s, v):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h, s, v))

def generate_complementary_color(r, g, b):
    h, s, v = rgb_to_hsv(r, g, b)
    h_complementary = (h + 0.5) % 1.0
    return hsv_to_rgb(h_complementary, s, v)

def generate_analogous_colors(r, g, b):
    h, s, v = rgb_to_hsv(r, g, b)
    h1 = (h + 0.1) % 1.0
    h2 = (h - 0.1) % 1.0
    return hsv_to_rgb(h1, s, v), hsv_to_rgb(h2, s, v)

def generate_triadic_colors(r, g, b):
    h, s, v = rgb_to_hsv(r, g, b)
    h1 = (h + 1/3) % 1.0
    h2 = (h + 2/3) % 1.0
    return hsv_to_rgb(h1, s, v), hsv_to_rgb(h2, s, v)

# Generate random RGB values for outfit colors
outfit_colors = np.random.randint(0, 256, size=(num_samples, 3))

matching_colors = []

for r, g, b in outfit_colors:
    # Randomly choose a color theory principle
    choice = np.random.choice(['complementary', 'analogous', 'triadic'])

    if choice == 'complementary':
        matching_color = generate_complementary_color(r, g, b)
    elif choice == 'analogous':
        matching_color = generate_analogous_colors(r, g, b)[0]  # Use one of the analogous colors
    elif choice == 'triadic':
        matching_color = generate_triadic_colors(r, g, b)[0]  # Use one of the triadic colors

    matching_colors.append(matching_color)

matching_colors = np.array(matching_colors)

# Create a DataFrame
df = pd.DataFrame({
    'outfit_color_r': outfit_colors[:, 0],
    'outfit_color_g': outfit_colors[:, 1],
    'outfit_color_b': outfit_colors[:, 2],
    'matching_color_r': matching_colors[:, 0],
    'matching_color_g': matching_colors[:, 1],
    'matching_color_b': matching_colors[:, 2],
})

# Save the dataset to a CSV file
df.to_csv('color_data.csv', index=False)

print("Color theory-based dataset generated and saved as 'color_matching_dataset_color_theory.csv'")

