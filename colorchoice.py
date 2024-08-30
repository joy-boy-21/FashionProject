from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Function to convert hex color to RGB
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# Mapping of human-readable color names to hex codes
color_names = {
    "Red": "#FF0000",
    "Green": "#00FF00",
    "Blue": "#0000FF",
    "Yellow": "#FFFF00",
    "Orange": "#FFA500",
    "Purple": "#800080",
    "Dark Green": "#008000",
    "Magenta": "#FF00FF",
    "Cyan": "#00FFFF",
    "Tomato": "#FF6347",
    "Gold": "#FFD700",
    "Teal": "#008080",
    "Orange Red": "#FF4500",
    "Steel Blue": "#4682B4",
    "Crimson": "#DC143C",
    "Dark Turquoise": "#00CED1",
    "Firebrick": "#B22222",
    "Light Sea Green": "#20B2AA",
    "Chocolate": "#D2691E",
    "Cornflower Blue": "#6495ED",
    "Deep Pink": "#FF1493",
    "Lime Green": "#32CD32"
}

# Reverse mapping to get the color name from hex code
reverse_color_names = {v: k for k, v in color_names.items()}

# Matching colors dictionary
comp_color = {
    "#FF0000": "#00FF00",
    "#00FF00": "#FF0000",
    "#0000FF": "#FFFF00",
    "#FFFF00": "#0000FF",
    "#FFA500": "#800080",
    "#800080": "#FFA500",
    "#008000": "#FF00FF",
    "#FF00FF": "#008000",
    "#00FFFF": "#FF00FF",
    "#FF6347": "#4682B4",
    "#FFD700": "#FF69B4",
    "#008080": "#FF4500",
    "#FF4500": "#008080",
    "#4682B4": "#FF6347",
    "#DC143C": "#00CED1",
    "#00CED1": "#DC143C",
    "#B22222": "#20B2AA",
    "#20B2AA": "#B22222",
    "#D2691E": "#6495ED",
    "#6495ED": "#D2691E",
    "#FF1493": "#32CD32",
    "#32CD32": "#FF1493"
}

# Convert hex colors to RGB
X = [hex_to_rgb(color) for color in comp_color.keys()]
y = [hex_to_rgb(color) for color in comp_color.values()]

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Train a simple K-Nearest Neighbors model
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X, y)

# Function to find the closest matching color in reverse_color_names
def find_closest_color_name(rgb_color):
    min_distance = float('inf')
    closest_color_name = None
    for hex_code, name in reverse_color_names.items():
        candidate_rgb = np.array(hex_to_rgb(hex_code))
        distance = np.linalg.norm(candidate_rgb - rgb_color)
        if distance < min_distance:
            min_distance = distance
            closest_color_name = name
    return closest_color_name

# Function to predict the matching color
def predict_matching_color(hex_color):
    rgb_color = np.array([hex_to_rgb(hex_color)])
    matching_color_rgb = model.predict(rgb_color)[0]
    # Find the closest matching color name
    return find_closest_color_name(matching_color_rgb)

# Function to display the color options and get user input
def get_user_color_choice():
    print("Please choose a color from the following options:")
    color_options = list(color_names.keys())
    for i, color in enumerate(color_options, start=1):
        print(f"{i}. {color}")

    choice = int(input("Enter the number corresponding to your color choice: "))
    
    # Validate the user input
    if 1 <= choice <= len(color_options):
        selected_color = color_names[color_options[choice - 1]]
        return selected_color
    else:
        print("Invalid choice. Please try again.")
        return get_user_color_choice()

# Get the user's color choice
user_color = get_user_color_choice()

# Predict the matching color
predicted_matching_color = predict_matching_color(user_color)
print(f"The matching color for your choice is {predicted_matching_color}")
