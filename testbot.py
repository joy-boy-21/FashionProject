import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier  # Example model
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the CSV file
data = pd.read_csv('color_data.csv')

# Step 2: Preprocess the data
# Assume the last column is the target variable and others are features
X = data.iloc[:, :-1].values  # Features
y = data.iloc[:, -1].values   # Target

# (Optional) Scaling the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Choose a model and train it
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))


df = pd.DataFrame(data)

# Features (input colors) and target (matching colors)
X = df[['outfit_color_r', 'outfit_color_g', 'outfit_color_b']]
y = df[['matching_color_r', 'matching_color_g', 'matching_color_b']]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

def predict_matching_color(outfit_color):
    outfit_color = np.array(outfit_color).reshape(1, -1)
    matching_color = model.predict(outfit_color)
    return np.clip(matching_color, 0, 255).astype(int).tolist()[0]

ip=int(input("ENTER::"))

while ip:
    def get_color_input():
        print("Enter the RGB values of the outfit color:")
        r = int(input("Red (0-255): "))
        g = int(input("Green (0-255): "))
        b = int(input("Blue (0-255): "))
        return [r, g, b]

# Get color input from user
    outfit_color = get_color_input()

# Predict the matching color
    predicted_matching_color = predict_matching_color(outfit_color)
    print(f'Predicted matching color for {outfit_color} is {predicted_matching_color}')

# Visualize the results
    def plot_colors(color1, color2, title1='Color 1', title2='Color 2'):
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow([[color1]])
        ax[0].set_title(title1)
        ax[1].imshow([[color2]])
        ax[1].set_title(title2)
        plt.show()

# Plot the predicted matching color against the input color
    plot_colors(outfit_color, predicted_matching_color, 'Outfit Color', 'Predicted Matching Color')

    
