from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load the CSV file
try:
    data = pd.read_csv('color_data.csv')  # Update this path if necessary
except FileNotFoundError:
    print("Error: The file 'color_data.csv' was not found.")
    exit()

# Preprocess the data for classification
X = data.iloc[:, :-1].values  # Features
y = data.iloc[:, -1].values   # Target

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier model
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Evaluate the classifier
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Prepare the data for color matching
df = pd.DataFrame(data)
X_colors = df[['outfit_color_r', 'outfit_color_g', 'outfit_color_b']]
y_colors = df[['matching_color_r', 'matching_color_g', 'matching_color_b']]

# Split the data into training and testing sets for color matching
X_train_colors, X_test_colors, y_train_colors, y_test_colors = train_test_split(X_colors, y_colors, test_size=0.2, random_state=42)

# Train a LinearRegression model for color matching
color_model = LinearRegression()
color_model.fit(X_train_colors, y_train_colors)

# Define a function to predict the matching color
def predict_matching_color(outfit_color):
    outfit_color = np.array(outfit_color).reshape(1, -1)
    matching_color = color_model.predict(outfit_color)
    return np.clip(matching_color, 0, 255).astype(int).tolist()[0]

# Define a function to visualize the color match
def plot_colors(color1, color2):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow([[color1]])
    ax[0].set_title('Outfit Color')
    ax[1].imshow([[color2]])
    ax[1].set_title('Predicted Matching Color')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return img_base64

@app.route('/')
def index():
    return render_template('samplechat.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '').strip().lower()  # Make it case-insensitive

    # Chatbot responses
    if user_message in ['hello', 'hi', 'hey']:
        bot_response = "Hello! How can I assist you today?"
    elif user_message in ['how are you', 'how are you doing']:
        bot_response = "I'm just a bot, but I'm here to help you with your outfit choices!"
    elif user_message in ['thank you', 'thanks']:
        bot_response = "You're welcome! Happy to help."
    elif user_message in ['bye', 'goodbye', 'see you']:
        bot_response = "Goodbye! See you again soon."
    else:
        bot_response = "I'm here to help you with your outfit choices. You can ask me anything related to that!"

    return jsonify({'response': bot_response})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    outfit_color = [data['r'], data['g'], data['b']]
    matching_color = predict_matching_color(outfit_color)
    img_base64 = plot_colors(outfit_color, matching_color)
    return jsonify({
        'matching_color': matching_color,
        'image': img_base64
    })

if __name__ == '__main__':
    app.run(debug=True)
