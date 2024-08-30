from flask import Flask, request, jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Sample dataset of color pairs (outfit color, matching color)

data = {
    'outfit_color_r': [255, 0, 0, 0, 128, 255, 255],
    'outfit_color_g': [0, 255, 0, 128, 0, 255, 0],
    'outfit_color_b': [0, 0, 255, 128, 128, 0, 255],
    'matching_color_r': [0, 255, 0, 255, 128, 0, 128],
    'matching_color_g': [255, 0, 255, 0, 128, 0, 255],
    'matching_color_b': [0, 0, 0, 0, 255, 255, 128],
}

df = pd.DataFrame(data)

# Features (input colors) and target (matching colors)
X = df[['outfit_color_r', 'outfit_color_g', 'outfit_color_b']]
y = df[['matching_color_r', 'matching_color_g', 'matching_color_b']]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Function to predict matching color for a given outfit color
def predict_matching_color(outfit_color):
    outfit_color = np.array(outfit_color).reshape(1, -1)
    matching_color = model.predict(outfit_color)
    return np.clip(matching_color, 0, 255).astype(int).tolist()[0]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    outfit_color = [data['r'], data['g'], data['b']]
    matching_color = predict_matching_color(outfit_color)
    return jsonify({'matching_color': matching_color})


@app.route('/')
def index():
    return render_template('samplechat.html')


if __name__ == '__main__':
    app.run(debug=True)

