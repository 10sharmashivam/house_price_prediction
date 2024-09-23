from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the saved Ridge model
model = joblib.load('House_Predict_Price/Notebook/ridge_regression_model.pkl')

@app.route('/')
def home():
    return "Welcome to the House Price Prediction API!"

# Create a prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON request data
        data = request.get_json()
        
        # Extract features from the request (make sure it matches your feature shape)
        features = np.array(data['features']).reshape(1, -1)  # Reshape for single prediction
        
        # Make prediction using the loaded model
        prediction = model.predict(features)
        
        # Return the prediction as JSON
        return jsonify({'prediction': prediction[0]})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)