from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
import os
from models.MaternalHealthChatbot import MaternalHealthChatbot
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Required for session handling
CORS(app)  # Enable CORS

# Initialize the chatbot
chatbot = MaternalHealthChatbot()

@app.route('/')
def index():
    """Render the vitals input page"""
    return render_template('index.html')

@app.route('/chat')
def chat():
    """Render the chat page for symptom input"""
    return render_template('chat.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Process user's vitals and symptoms to provide health predictions"""
    try:
        data = request.get_json()
        
        # Get vitals from the request
        vitals = {
            'SystolicBP': int(data.get('bp_systolic', 0)),
            'DiastolicBP': int(data.get('bp_diastolic', 0)),
            'BS': float(data.get('sugar', 0)),
            'BodyTemp': float(data.get('body_temp', 0)),
            'HeartRate': int(data.get('heart', 0)),
            'Age': float(data.get('age', 0))
        }

        symptoms = data.get('symptoms', "")

        # Process with the maternal health chatbot
        result = chatbot.process_user_input(vitals, symptoms)

        # Format the response for detected conditions
        detected_conditions = []
        if 'detected_conditions' in result:
            for condition in result['detected_conditions']:
                if isinstance(condition, dict) and 'disease' in condition:
                    detected_conditions.append(condition['disease'])
                else:
                    detected_conditions.append(str(condition))

        # Format the response
        response = {
            'risk_level': result.get('risk_level', 'Unknown'),
            'detected_conditions': detected_conditions,
            'concerns': result.get('concerns', {}),
            'recommendations': result.get('recommendations', {'general': [], 'condition_specific': {}}),
            'response': result.get('response', 'Unable to generate a response. Please try again.')
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e), 'message': 'An error occurred during prediction'}), 500

if __name__ == '__main__':
    app.run(debug=True)