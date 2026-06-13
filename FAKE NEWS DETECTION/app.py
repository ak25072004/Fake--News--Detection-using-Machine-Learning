# app.py
from flask import Flask, render_template, request, jsonify
import pickle
import os

app = Flask(__name__)

# Load the trained ML model pipeline
MODEL_PATH = 'fake_news_model.pkl'

def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    return None

model_pipeline = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model_pipeline:
        return jsonify({'error': 'Model not trained. Please run model_trainer.py first.'}), 500

    data = request.get_json()
    news_text = data.get('text', '').strip()

    if not news_text:
        return jsonify({'error': 'Please enter some text to analyze.'}), 400

    try:
        # Perform prediction using the loaded pipeline
        prediction = model_pipeline.predict([news_text])[0]
        
        # Determine confidence/status (For true probabilities, use LogisticRegression or passives calibrated)
        # Here we return the direct string class ('REAL' or 'FAKE')
        return jsonify({
            'status': 'success',
            'prediction': prediction,
            'message': f"This article appears to be {prediction}."
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Set debug=False in production deployment environments
    app.run(debug=True, port=5000)