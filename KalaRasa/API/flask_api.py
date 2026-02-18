# api/flask_api.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from functools import wraps
import requests
import os
from datetime import datetime
import sys
from dotenv import load_dotenv  # Tambahkan ini

load_dotenv()  # Tambahkan ini

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.conversational_ai import ConversationalAI

# Initialize Flask app
app = Flask(__name__)
CORS(app)  

x_internal_key = os.getenv("NLP_SERVICE_KEY")

# Initialize NLP Engine
try:
    ai = ConversationalAI(db_config=None)  
    print("✓ NLP Service initialized")
except Exception as e:
    print(f"✗ Failed to initialize NLP: {e}")
    ai = None


# ================================================
# CORE ENDPOINTS
# ================================================


@app.route('/api/nlp/process', methods=['POST'])
@app.route('/api/nlp/process', methods=['POST'])
def process_nlp():

    data = request.get_json()

    if not data or 'message' not in data:
        return jsonify({
            'success': False,
            'error': 'message is required'
        }), 400

    try:
        nlp_result = ai.nlp_engine.process(data['message'])

        return jsonify({
            'success': True,
            'intent': nlp_result['intent'],
            'confidence': nlp_result['confidence'],
            'entities': nlp_result['entities'],
            'needs_clarification': nlp_result['action'] == 'ask_clarification',
            'clarification_question': nlp_result['message']
                if nlp_result['action'] == 'ask_clarification'
                else None
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok'
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)