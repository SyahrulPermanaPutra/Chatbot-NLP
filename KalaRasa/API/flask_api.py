# api/flask_api.py
# RESTful API untuk Recipe Chatbot (Laravel Integration)

from flask import Flask, request, jsonify
from flask_cors import CORS
from functools import wraps
import requests
import jwt
import os
from datetime import datetime, timedelta
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.conversational_ai import ConversationalAI

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for Laravel integration

# Configuration
app.config['SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'your-secret-key')
app.config['JWT_EXPIRATION_HOURS'] = 24

#URL Laravel Service
LARAVEL_API_URL = os.getenv('LARAVEL_API_URL', 'http://localhost:8000/api')
LARAVEL_API_KEY = os.getenv('LARAVEL_API_KEY', 'your-api-key')

# Initialize NLP Only
try:
    ai = ConversationalAI(db_config=None)  # Tanpa DB config
    print("✓ NLP Service initialized")
except Exception as e:
    print(f"✗ Failed to initialize NLP: {e}")
    ai = None

# ================================================
# MIDDLEWARE & UTILITIES
# ================================================

def token_required(f):
    """Decorator untuk API yang membutuhkan authentication"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        
        if not token:
            return jsonify({
                'success': False,
                'error': 'Token is missing'
            }), 401
        
        try:
            # Remove 'Bearer ' prefix if present
            if token.startswith('Bearer '):
                token = token[7:]
 
            # Verify token (bisa sync dengan Laravel JWT)
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user_id = data['user_id']
        except jwt.ExpiredSignatureError:
            return jsonify({
                'success': False,
                'error': 'Token has expired'
            }), 401
        except Exception as e:
            return jsonify({
                'success': False,
                'error': 'Token is invalid'
            }), 401
        
        return f(current_user_id, *args, **kwargs)
    
    return decorated

def call_laravel_api(endpoint: str, method: str = 'GET', data: dict = None):
    """Call Laravel API untuk recipe matching"""
    headers = {
        'X-API-Key': LARAVEL_API_KEY,
        'Content-Type': 'application/json'
    }
    
    url = f"{LARAVEL_API_URL}/{endpoint}"
    
    try:
        if method == 'GET':
            response = requests.get(url, headers=headers, params=data)
        else:
            response = requests.post(url, headers=headers, json=data)
        
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        print(f"Error calling Laravel API: {e}")
        return None


# ================================================
# CORE ENDPOINTS
# ================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'success': True,
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '4.0'
    })


@app.route('/api/nl/process', methods=['POST'])
@token_required
def process_nlp(current_user_id):
    
    data = request.get_json()
    
    if not data or 'message' not in data:
        return jsonify({'success': False, 'error': 'message is required'}), 400
    
    try:
        # Process ONLY NLP
        nlp_result = ai.nlp_engine.process(data['message'])
        
        # Get context if needed (optional)
        context = ai.get_or_create_context(current_user_id)
        if nlp_result.get('entities'):
            context.update_entities(nlp_result['entities'])
        
        # Return ONLY NLP result
        return jsonify({
            'success': True,
            'nlp_result': {
                'status': nlp_result['status'],
                'intent': nlp_result['intent'],
                'confidence': nlp_result['confidence'],
                'entities': nlp_result['entities'],
                'action': nlp_result['action'],
                'message': nlp_result['message']
            },
            'needs_clarification': nlp_result['action'] == 'ask_clarification',
            'clarification_question': nlp_result['message'] if nlp_result['action'] == 'ask_clarification' else None,
            'context_summary': context.get_summary()
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/nlp/batch', methods=['POST'])
@token_required
def process_batch_nlp(current_user_id):
    """
    Batch NLP Processing untuk multiple messages
    """
    data = request.get_json()
    
    if not data or 'messages' not in data:
        return jsonify({'success': False, 'error': 'messages array is required'}), 400
    
    messages = data['messages']
    results = []
    
    for msg in messages:
        try:
            nlp_result = ai.nlp_engine.process(msg)
            results.append({
                'message': msg,
                'nlp_result': nlp_result
            })
        except Exception as e:
            results.append({
                'message': msg,
                'error': str(e)
            })
    
    return jsonify({
        'success': True,
        'results': results,
        'total': len(results)
    })

@app.route('/api/nlp/context/<user_id>', methods=['GET'])
@token_required
def get_context(current_user_id, user_id):
    """
    Get conversation context for user
    """
    context = ai.get_or_create_context(user_id)
    
    return jsonify({
        'success': True,
        'context': {
            'user_id': context.user_id,
            'collected_entities': context.collected_entities,
            'conversation_turns': len(context.history),
            'last_recipes_count': len(context.last_recipes)
        }
    })

@app.route('/api/nlp/context/<user_id>/clear', methods=['POST'])
@token_required
def clear_context(current_user_id, user_id):
    """
    Clear conversation context
    """
    context = ai.get_or_create_context(user_id)
    context.clear_context()
    
    return jsonify({
        'success': True,
        'message': 'Context cleared'
    })

# ================================================
# ERROR HANDLERS
# ================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  NLP-ONLY  ")
    print("="*60)
    print("\nEndpoints:")
    print("  POST   /api/nlp/process     # NLP processing only")
    print("  POST   /api/nlp/batch        # Batch NLP processing")
    print("  GET    /api/nlp/context/<id> # Get context")
    print("  POST   /api/nlp/context/<id>/clear # Clear context")
    print("\nServer starting on http://0.0.0.0:5000")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
