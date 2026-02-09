# api/flask_api.py
# RESTful API untuk Recipe Chatbot (Laravel Integration)

from flask import Flask, request, jsonify
from flask_cors import CORS
from functools import wraps
import jwt
import os
from datetime import datetime, timedelta
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.conversational_ai import ConversationalAI
from src.database_connector import DatabaseConnector

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for Laravel integration

# Configuration
app.config['SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-this')
app.config['JWT_EXPIRATION_HOURS'] = 24

# Initialize AI
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME', 'kala_rasa_jtv')
}

try:
    ai = ConversationalAI(DB_CONFIG)
    db = DatabaseConnector(DB_CONFIG)
    db.connect()
    print("✓ API initialized with MySQL")
except Exception as e:
    print(f"Warning: MySQL not available, using JSON fallback: {e}")
    ai = ConversationalAI(db_config=None)
    db = None


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


def generate_token(user_id: str) -> str:
    """Generate JWT token for user"""
    payload = {
        'user_id': user_id,
        'exp': datetime.utcnow() + timedelta(hours=app.config['JWT_EXPIRATION_HOURS']),
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')


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
        'database': 'mysql' if db else 'json',
        'version': '4.0'
    })


@app.route('/api/auth/token', methods=['POST'])
def get_token():
    """
    Generate authentication token
    
    POST /api/auth/token
    Body: { "user_id": "user123" }
    """
    data = request.get_json()
    
    if not data or 'user_id' not in data:
        return jsonify({
            'success': False,
            'error': 'user_id is required'
        }), 400
    
    token = generate_token(data['user_id'])
    
    return jsonify({
        'success': True,
        'token': token,
        'expires_in': app.config['JWT_EXPIRATION_HOURS'] * 3600
    })


@app.route('/api/chat', methods=['POST'])
@token_required
def chat(current_user_id):
    """
    Main chat endpoint
    
    POST /api/chat
    Headers: { "Authorization": "Bearer <token>" }
    Body: { 
        "message": "mau masak ayam goreng",
        "session_id": "optional-session-id"
    }
    
    Response: {
        "success": true,
        "response": "Natural language response",
        "recipes": [...],
        "suggestions": [...],
        "context": {...}
    }
    """
    data = request.get_json()
    
    if not data or 'message' not in data:
        return jsonify({
            'success': False,
            'error': 'message is required'
        }), 400
    
    message = data['message']
    
    # Process message
    try:
        result = ai.process_message(current_user_id, message)
        
        # Get context summary
        context = ai.get_or_create_context(current_user_id)
        
        return jsonify({
            'success': True,
            'response': result['response'],
            'recipes': result.get('recipes', []),
            'suggestions': result.get('suggestions', []),
            'context': {
                'collected_entities': context.collected_entities,
                'conversation_turns': len(context.history)
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/recipes/search', methods=['POST'])
@token_required
def search_recipes(current_user_id):
    """
    Search recipes with filters
    
    POST /api/recipes/search
    Body: {
        "ingredients": ["ayam", "ikan"],
        "cooking_methods": ["goreng"],
        "health_conditions": ["diabetes"],
        "max_time": 30,
        "limit": 10
    }
    """
    data = request.get_json()
    
    try:
        if db:
            recipes = db.search_recipes(
                ingredients=data.get('ingredients'),
                cooking_methods=data.get('cooking_methods'),
                taste_preferences=data.get('taste_preferences'),
                health_conditions=data.get('health_conditions'),
                max_time=data.get('max_time'),
                difficulty=data.get('difficulty'),
                limit=data.get('limit', 10)
            )
        else:
            # Fallback to JSON matcher
            recipes = []
        
        return jsonify({
            'success': True,
            'recipes': recipes,
            'count': len(recipes)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/recipes/<int:recipe_id>', methods=['GET'])
@token_required
def get_recipe(current_user_id, recipe_id):
    """
    Get recipe details
    
    GET /api/recipes/123
    """
    try:
        if db:
            recipe = db.get_recipe_details(recipe_id)
            
            if recipe:
                return jsonify({
                    'success': True,
                    'recipe': recipe
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Recipe not found'
                }), 404
        else:
            return jsonify({
                'success': False,
                'error': 'Database not available'
            }), 503
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/chat/history', methods=['GET'])
@token_required
def get_chat_history(current_user_id):
    """
    Get conversation history for user
    
    GET /api/chat/history
    """
    try:
        context = ai.get_or_create_context(current_user_id)
        
        return jsonify({
            'success': True,
            'history': context.history,
            'total_turns': len(context.history),
            'collected_entities': context.collected_entities
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/chat/clear', methods=['POST'])
@token_required
def clear_chat(current_user_id):
    """
    Clear conversation context
    
    POST /api/chat/clear
    """
    try:
        context = ai.get_or_create_context(current_user_id)
        context.clear_context()
        
        return jsonify({
            'success': True,
            'message': 'Conversation context cleared'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/ingredients', methods=['GET'])
@token_required
def get_ingredients(current_user_id):
    """
    Get list of all ingredients
    
    GET /api/ingredients?category=protein
    """
    try:
        if not db:
            return jsonify({
                'success': False,
                'error': 'Database not available'
            }), 503
        
        category = request.args.get('category')
        
        if category:
            db.cursor.execute("""
                SELECT id, nama, kategori, sub_kategori
                FROM ingredients
                WHERE kategori = %s
                ORDER BY nama
            """, (category,))
        else:
            db.cursor.execute("""
                SELECT id, nama, kategori, sub_kategori
                FROM ingredients
                ORDER BY kategori, nama
            """)
        
        ingredients = db.cursor.fetchall()
        
        return jsonify({
            'success': True,
            'ingredients': ingredients,
            'count': len(ingredients)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health-conditions', methods=['GET'])
@token_required
def get_health_conditions(current_user_id):
    """
    Get list of health conditions
    
    GET /api/health-conditions
    """
    try:
        if not db:
            return jsonify({
                'success': False,
                'error': 'Database not available'
            }), 503
        
        db.cursor.execute("""
            SELECT id, nama, description
            FROM health_conditions
            ORDER BY nama
        """)
        
        conditions = db.cursor.fetchall()
        
        return jsonify({
            'success': True,
            'conditions': conditions,
            'count': len(conditions)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health-conditions/<string:condition_name>/restrictions', methods=['GET'])
@token_required
def get_restrictions(current_user_id, condition_name):
    """
    Get ingredient restrictions for health condition
    
    GET /api/health-conditions/diabetes/restrictions
    """
    try:
        if not db:
            return jsonify({
                'success': False,
                'error': 'Database not available'
            }), 503
        
        restrictions = db.get_restricted_ingredients(condition_name)
        
        return jsonify({
            'success': True,
            'condition': condition_name,
            'restrictions': restrictions,
            'count': len(restrictions)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/analytics/popular-recipes', methods=['GET'])
@token_required
def get_popular_recipes(current_user_id):
    """
    Get popular recipes based on matches
    
    GET /api/analytics/popular-recipes?limit=10
    """
    try:
        if not db:
            return jsonify({
                'success': False,
                'error': 'Database not available'
            }), 503
        
        limit = request.args.get('limit', 10, type=int)
        
        db.cursor.execute("""
            SELECT 
                r.id,
                r.nama,
                COUNT(mr.id) as match_count,
                AVG(mr.match_score) as avg_score
            FROM matched_recipes mr
            JOIN recipes r ON mr.recipe_id = r.id
            GROUP BY r.id, r.nama
            ORDER BY match_count DESC
            LIMIT %s
        """, (limit,))
        
        popular = db.cursor.fetchall()
        
        return jsonify({
            'success': True,
            'popular_recipes': popular,
            'count': len(popular)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


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


# ================================================
# RUN SERVER
# ================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  RECIPE CHATBOT API SERVER")
    print("="*60)
    print("\nEndpoints:")
    print("  POST   /api/auth/token")
    print("  POST   /api/chat")
    print("  GET    /api/chat/history")
    print("  POST   /api/chat/clear")
    print("  POST   /api/recipes/search")
    print("  GET    /api/recipes/<id>")
    print("  GET    /api/ingredients")
    print("  GET    /api/health-conditions")
    print("  GET    /api/analytics/popular-recipes")
    print("\nServer starting on http://0.0.0.0:5000")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
