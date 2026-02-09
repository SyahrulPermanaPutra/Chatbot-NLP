# src/conversational_ai.py
# Advanced Conversational AI dengan Context Memory

import json
from typing import Dict, List, Optional
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.enhanced_nlp_engine import EnhancedNLPEngine
from src.recipe_matcher_mysql import RecipeMatcherMySQL
from src.database_connector import DatabaseConnector


class ConversationContext:
    """
    Manage conversation context and history
    """
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.history = []
        self.current_intent = None
        self.collected_entities = {
            'ingredients': {'main': [], 'avoid': []},
            'cooking_methods': [],
            'taste_preferences': [],
            'health_conditions': [],
            'time_constraint': None
        }
        self.pending_clarification = None
        self.last_recipes = []
        self.created_at = datetime.now()
    
    def add_turn(self, user_message: str, bot_response: str, nlp_result: Dict):
        """Add conversation turn to history"""
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'user': user_message,
            'bot': bot_response,
            'nlp': nlp_result
        })
    
    def update_entities(self, new_entities: Dict):
        """Merge new entities dengan existing context"""
        # Merge ingredients
        if 'ingredients' in new_entities:
            if 'main' in new_entities['ingredients']:
                self.collected_entities['ingredients']['main'].extend(
                    new_entities['ingredients']['main']
                )
            if 'avoid' in new_entities['ingredients']:
                self.collected_entities['ingredients']['avoid'].extend(
                    new_entities['ingredients']['avoid']
                )
        
        # Merge other entities
        for key in ['cooking_methods', 'taste_preferences', 'health_conditions']:
            if key in new_entities and new_entities[key]:
                if isinstance(new_entities[key], list):
                    self.collected_entities[key].extend(new_entities[key])
                else:
                    self.collected_entities[key].append(new_entities[key])
        
        # Update time constraint
        if new_entities.get('time_constraint'):
            self.collected_entities['time_constraint'] = new_entities['time_constraint']
        
        # Deduplicate
        for key in ['cooking_methods', 'taste_preferences', 'health_conditions']:
            self.collected_entities[key] = list(set(self.collected_entities[key]))
        
        self.collected_entities['ingredients']['main'] = list(set(
            self.collected_entities['ingredients']['main']
        ))
        self.collected_entities['ingredients']['avoid'] = list(set(
            self.collected_entities['ingredients']['avoid']
        ))
    
    def clear_context(self):
        """Clear collected entities untuk conversation baru"""
        self.collected_entities = {
            'ingredients': {'main': [], 'avoid': []},
            'cooking_methods': [],
            'taste_preferences': [],
            'health_conditions': [],
            'time_constraint': None
        }
        self.pending_clarification = None
    
    def get_summary(self) -> str:
        """Get conversation summary"""
        return f"User {self.user_id}: {len(self.history)} turns, " \
               f"Ingredients: {self.collected_entities['ingredients']['main']}, " \
               f"Methods: {self.collected_entities['cooking_methods']}"


class ConversationalAI:
    """
    Advanced Conversational AI untuk Recipe Chatbot
    Features:
    - Multi-turn conversation
    - Context memory
    - Smart clarification
    - Natural responses
    - Intent chaining
    """
    
    def __init__(self, db_config: Dict = None):
        """Initialize conversational AI"""
        self.nlp_engine = EnhancedNLPEngine()
        
        if db_config:
            self.matcher = RecipeMatcherMySQL(db_config)
            self.db = DatabaseConnector(db_config)
            self.db.connect()
        else:
            # Fallback to JSON-based matcher
            from src.recipe_matcher import RecipeMatcher
            self.matcher = RecipeMatcher()
            self.db = None
        
        self.conversations = {}  # user_id -> ConversationContext
        
        print("✓ Conversational AI initialized")
    
    def get_or_create_context(self, user_id: str) -> ConversationContext:
        """Get or create conversation context for user"""
        if user_id not in self.conversations:
            self.conversations[user_id] = ConversationContext(user_id)
        return self.conversations[user_id]
    
    def process_message(self, user_id: str, message: str) -> Dict:
        """
        Process user message dalam conversational context
        
        Returns:
            {
                'response': str,  # Natural language response
                'recipes': List[Dict],  # Matched recipes (if any)
                'suggestions': List[str],  # Suggested actions
                'context_updated': bool
            }
        """
        context = self.get_or_create_context(user_id)
        
        # Handle pending clarification first
        if context.pending_clarification:
            return self._handle_clarification_response(user_id, message, context)
        
        # Process with NLP
        nlp_result = self.nlp_engine.process(message)
        
        # Update context with new entities
        if nlp_result.get('entities'):
            context.update_entities(nlp_result['entities'])
        
        # Generate response based on action
        if nlp_result['action'] == 'match_recipe':
            response_data = self._handle_recipe_search(user_id, context, nlp_result)
        
        elif nlp_result['action'] == 'ask_clarification':
            response_data = self._handle_clarification(user_id, context, nlp_result)
        
        elif nlp_result['action'] == 'reject_input':
            response_data = self._handle_rejection(user_id, context, nlp_result)
        
        else:
            response_data = self._handle_general(user_id, context, nlp_result)
        
        # Add to conversation history
        context.add_turn(
            user_message=message,
            bot_response=response_data['response'],
            nlp_result=nlp_result
        )
        
        # Log to database if available
        if self.db:
            self._log_conversation(user_id, message, nlp_result, response_data)
        
        return response_data
    
    def _handle_recipe_search(
        self, 
        user_id: str, 
        context: ConversationContext,
        nlp_result: Dict
    ) -> Dict:
        """Handle recipe search with context"""
        
        # Create enhanced NLP output with context
        enhanced_output = nlp_result.copy()
        enhanced_output['entities'] = context.collected_entities
        
        # Search recipes
        matches = self.matcher.match_recipes(enhanced_output, top_k=5)
        
        context.last_recipes = matches
        
        if matches:
            # Generate natural response
            num_recipes = len(matches)
            response = f"Oke, aku menemukan {num_recipes} resep yang cocok! 🍳\n\n"
            
            # Show top 3
            for i, match in enumerate(matches[:3], 1):
                recipe = match['recipe']
                score = match['score']
                response += f"{i}. **{recipe['nama']}** ({score:.0f}/100)\n"
                response += f"   ⏱ {recipe['waktu_masak']} menit | "
                response += f"👨‍🍳 {recipe['tingkat_kesulitan']} | "
                response += f"🔥 {recipe['kalori_per_porsi']} kal\n\n"
            
            if num_recipes > 3:
                response += f"...dan {num_recipes - 3} resep lainnya.\n\n"
            
            # Add suggestions
            suggestions = [
                f"Lihat detail resep nomor {i}" for i in range(1, min(4, num_recipes + 1))
            ]
            suggestions.append("Cari resep lain")
            suggestions.append("Ubah kriteria")
            
            return {
                'response': response,
                'recipes': matches,
                'suggestions': suggestions,
                'context_updated': True
            }
        
        else:
            response = "Hmm, aku belum menemukan resep yang pas dengan kriteria kamu 🤔\n\n"
            
            # Suggest relaxing constraints
            if context.collected_entities['ingredients']['avoid']:
                response += "Mungkin pantangannya terlalu banyak? Coba kurangi batasan.\n"
            
            if context.collected_entities['time_constraint']:
                response += "Atau waktunya terlalu singkat? Coba perpanjang waktu.\n"
            
            suggestions = [
                "Ubah bahan utama",
                "Kurangi pantangan",
                "Coba teknik masak lain"
            ]
            
            return {
                'response': response,
                'recipes': [],
                'suggestions': suggestions,
                'context_updated': False
            }
    
    def _handle_clarification(
        self,
        user_id: str,
        context: ConversationContext,
        nlp_result: Dict
    ) -> Dict:
        """Handle clarification request"""
        
        message = nlp_result.get('message', '')
        
        # Mark as pending clarification
        context.pending_clarification = {
            'type': 'ingredient',  # or 'method', 'taste', etc
            'original_message': message
        }
        
        # Make response more conversational
        response = "Oke, aku butuh info lebih dulu! 😊\n\n" + message
        
        suggestions = ["Ayam", "Ikan", "Sayur", "Daging", "Seafood"]
        
        return {
            'response': response,
            'recipes': [],
            'suggestions': suggestions,
            'context_updated': False
        }
    
    def _handle_clarification_response(
        self,
        user_id: str,
        message: str,
        context: ConversationContext
    ) -> Dict:
        """Handle response to clarification"""
        
        # Extract ingredient from response
        message_lower = message.lower()
        
        # Add to context
        if any(word in message_lower for word in ['ayam', 'chicken']):
            context.collected_entities['ingredients']['main'].append('ayam')
        elif any(word in message_lower for word in ['ikan', 'fish']):
            context.collected_entities['ingredients']['main'].append('ikan')
        elif any(word in message_lower for word in ['sayur', 'vegetable', 'veggie']):
            context.collected_entities['ingredients']['main'].append('sayur')
        elif any(word in message_lower for word in ['daging', 'beef', 'sapi']):
            context.collected_entities['ingredients']['main'].append('daging')
        elif any(word in message_lower for word in ['seafood', 'udang', 'cumi']):
            context.collected_entities['ingredients']['main'].append('seafood')
        else:
            # Try to extract from message
            context.collected_entities['ingredients']['main'].append(message.strip())
        
        # Clear pending
        context.pending_clarification = None
        
        # Now search with updated context
        enhanced_output = {
            'status': 'ok',
            'intent': 'cari_resep',
            'confidence': 0.9,
            'entities': context.collected_entities,
            'action': 'match_recipe',
            'message': ''
        }
        
        return self._handle_recipe_search(user_id, context, enhanced_output)
    
    def _handle_rejection(
        self,
        user_id: str,
        context: ConversationContext,
        nlp_result: Dict
    ) -> Dict:
        """Handle input rejection"""
        
        response = nlp_result.get('message', 'Maaf, aku tidak mengerti.')
        response += "\n\nContoh yang bisa kamu coba:\n"
        response += "• 'Mau masak ayam goreng'\n"
        response += "• 'Cariin resep ikan bakar'\n"
        response += "• 'Aku diabetes, mau masak apa ya?'\n"
        
        return {
            'response': response,
            'recipes': [],
            'suggestions': ["Cari resep ayam", "Cari resep ikan", "Tanya kondisi kesehatan"],
            'context_updated': False
        }
    
    def _handle_general(
        self,
        user_id: str,
        context: ConversationContext,
        nlp_result: Dict
    ) -> Dict:
        """Handle general intents"""
        
        intent = nlp_result.get('intent', 'unknown')
        
        if intent == 'chitchat':
            responses = {
                'terima kasih': 'Sama-sama! Senang bisa bantu 😊',
                'thanks': 'You\'re welcome! 😊',
                'halo': 'Halo! Ada yang bisa aku bantu? 🍳',
                'hai': 'Hai! Mau masak apa hari ini? 😊',
                'oke': 'Oke! Ada yang mau ditanyakan lagi?',
                'baik': 'Baik! Silakan kalau ada yang mau ditanyakan.',
            }
            
            message_lower = nlp_result.get('entities', {}).get('original_text', '').lower()
            
            for key, value in responses.items():
                if key in message_lower:
                    return {
                        'response': value,
                        'recipes': [],
                        'suggestions': ["Cari resep", "Lihat resep favorit", "Tanya nutrisi"],
                        'context_updated': False
                    }
        
        # Default response
        return {
            'response': "Hmm, aku kurang paham maksud kamu. Bisa dijelaskan lebih detail? 🤔",
            'recipes': [],
            'suggestions': ["Cari resep", "Tanya alternatif bahan", "Tanya cara masak"],
            'context_updated': False
        }
    
    def _log_conversation(
        self,
        user_id: str,
        message: str,
        nlp_result: Dict,
        response_data: Dict
    ):
        """Log conversation to database"""
        try:
            query_id = self.db.log_user_query(
                query_text=message,
                intent=nlp_result.get('intent', 'unknown'),
                confidence=nlp_result.get('confidence', 0.0),
                status=nlp_result.get('status', 'unknown'),
                entities=nlp_result.get('entities', {})
            )
            
            if response_data.get('recipes') and query_id:
                self.db.log_matched_recipes(query_id, response_data['recipes'])
        
        except Exception as e:
            print(f"Warning: Could not log conversation: {e}")
    
    def get_recipe_detail(self, user_id: str, recipe_index: int) -> Dict:
        """Get detailed recipe by index from last search"""
        context = self.get_or_create_context(user_id)
        
        if not context.last_recipes:
            return {
                'response': "Belum ada pencarian resep. Coba cari resep dulu! 😊",
                'recipe': None,
                'suggestions': ["Cari resep ayam", "Cari resep ikan"]
            }
        
        if recipe_index < 1 or recipe_index > len(context.last_recipes):
            return {
                'response': f"Nomor resep tidak valid. Pilih antara 1-{len(context.last_recipes)}",
                'recipe': None,
                'suggestions': [f"Lihat resep {i}" for i in range(1, min(4, len(context.last_recipes) + 1))]
            }
        
        match = context.last_recipes[recipe_index - 1]
        recipe = match['recipe']
        
        # Format detailed response
        response = f"**{recipe['nama']}** 📖\n\n"
        response += f"⏱ Waktu: {recipe['waktu_masak']} menit\n"
        response += f"👨‍🍳 Tingkat: {recipe['tingkat_kesulitan']}\n"
        response += f"🔥 Kalori: {recipe['kalori_per_porsi']} per porsi\n\n"
        
        response += "**Bahan Utama:**\n"
        for ing in recipe.get('bahan_utama', []):
            response += f"• {ing}\n"
        
        response += "\n**Teknik Memasak:**\n"
        for method in recipe.get('teknik_masak', []):
            response += f"• {method.capitalize()}\n"
        
        if recipe.get('cocok_untuk'):
            response += "\n✅ **Cocok untuk:**\n"
            for cocok in recipe['cocok_untuk']:
                response += f"• {cocok}\n"
        
        if recipe.get('tidak_cocok_untuk'):
            response += "\n⚠️ **Tidak cocok untuk:**\n"
            for tidak in recipe['tidak_cocok_untuk']:
                response += f"• {tidak}\n"
        
        suggestions = [
            "Cari resep serupa",
            "Tanya substitusi bahan",
            "Hitung nutrisi",
            "Buat shopping list"
        ]
        
        return {
            'response': response,
            'recipe': recipe,
            'suggestions': suggestions
        }


if __name__ == "__main__":
    # Testing
    print("=== Conversational AI Test ===\n")
    
    # Initialize
    ai = ConversationalAI(db_config=None)  # Use JSON matcher for test
    
    # Simulate conversation
    user_id = "test_user_1"
    
    conversations = [
        "halo",
        "mau masak ayam",
        "yang goreng",
        "lihat resep 1",
        "terima kasih"
    ]
    
    for msg in conversations:
        print(f"\nUser: {msg}")
        result = ai.process_message(user_id, msg)
        print(f"Bot: {result['response']}")
        if result.get('suggestions'):
            print(f"Suggestions: {result['suggestions'][:3]}")
