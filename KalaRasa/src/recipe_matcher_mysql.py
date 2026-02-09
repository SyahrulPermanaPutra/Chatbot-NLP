# src/recipe_matcher_mysql.py
# MySQL-based Recipe Matcher

from typing import Dict, List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database_connector import DatabaseConnector


class RecipeMatcherMySQL:
    """
    MySQL-based Recipe Matcher
    Uses database queries for efficient recipe matching
    """
    
    def __init__(self, db_config: Dict = None):
        """
        Initialize with database connection
        
        Args:
            db_config: MySQL connection config
        """
        self.db = DatabaseConnector(db_config)
        if not self.db.connect():
            raise ConnectionError("Failed to connect to MySQL database")
    
    def match_recipes(self, nlp_output: Dict, top_k: int = 5) -> List[Dict]:
        """
        Match recipes based on NLP output
        
        Args:
            nlp_output: Output from Enhanced NLP Engine
            top_k: Number of top recipes to return
            
        Returns:
            List of matched recipes with scores
        """
        # Extract criteria from NLP output
        entities = nlp_output.get('entities', {})
        
        ingredients = entities.get('ingredients', {}).get('main', [])
        cooking_methods = entities.get('cooking_methods', [])
        taste_preferences = entities.get('taste_preferences', [])
        health_conditions = entities.get('health_conditions', [])
        time_constraint = entities.get('time_constraint')
        
        # Parse time constraint
        max_time = None
        if time_constraint:
            if 'cepat' in time_constraint.lower() or 'quick' in time_constraint.lower():
                max_time = 30
            elif 'minutes' in time_constraint.lower() or 'menit' in time_constraint.lower():
                import re
                match = re.search(r'(\d+)', time_constraint)
                if match:
                    max_time = int(match.group(1))
        
        # Search recipes
        recipes = self.db.search_recipes(
            ingredients=ingredients,
            cooking_methods=cooking_methods,
            taste_preferences=taste_preferences,
            health_conditions=health_conditions,
            max_time=max_time,
            limit=50  # Get more candidates for scoring
        )
        
        # Score and rank
        scored_recipes = []
        for recipe in recipes:
            score = self._calculate_score(recipe, entities)
            
            # Only include recipes with positive score
            if score > 0:
                scored_recipes.append({
                    'recipe': recipe,
                    'score': score,
                    'match_details': self._get_match_details(recipe, entities)
                })
        
        # Sort by score (descending)
        scored_recipes.sort(key=lambda x: x['score'], reverse=True)
        
        return scored_recipes[:top_k]
    
    def _calculate_score(self, recipe: Dict, entities: Dict) -> float:
        """
        Calculate match score for recipe
        
        Score components:
        - Ingredient match: 40 points
        - Cooking method match: 30 points  
        - Taste preference match: 20 points
        - Time constraint: 10 points
        """
        score = 0.0
        
        # 1. Ingredient Match (40 points)
        ingredient_score = self._score_ingredients(
            recipe,
            entities.get('ingredients', {})
        )
        score += ingredient_score * 40
        
        # 2. Cooking Method Match (30 points)
        method_score = self._score_cooking_methods(
            recipe,
            entities.get('cooking_methods', [])
        )
        score += method_score * 30
        
        # 3. Taste Preference Match (20 points)
        taste_score = self._score_taste_preferences(
            recipe,
            entities.get('taste_preferences', [])
        )
        score += taste_score * 20
        
        # 4. Time Constraint (10 points)
        time_score = self._score_time_constraint(
            recipe,
            entities.get('time_constraint')
        )
        score += time_score * 10
        
        return score
    
    def _score_ingredients(self, recipe: Dict, ingredients: Dict) -> float:
        """Score based on ingredient match"""
        main_ingredients = ingredients.get('main', [])
        
        if not main_ingredients:
            return 0.5  # Neutral score
        
        recipe_ingredients = recipe.get('bahan_utama', [])
        
        matches = 0
        for user_ing in main_ingredients:
            for recipe_ing in recipe_ingredients:
                if user_ing.lower() in recipe_ing.lower():
                    matches += 1
                    break
        
        score = matches / len(main_ingredients) if main_ingredients else 0
        return min(score, 1.0)
    
    def _score_cooking_methods(self, recipe: Dict, methods: List[str]) -> float:
        """Score based on cooking method match"""
        if not methods:
            return 0.5  # Neutral score
        
        recipe_methods = recipe.get('teknik_masak', [])
        
        matches = 0
        for user_method in methods:
            for recipe_method in recipe_methods:
                if user_method.lower() in recipe_method.lower():
                    matches += 1
                    break
        
        score = matches / len(methods) if methods else 0
        return min(score, 1.0)
    
    def _score_taste_preferences(self, recipe: Dict, preferences: List[str]) -> float:
        """Score based on taste preference match"""
        if not preferences:
            return 0.5  # Neutral score
        
        recipe_tastes = recipe.get('kategori_rasa', [])
        
        matches = 0
        for pref in preferences:
            for taste in recipe_tastes:
                if pref.lower() in taste.lower():
                    matches += 1
                    break
        
        score = matches / len(preferences) if preferences else 0
        return min(score, 1.0)
    
    def _score_time_constraint(self, recipe: Dict, time_constraint: str) -> float:
        """Score based on time constraint"""
        if not time_constraint:
            return 0.5  # Neutral score
        
        recipe_time = recipe.get('waktu_masak', 0)
        
        if 'cepat' in time_constraint.lower() or 'quick' in time_constraint.lower():
            return 1.0 if recipe_time <= 30 else 0.3
        
        elif 'simple' in time_constraint.lower() or 'mudah' in time_constraint.lower():
            difficulty = recipe.get('tingkat_kesulitan', '')
            return 1.0 if difficulty == 'mudah' else 0.5
        
        elif 'menit' in time_constraint.lower() or 'minutes' in time_constraint.lower():
            import re
            match = re.search(r'(\d+)', time_constraint)
            if match:
                requested_time = int(match.group(1))
                if recipe_time <= requested_time:
                    return 1.0
                elif recipe_time <= requested_time * 1.5:
                    return 0.5
                else:
                    return 0.0
        
        return 0.5
    
    def _get_match_details(self, recipe: Dict, entities: Dict) -> Dict:
        """Get detailed match information"""
        details = {
            'matched_ingredients': [],
            'matched_methods': [],
            'matched_tastes': [],
            'safe_for_conditions': []
        }
        
        # Matched ingredients
        main_ingredients = entities.get('ingredients', {}).get('main', [])
        recipe_ingredients = recipe.get('bahan_utama', [])
        for user_ing in main_ingredients:
            for recipe_ing in recipe_ingredients:
                if user_ing.lower() in recipe_ing.lower():
                    details['matched_ingredients'].append(recipe_ing)
        
        # Matched methods
        methods = entities.get('cooking_methods', [])
        recipe_methods = recipe.get('teknik_masak', [])
        for user_method in methods:
            for recipe_method in recipe_methods:
                if user_method.lower() in recipe_method.lower():
                    details['matched_methods'].append(recipe_method)
        
        # Matched tastes
        preferences = entities.get('taste_preferences', [])
        recipe_tastes = recipe.get('kategori_rasa', [])
        for pref in preferences:
            for taste in recipe_tastes:
                if pref.lower() in taste.lower():
                    details['matched_tastes'].append(taste)
        
        # Safe for conditions
        health_conditions = entities.get('health_conditions', [])
        cocok_untuk = recipe.get('cocok_untuk', [])
        for condition in health_conditions:
            for cocok in cocok_untuk:
                if condition.lower() in cocok.lower():
                    details['safe_for_conditions'].append(cocok)
        
        return details
    
    def format_recipe_display(self, matched_recipe: Dict) -> str:
        """Format recipe for user-friendly display"""
        recipe = matched_recipe['recipe']
        score = matched_recipe['score']
        details = matched_recipe['match_details']
        
        output = []
        output.append(f"{'='*60}")
        output.append(f"📖 {recipe['nama'].upper()}")
        output.append(f"   Match Score: {score:.1f}/100")
        output.append(f"{'='*60}")
        
        # Basic info
        output.append(f"⏱️  Waktu: {recipe['waktu_masak']} menit")
        output.append(f"👨‍🍳 Tingkat: {recipe['tingkat_kesulitan'].capitalize()}")
        output.append(f"🔥 Kalori: {recipe['kalori_per_porsi']} per porsi")
        
        # Ingredients
        output.append(f"\n🥘 Bahan Utama:")
        for ing in recipe['bahan_utama']:
            mark = "✓" if ing in details['matched_ingredients'] else " "
            output.append(f"  [{mark}] {ing}")
        
        # Cooking methods
        output.append(f"\n👩‍🍳 Teknik Memasak:")
        for method in recipe['teknik_masak']:
            mark = "✓" if method in details['matched_methods'] else " "
            output.append(f"  [{mark}] {method.capitalize()}")
        
        # Taste
        output.append(f"\n👅 Rasa:")
        for taste in recipe['kategori_rasa']:
            mark = "✓" if taste in details['matched_tastes'] else " "
            output.append(f"  [{mark}] {taste.capitalize()}")
        
        # Health info
        if recipe.get('cocok_untuk'):
            output.append(f"\n✅ Cocok untuk:")
            for cocok in recipe['cocok_untuk']:
                output.append(f"  • {cocok}")
        
        if recipe.get('tidak_cocok_untuk'):
            output.append(f"\n⚠️  Tidak cocok untuk:")
            for tidak in recipe['tidak_cocok_untuk']:
                output.append(f"  • {tidak}")
        
        return '\n'.join(output)
    
    def __del__(self):
        """Cleanup database connection"""
        if hasattr(self, 'db'):
            self.db.disconnect()


if __name__ == "__main__":
    # Testing
    print("=== MySQL Recipe Matcher Test ===\n")
    
    try:
        # Initialize
        db_config = {
            'host': 'localhost',
            'user': 'root',
            'password': '', 
            'database': 'kala_rasa_jtv'
        }
        
        matcher = RecipeMatcherMySQL(db_config)
        
        # Test with sample NLP output
        nlp_output = {
            'status': 'ok',
            'intent': 'cari_resep',
            'confidence': 0.85,
            'entities': {
                'ingredients': {
                    'main': ['ayam'],
                    'avoid': []
                },
                'cooking_methods': ['goreng'],
                'taste_preferences': [],
                'health_conditions': [],
                'time_constraint': None
            }
        }
        
        print("Test Query: mau masak ayam goreng\n")
        
        matches = matcher.match_recipes(nlp_output, top_k=3)
        
        if matches:
            print(f"Found {len(matches)} matching recipes:\n")
            for i, match in enumerate(matches, 1):
                print(f"#{i}")
                print(matcher.format_recipe_display(match))
                print()
        else:
            print("No matching recipes found")
        
    except ConnectionError as e:
        print(f"✗ Database connection failed: {e}")