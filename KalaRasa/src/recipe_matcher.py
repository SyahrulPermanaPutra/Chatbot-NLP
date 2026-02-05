# src/recipe_matcher.py
# Recipe Matching & Scoring System

import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import List, Dict, Tuple
from config.config import RECIPE_DATABASE


class RecipeMatcher:
    """
    Sistem untuk matching dan scoring resep berdasarkan
    hasil ekstraksi NLP pipeline
    """
    
    def __init__(self, recipe_db_path: str = RECIPE_DATABASE):
        """
        Initialize matcher dengan database resep
        Args:
            recipe_db_path: Path ke JSON database resep
        """
        self.recipes = self._load_recipes(recipe_db_path)
        print(f"Loaded {len(self.recipes)} recipes from database")
    
    def _load_recipes(self, path: str) -> List[Dict]:
        """Load recipe database dari JSON file"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading recipes: {e}")
            return []
    
    def match_recipes(self, nlp_output: Dict, top_k: int = 5) -> List[Dict]:
        """
        Match resep berdasarkan NLP output
        Args:
            nlp_output: Output dari NLP pipeline
            top_k: Jumlah resep teratas yang dikembalikan
        Returns:
            List of matched recipes dengan score
        """
        scored_recipes = []
        
        for recipe in self.recipes:
            score = self._calculate_score(recipe, nlp_output)
            
            if score > 0:  # Hanya masukkan resep yang punya score > 0
                scored_recipes.append({
                    'recipe': recipe,
                    'score': score,
                    'match_details': self._get_match_details(recipe, nlp_output)
                })
        
        # Sort by score (descending)
        scored_recipes.sort(key=lambda x: x['score'], reverse=True)
        
        return scored_recipes[:top_k]
    
    def _calculate_score(self, recipe: Dict, nlp_output: Dict) -> float:
        """
        Calculate matching score untuk resep
        Score components:
        - Ingredient match: 40 points
        - Cooking method match: 30 points
        - Taste preference match: 20 points
        - Safety check (constraints): DISQUALIFYING
        - Time constraint: 10 points
        """
        entities = nlp_output.get('entities', {})
        constraints = nlp_output.get('constraints', {})
        
        # SAFETY CHECK FIRST - Eliminasi resep yang tidak aman
        if not self._is_recipe_safe(recipe, constraints):
            return 0  # Automatic disqualification
        
        score = 0.0
        
        # 1. Ingredient Match (40 points max)
        ingredient_score = self._score_ingredients(
            recipe, 
            entities.get('ingredients', {})
        )
        score += ingredient_score * 40
        
        # 2. Cooking Method Match (30 points max)
        method_score = self._score_cooking_methods(
            recipe,
            entities.get('cooking_methods', [])
        )
        score += method_score * 30
        
        # 3. Taste Preference Match (20 points max)
        taste_score = self._score_taste_preferences(
            recipe,
            entities.get('taste_preferences', [])
        )
        score += taste_score * 20
        
        # 4. Time Constraint (10 points max)
        time_score = self._score_time_constraint(
            recipe,
            entities.get('time_constraint')
        )
        score += time_score * 10
        
        return score
    
    def _is_recipe_safe(self, recipe: Dict, constraints: Dict) -> bool:
        """
        Check apakah resep aman untuk user (tidak mengandung pantangan)
        """
        # Check must_exclude ingredients
        must_exclude = constraints.get('must_exclude', [])
        recipe_ingredients = recipe.get('bahan_utama', []) + recipe.get('bahan_tambahan', [])
        
        for exclude_item in must_exclude:
            for recipe_ingredient in recipe_ingredients:
                if exclude_item in recipe_ingredient.lower():
                    return False  # Resep mengandung bahan yang harus dihindari
        
        # Check dietary restrictions (kondisi kesehatan)
        dietary_restrictions = constraints.get('dietary_restrictions', [])
        for restriction in dietary_restrictions:
            condition = restriction.get('condition', '').lower()
            
            # Check apakah resep cocok untuk kondisi ini
            tidak_cocok_untuk = recipe.get('tidak_cocok_untuk', [])
            for nc in tidak_cocok_untuk:
                if condition in nc.lower():
                    return False  # Resep tidak cocok untuk kondisi kesehatan user
        
        return True
    
    def _score_ingredients(self, recipe: Dict, ingredients: Dict) -> float:
        """
        Score berdasarkan ingredient match
        Returns: 0.0 - 1.0
        """
        main_ingredients = ingredients.get('main', [])
        
        if not main_ingredients:
            return 0.5  # Neutral score jika tidak ada ingredient spesifik
        
        recipe_ingredients = recipe.get('bahan_utama', [])
        
        # Hitung berapa banyak ingredient yang match
        matches = 0
        for user_ing in main_ingredients:
            for recipe_ing in recipe_ingredients:
                if user_ing.lower() in recipe_ing.lower():
                    matches += 1
                    break
        
        # Normalize score
        score = matches / len(main_ingredients) if main_ingredients else 0
        return min(score, 1.0)
    
    def _score_cooking_methods(self, recipe: Dict, methods: List[str]) -> float:
        """
        Score berdasarkan cooking method match
        Returns: 0.0 - 1.0
        """
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
        """
        Score berdasarkan taste preference match
        Returns: 0.0 - 1.0
        """
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
        """
        Score berdasarkan time constraint
        Returns: 0.0 - 1.0
        """
        if not time_constraint:
            return 0.5  # Neutral score
        
        recipe_time = recipe.get('waktu_masak', 0)
        
        # Parse time constraint
        if 'cepat' in time_constraint.lower() or 'quick' in time_constraint.lower():
            # Cepat = < 30 menit
            return 1.0 if recipe_time <= 30 else 0.3
        
        elif 'simple' in time_constraint.lower() or 'mudah' in time_constraint.lower():
            # Simple based on difficulty
            difficulty = recipe.get('tingkat_kesulitan', '')
            return 1.0 if difficulty == 'mudah' else 0.5
        
        elif 'menit' in time_constraint.lower() or 'minutes' in time_constraint.lower():
            # Extract number
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
    
    def _get_match_details(self, recipe: Dict, nlp_output: Dict) -> Dict:
        """
        Get detailed match information
        """
        entities = nlp_output.get('entities', {})
        
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
        """
        Format resep untuk display yang user-friendly
        """
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


if __name__ == "__main__":
    # Testing
    print("=== Recipe Matcher Test ===\n")
    
    from src.nlp_pipeline import RecipeNLPPipeline
    
    # Initialize
    pipeline = RecipeNLPPipeline(load_models=True)
    matcher = RecipeMatcher()
    
    # Test queries
    test_queries = [
        "mau masak ayam goreng",
        "aku diabetes ga boleh gula",
        "pengen yang pedas gurih",
        "cariin resep ikan yang cepat",
        "kolesterol tinggi ga bisa santan"
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}")
        
        # Get NLP output
        nlp_output = pipeline.process(query)
        
        # Match recipes
        matched = matcher.match_recipes(nlp_output, top_k=3)
        
        if matched:
            print(f"\nFound {len(matched)} matching recipes:\n")
            for i, m in enumerate(matched, 1):
                print(f"\n#{i}")
                print(matcher.format_recipe_display(m))
        else:
            print("\nNo matching recipes found.")