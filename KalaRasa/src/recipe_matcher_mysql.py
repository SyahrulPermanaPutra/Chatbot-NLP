# src/recipe_matcher_mysql.py - VERSION 2.0
# Enhanced Recipe Matcher dengan akurasi tinggi

from typing import Dict, List, Tuple, Set
import sys
import os
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database_connector import DatabaseConnector


class RecipeMatcherMySQL:
    """
    Enhanced MySQL-based Recipe Matcher dengan:
    1. Semantic ingredient matching
    2. Category-based scoring  
    3. Priority-based ranking
    4. Better handling for seafood, meat, etc.
    """
    
    # Ingredient categories untuk semantic matching
    INGREDIENT_CATEGORIES = {
        'seafood': {
            'keywords': ['ikan', 'udang', 'cumi', 'kepiting', 'kerang', 'seafood', 'laut', 'tiram', 'remis'],
            'related': ['salmon', 'tuna', 'kakap', 'gurame', 'lele', 'teri', 'tongkol']
        },
        'poultry': {
            'keywords': ['ayam', 'bebek', 'angsa', 'unggas', 'puyuh'],
            'related': ['daging ayam', 'ayam kampung', 'ayam potong']
        },
        'red_meat': {
            'keywords': ['sapi', 'kambing', 'domba', 'daging', 'daging sapi', 'daging kambing'],
            'related': ['steak', 'rendang', 'iga', 'has dalam']
        },
        'vegetarian': {
            'keywords': ['tempe', 'tahu', 'tahu sutra', 'tempe goreng', 'tahu goreng'],
            'related': ['protein nabati', 'kedelai']
        },
        'carbohydrate': {
            'keywords': ['nasi', 'mie', 'pasta', 'spaghetti', 'kwetiau', 'bihun', 'vermicelli', 'makaron'],
            'related': ['beras', 'mie instan', 'pasta gandum']
        },
        'vegetables': {
            'keywords': ['wortel', 'brokoli', 'kangkung', 'bayam', 'kentang', 'jagung', 'kol', 'sawi'],
            'related': ['sayur', 'sayuran', 'sayur mayur']
        }
    }
    
    # Cooking method categories
    COOKING_METHODS = {
        'goreng': ['goreng', 'deep fry', 'fry', 'menggoreng'],
        'rebus': ['rebus', 'boil', 'merebus'],
        'panggang': ['panggang', 'bake', 'roast', 'memanggang', 'bakar'],
        'tumis': ['tumis', 'sauté', 'sautéed', 'menumis'],
        'kukus': ['kukus', 'steam', 'mengukus'],
        'pepes': ['pepes', 'pepesan'],
        'gulai': ['gulai', 'gulai kambing', 'gulai ayam'],
        'rica_rica': ['rica-rica', 'rica rica'],
        'balado': ['balado'],
        'rendang': ['rendang']
    }
    
    def __init__(self, db_config: Dict = None):
        """
        Initialize dengan config yang lebih robust
        """
        self.db = DatabaseConnector(db_config)
        if not self.db.connect():
            print("⚠️  Warning: Could not connect to MySQL, using fallback mode")
            self.connection_ok = False
        else:
            self.connection_ok = True
    
    def match_recipes(self, nlp_output: Dict, top_k: int = 5) -> List[Dict]:
        """
        Enhanced recipe matching dengan multi-stage filtering
        """
        # Extract criteria
        entities = nlp_output.get('entities', {})
        ingredients = entities.get('ingredients', {}).get('main', [])
        avoid_ingredients = entities.get('ingredients', {}).get('avoid', [])
        cooking_methods = entities.get('cooking_methods', [])
        taste_preferences = entities.get('taste_preferences', [])
        health_conditions = entities.get('health_conditions', [])
        time_constraint = entities.get('time_constraint')
        
        # Debug info
        print(f"[Matcher] Ingredients: {ingredients}")
        print(f"[Matcher] Avoid: {avoid_ingredients}")
        print(f"[Matcher] Methods: {cooking_methods}")
        
        # Stage 1: Database query dengan expanded keywords
        expanded_ingredients = self._expand_ingredient_keywords(ingredients)
        expanded_methods = self._expand_cooking_methods(cooking_methods)
        
        # Parse time constraint
        max_time = self._parse_time_constraint(time_constraint)
        
        # Search dengan expanded terms
        recipes = []
        if self.connection_ok:
            recipes = self.db.search_recipes(
                ingredients=expanded_ingredients,
                cooking_methods=expanded_methods,
                taste_preferences=taste_preferences,
                health_conditions=health_conditions,
                max_time=max_time,
                limit=100
            )
        
        # Jika tidak ada hasil, cari lebih luas
        if not recipes and ingredients:
            print(f"[Matcher] No results, searching broader...")
            # Cari dengan kategori saja
            category_search = []
            for ing in ingredients:
                category = self._get_ingredient_category(ing)
                if category:
                    category_search.extend(self.INGREDIENT_CATEGORIES[category]['keywords'][:2])
            
            if category_search:
                recipes = self.db.search_recipes(
                    ingredients=category_search,
                    cooking_methods=[],
                    taste_preferences=[],
                    health_conditions=[],
                    max_time=max_time,
                    limit=50
                )
        
        # Stage 2: Advanced scoring
        scored_recipes = []
        for recipe in recipes:
            # Safety check - exclude jika mengandung bahan yang dihindari
            if avoid_ingredients and self._contains_avoided_ingredients(recipe, avoid_ingredients):
                continue
                
            # Health condition safety check
            if health_conditions and not self._is_safe_for_health(recipe, health_conditions):
                continue
            
            # Calculate enhanced score
            score, score_details = self._calculate_enhanced_score(recipe, entities)
            
            # Minimum score threshold
            if score >= 40:  # 40/100 minimum
                scored_recipes.append({
                    'recipe': recipe,
                    'score': score,
                    'score_details': score_details,
                    'match_details': self._get_enhanced_match_details(recipe, entities)
                })
        
        # Stage 3: Smart ranking dengan bonus dan penalties
        if scored_recipes:
            scored_recipes = self._apply_ranking_bonuses(scored_recipes, entities)
            scored_recipes.sort(key=lambda x: x['score'], reverse=True)
        else:
            # Fallback: return top recipes dengan default score
            print(f"[Matcher] No high-score matches, returning top {top_k} recipes")
            for recipe in recipes[:top_k]:
                score, _ = self._calculate_enhanced_score(recipe, entities)
                scored_recipes.append({
                    'recipe': recipe,
                    'score': max(score, 50.0),  # Minimum 50 untuk fallback
                    'score_details': {},
                    'match_details': self._get_enhanced_match_details(recipe, entities)
                })
        
        return scored_recipes[:top_k]
    
    def _expand_ingredient_keywords(self, ingredients: List[str]) -> List[str]:
        """
        Expand ingredient keywords untuk search yang lebih baik
        Contoh: "ikan" -> ["ikan", "salmon", "tuna", "kakap", "seafood"]
        """
        expanded = set()
        
        for ing in ingredients:
            ing_lower = ing.lower()
            expanded.add(ing_lower)
            
            # Cari kategori
            for category, data in self.INGREDIENT_CATEGORIES.items():
                # Jika ingredient termasuk dalam kategori
                if ing_lower in data['keywords'] or ing_lower in data.get('related', []):
                    # Tambahkan semua keywords dari kategori
                    for keyword in data['keywords'][:3]:  # Ambil 3 keywords utama
                        expanded.add(keyword)
                    break
                # Jika ingredient mengandung kata kunci kategori
                for keyword in data['keywords']:
                    if keyword in ing_lower:
                        expanded.add(keyword)
                        for related in data['keywords'][:2]:
                            expanded.add(related)
                        break
            
            # Special case: "seafood" 
            if 'seafood' in ing_lower or 'laut' in ing_lower:
                expanded.update(['ikan', 'udang', 'cumi', 'kepiting'])
        
        return list(expanded)
    
    def _expand_cooking_methods(self, methods: List[str]) -> List[str]:
        """Expand cooking methods untuk search yang lebih baik"""
        expanded = set()
        
        for method in methods:
            method_lower = method.lower()
            expanded.add(method_lower)
            
            # Cari dalam kategori cooking methods
            for category, variants in self.COOKING_METHODS.items():
                if method_lower in variants:
                    expanded.update(variants[:2])  # Tambahkan variants
        
        return list(expanded)
    
    def _get_ingredient_category(self, ingredient: str) -> str:
        """Get ingredient category untuk semantic matching"""
        ing_lower = ingredient.lower()
        
        for category, data in self.INGREDIENT_CATEGORIES.items():
            if ing_lower in data['keywords']:
                return category
            for keyword in data['keywords']:
                if keyword in ing_lower:
                    return category
            for related in data.get('related', []):
                if related in ing_lower:
                    return category
        
        return None
    
    def _parse_time_constraint(self, time_constraint: str) -> int:
        """Parse time constraint string ke minutes"""
        if not time_constraint:
            return None
        
        if 'cepat' in time_constraint.lower() or 'cepet' in time_constraint.lower():
            return 30
        elif 'mudah' in time_constraint.lower() or 'simple' in time_constraint.lower():
            return 45  # Lebih longgar dari 'cepat'
        
        # Extract number
        match = re.search(r'(\d+)', time_constraint)
        if match:
            return int(match.group(1))
        
        return None
    
    def _contains_avoided_ingredients(self, recipe: Dict, avoid_list: List[str]) -> bool:
        """Check jika resep mengandung bahan yang dihindari"""
        recipe_ingredients = recipe.get('bahan_utama', []) + recipe.get('bahan_tambahan', [])
        
        for avoid in avoid_list:
            avoid_lower = avoid.lower()
            for ingredient in recipe_ingredients:
                if avoid_lower in ingredient.lower():
                    return True
        
        return False
    
    def _is_safe_for_health(self, recipe: Dict, health_conditions: List) -> bool:
        """Check jika resep aman untuk kondisi kesehatan"""
        tidak_cocok = recipe.get('tidak_cocok_untuk', [])
        
        for condition in health_conditions:
            if isinstance(condition, dict):
                cond_name = condition.get('name', '').lower()
            else:
                cond_name = str(condition).lower()
            
            for tidak in tidak_cocok:
                if cond_name in tidak.lower():
                    return False
        
        return True
    
    def _calculate_enhanced_score(self, recipe: Dict, entities: Dict) -> Tuple[float, Dict]:
        """
        Calculate score dengan multiple factors dan weights yang diperbaiki
        Returns: (score, score_details)
        """
        ingredients = entities.get('ingredients', {}).get('main', [])
        cooking_methods = entities.get('cooking_methods', [])
        taste_preferences = entities.get('taste_preferences', [])
        time_constraint = entities.get('time_constraint')
        
        score_details = {}
        total_score = 0.0
        
        # 1. INGREDIENT MATCH (60 points) - PALING PENTING
        ingredient_score, ing_details = self._calculate_ingredient_score(recipe, ingredients)
        total_score += ingredient_score * 60
        score_details['ingredient'] = {
            'score': ingredient_score * 60,
            'details': ing_details
        }
        
        # 2. COOKING METHOD MATCH (25 points)
        method_score, method_details = self._calculate_method_score(recipe, cooking_methods)
        total_score += method_score * 25
        score_details['method'] = {
            'score': method_score * 25,
            'details': method_details
        }
        
        # 3. TASTE PREFERENCE MATCH (10 points)
        taste_score, taste_details = self._calculate_taste_score(recipe, taste_preferences)
        total_score += taste_score * 10
        score_details['taste'] = {
            'score': taste_score * 10,
            'details': taste_details
        }
        
        # 4. TIME CONSTRAINT (5 points)
        time_score = self._calculate_time_score(recipe, time_constraint)
        total_score += time_score * 5
        score_details['time'] = time_score * 5
        
        return total_score, score_details
    
    def _calculate_ingredient_score(self, recipe: Dict, user_ingredients: List[str]) -> Tuple[float, Dict]:
        """
        Enhanced ingredient scoring dengan semantic matching
        """
        if not user_ingredients:
            return 0.0, {"reason": "No ingredients specified"}
        
        recipe_ingredients = recipe.get('bahan_utama', [])
        
        # Convert semua ke lowercase untuk matching
        user_ing_lower = [ing.lower() for ing in user_ingredients]
        recipe_ing_lower = [ing.lower() for ing in recipe_ingredients]
        
        matches = []
        match_details = []
        
        for user_ing in user_ing_lower:
            best_match = None
            best_score = 0.0
            
            # Check setiap ingredient dalam resep
            for recipe_ing in recipe_ing_lower:
                match_score = self._calculate_ingredient_match_score(user_ing, recipe_ing)
                
                if match_score > best_score:
                    best_score = match_score
                    best_match = recipe_ing
            
            if best_match and best_score > 0.3:  # Threshold untuk dianggap match
                matches.append(best_score)
                match_details.append({
                    'user': user_ing,
                    'recipe': best_match,
                    'score': best_score,
                    'type': self._get_match_type(user_ing, best_match)
                })
        
        # Calculate final score
        if not matches:
            return 0.0, {"reason": "No ingredient matches found"}
        
        # Average match score, weighted by importance
        avg_score = sum(matches) / len(user_ingredients)
        
        # Bonus jika semua ingredients match
        if len(matches) == len(user_ingredients):
            avg_score = min(avg_score * 1.2, 1.0)  # 20% bonus
        
        return avg_score, {"matches": match_details, "average_score": avg_score}
    
    def _calculate_ingredient_match_score(self, user_ing: str, recipe_ing: str) -> float:
        """
        Calculate match score antara user ingredient dan recipe ingredient
        """
        # 1. Exact match (tertinggi)
        if user_ing == recipe_ing:
            return 1.0
        
        # 2. User ingredient ada dalam recipe ingredient
        # Contoh: "ikan" dalam "ikan bakar"
        if user_ing in recipe_ing:
            return 0.9
        
        # 3. Recipe ingredient ada dalam user ingredient
        if recipe_ing in user_ing:
            return 0.8
        
        # 4. Semantic match melalui kategori
        user_category = self._get_ingredient_category(user_ing)
        recipe_category = self._get_ingredient_category(recipe_ing)
        
        if user_category and recipe_category:
            # Same category (contoh: "ikan" dan "udang" sama-sama seafood)
            if user_category == recipe_category:
                return 0.7
            
            # Related categories (contoh: "daging" dan "ayam")
            related_categories = {
                'seafood': ['seafood'],
                'poultry': ['poultry', 'red_meat'],
                'red_meat': ['red_meat', 'poultry'],
                'vegetarian': ['vegetarian']
            }
            
            if user_category in related_categories and recipe_category in related_categories[user_category]:
                return 0.6
        
        # 5. Partial word match (stemming sederhana)
        user_words = set(user_ing.split())
        recipe_words = set(recipe_ing.split())
        
        common_words = user_words.intersection(recipe_words)
        if common_words:
            return 0.4 + (len(common_words) * 0.1)
        
        # 6. No match
        return 0.0
    
    def _get_match_type(self, user_ing: str, recipe_ing: str) -> str:
        """Get match type untuk display"""
        if user_ing == recipe_ing:
            return "exact"
        elif user_ing in recipe_ing or recipe_ing in user_ing:
            return "partial"
        else:
            user_cat = self._get_ingredient_category(user_ing)
            recipe_cat = self._get_ingredient_category(recipe_ing)
            if user_cat and recipe_cat and user_cat == recipe_cat:
                return "category"
            return "semantic"
    
    def _calculate_method_score(self, recipe: Dict, user_methods: List[str]) -> Tuple[float, Dict]:
        """Calculate cooking method match score"""
        if not user_methods:
            return 0.0, {"reason": "No methods specified"}
        
        recipe_methods = recipe.get('teknik_masak', [])
        user_methods_lower = [m.lower() for m in user_methods]
        recipe_methods_lower = [m.lower() for m in recipe_methods]
        
        matches = []
        match_details = []
        
        for user_method in user_methods_lower:
            best_match = None
            best_score = 0.0
            
            for recipe_method in recipe_methods_lower:
                # Exact match
                if user_method == recipe_method:
                    best_score = 1.0
                    best_match = recipe_method
                    break
                # Partial match
                elif user_method in recipe_method or recipe_method in user_method:
                    score = 0.8
                    if score > best_score:
                        best_score = score
                        best_match = recipe_method
                # Category match
                elif self._are_methods_related(user_method, recipe_method):
                    score = 0.6
                    if score > best_score:
                        best_score = score
                        best_match = recipe_method
            
            if best_match and best_score > 0:
                matches.append(best_score)
                match_details.append({
                    'user': user_method,
                    'recipe': best_match,
                    'score': best_score
                })
        
        if not matches:
            return 0.0, {"reason": "No method matches found"}
        
        avg_score = sum(matches) / len(user_methods)
        return avg_score, {"matches": match_details}
    
    def _are_methods_related(self, method1: str, method2: str) -> bool:
        """Check jika cooking methods related"""
        for category, variants in self.COOKING_METHODS.items():
            if method1 in variants and method2 in variants:
                return True
        return False
    
    def _calculate_taste_score(self, recipe: Dict, user_tastes: List[str]) -> Tuple[float, Dict]:
        """Calculate taste preference match score"""
        if not user_tastes:
            return 0.0, {"reason": "No taste preferences specified"}
        
        recipe_tastes = recipe.get('kategori_rasa', [])
        user_tastes_lower = [t.lower() for t in user_tastes]
        recipe_tastes_lower = [t.lower() for t in recipe_tastes]
        
        matches = 0
        match_details = []
        
        for user_taste in user_tastes_lower:
            for recipe_taste in recipe_tastes_lower:
                if user_taste in recipe_taste or recipe_taste in user_taste:
                    matches += 1
                    match_details.append({
                        'user': user_taste,
                        'recipe': recipe_taste
                    })
                    break
        
        score = matches / len(user_tastes) if user_tastes else 0
        return score, {"matches": match_details}
    
    def _calculate_time_score(self, recipe: Dict, time_constraint: str) -> float:
        """Calculate time constraint match score"""
        if not time_constraint:
            return 0.0
        
        recipe_time = recipe.get('waktu_masak', 0)
        max_time = self._parse_time_constraint(time_constraint)
        
        if not max_time:
            return 0.0
        
        if recipe_time <= max_time:
            return 1.0
        elif recipe_time <= max_time * 1.5:
            return 0.5
        else:
            return 0.0
    
    def _apply_ranking_bonuses(self, scored_recipes: List[Dict], entities: Dict) -> List[Dict]:
        """Apply ranking bonuses untuk hasil yang lebih baik"""
        ingredients = entities.get('ingredients', {}).get('main', [])
        
        for recipe_data in scored_recipes:
            recipe = recipe_data['recipe']
            bonus = 0
            
            # Bonus untuk exact ingredient match dalam nama resep
            recipe_name = recipe.get('nama', '').lower()
            for ing in ingredients:
                if ing.lower() in recipe_name:
                    bonus += 15  # Bonus besar untuk ingredient di nama resep
            
            # Bonus untuk resep populer (jika ada field popularity)
            if recipe.get('popularitas', 0) > 7:
                bonus += 10
            
            # Bonus untuk resep mudah
            if recipe.get('tingkat_kesulitan', '') == 'mudah':
                bonus += 5
            
            recipe_data['score'] += bonus
        
        return scored_recipes
    
    def _get_enhanced_match_details(self, recipe: Dict, entities: Dict) -> Dict:
        """Get detailed match information"""
        details = {
            'matched_ingredients': [],
            'matched_methods': [],
            'matched_tastes': [],
            'safe_for_conditions': [],
            'why_matched': []
        }
        
        ingredients = entities.get('ingredients', {}).get('main', [])
        cooking_methods = entities.get('cooking_methods', [])
        taste_preferences = entities.get('taste_preferences', [])
        health_conditions = entities.get('health_conditions', [])
        
        # Ingredient matches dengan reasoning
        recipe_ingredients = recipe.get('bahan_utama', [])
        for user_ing in ingredients:
            for recipe_ing in recipe_ingredients:
                if user_ing.lower() in recipe_ing.lower():
                    details['matched_ingredients'].append({
                        'user': user_ing,
                        'recipe': recipe_ing,
                        'match_type': 'exact' if user_ing.lower() == recipe_ing.lower() else 'partial'
                    })
                    details['why_matched'].append(f"Karena mengandung {user_ing}")
                    break
        
        # Method matches
        recipe_methods = recipe.get('teknik_masak', [])
        for user_method in cooking_methods:
            for recipe_method in recipe_methods:
                if user_method.lower() in recipe_method.lower():
                    details['matched_methods'].append(recipe_method)
                    details['why_matched'].append(f"Teknik {user_method} cocok")
                    break
        
        # Taste matches
        recipe_tastes = recipe.get('kategori_rasa', [])
        for user_taste in taste_preferences:
            for recipe_taste in recipe_tastes:
                if user_taste.lower() in recipe_taste.lower():
                    details['matched_tastes'].append(recipe_taste)
                    details['why_matched'].append(f"Rasa {user_taste} cocok")
                    break
        
        # Health safety
        cocok_untuk = recipe.get('cocok_untuk', [])
        for condition in health_conditions:
            if isinstance(condition, dict):
                cond_name = condition.get('name', '')
            else:
                cond_name = str(condition)
            
            for cocok in cocok_untuk:
                if cond_name.lower() in cocok.lower():
                    details['safe_for_conditions'].append(cocok)
                    details['why_matched'].append(f"Aman untuk {cond_name}")
        
        return details
    
    def format_recipe_display(self, matched_recipe: Dict) -> str:
        """Fixed version - handle dictionary properly"""
        recipe = matched_recipe['recipe']
        score = matched_recipe['score']
        details = matched_recipe.get('match_details', {})
    
        output = []
        output.append(f"{'='*60}")
        output.append(f"📖 {recipe['nama'].upper()}")
        output.append(f"   🎯 Match Score: {score:.1f}/100")
        output.append(f"{'='*60}")
    
        # Basic info
        output.append(f"⏱️  Waktu: {recipe['waktu_masak']} menit")
        output.append(f"👨‍🍳 Tingkat: {recipe['tingkat_kesulitan'].capitalize()}")
    
        if recipe.get('kalori_per_porsi'):
            output.append(f"🔥 Kalori: {recipe['kalori_per_porsi']} per porsi")
    
        # Ingredients - SIMPLIFIED VERSION
        output.append(f"\n🥘 Bahan Utama:")
        for ing in recipe.get('bahan_utama', []):
            output.append(f"  • {ing}")
    
        # Simple match reason
        if details.get('matched_ingredients'):
            match_strs = []
            for match in details['matched_ingredients'][:3]:
                if isinstance(match, dict):
                    match_strs.append(match['user'])
                else:
                    match_strs.append(str(match))
            output.append(f"\n✅ Cocok karena mengandung: {', '.join(match_strs)}")
        return '\n'.join(output)
    
    def get_match_summary(self, matched_recipes: List[Dict]) -> str:
        """Get summary of matches untuk display cepat"""
        if not matched_recipes:
            return "Tidak ditemukan resep yang cocok."
        
        summary = []
        summary.append(f"✅ Ditemukan {len(matched_recipes)} resep yang cocok:\n")
        
        for i, match in enumerate(matched_recipes, 1):
            recipe = match['recipe']
            score = match['score']
            
            # Cari primary match reason
            match_reason = "Cocok dengan permintaan"
            if match.get('match_details', {}).get('matched_ingredients'):
                matches = match['match_details']['matched_ingredients']
                if matches:
                    if isinstance(matches[0], dict):
                        match_reason = f"Mengandung {matches[0]['user']}"
            
            summary.append(f"{i}. {recipe['nama']} (Score: {score:.1f})")
            summary.append(f"   ⏱️  {recipe['waktu_masak']} menit | 👨‍🍳 {recipe['tingkat_kesulitan']}")
            summary.append(f"   💡 {match_reason}\n")
        
        return '\n'.join(summary)
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'db'):
            self.db.disconnect()