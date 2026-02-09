# src/database_connector.py
# MySQL Database Connector untuk Recipe Chatbot

import mysql.connector
from mysql.connector import Error
from typing import List, Dict, Optional
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DatabaseConnector:
    """
    MySQL Database Connector untuk Recipe Chatbot
    Handles all database operations
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize database connection
        
        Args:
            config: Dict dengan host, user, password, database
        """
        if config is None:
            # Default configuration
            config = {
                'host': 'localhost',
                'user': 'root',
                'password': '',  
                'database': 'kala_rasa_jtv',
                'charset': 'utf8mb4',
                'use_unicode': True
            }
        
        self.config = config
        self.connection = None
        self.cursor = None
    
    def connect(self):
        """Establish database connection"""
        try:
            self.connection = mysql.connector.connect(**self.config)
            if self.connection.is_connected():
                self.cursor = self.connection.cursor(dictionary=True)
                print(f"✓ Connected to MySQL database: {self.config['database']}")
                return True
        except Error as e:
            print(f"✗ Error connecting to MySQL: {e}")
            return False
    
    def disconnect(self):
        """Close database connection"""
        if self.connection and self.connection.is_connected():
            self.cursor.close()
            self.connection.close()
            print("✓ MySQL connection closed")
    
    def search_recipes(
        self,
        ingredients: List[str] = None,
        cooking_methods: List[str] = None,
        taste_preferences: List[str] = None,
        health_conditions: List[str] = None,
        max_time: int = None,
        difficulty: str = None,
        limit: int = 20
    ) -> List[Dict]:
        """
        Search recipes berdasarkan kriteria
        
        Returns:
            List of recipe dictionaries
        """
        try:
            query = """
                SELECT DISTINCT
                    r.id,
                    r.nama,
                    r.tingkat_kesulitan,
                    r.waktu_masak,
                    r.kalori_per_porsi,
                    r.region
                FROM recipes r
            """
            
            joins = []
            conditions = []
            params = []
            
            # Filter by ingredients
            if ingredients and len(ingredients) > 0:
                joins.append("""
                    INNER JOIN recipe_ingredients ri ON r.id = ri.recipe_id
                    INNER JOIN ingredients i ON ri.ingredient_id = i.id
                """)
                placeholders = ','.join(['%s'] * len(ingredients))
                conditions.append(f"i.nama IN ({placeholders})")
                params.extend(ingredients)
            
            # Filter by cooking methods
            if cooking_methods and len(cooking_methods) > 0:
                joins.append("""
                    INNER JOIN recipe_cooking_methods rcm ON r.id = rcm.recipe_id
                    INNER JOIN cooking_methods cm ON rcm.cooking_method_id = cm.id
                """)
                placeholders = ','.join(['%s'] * len(cooking_methods))
                conditions.append(f"cm.nama IN ({placeholders})")
                params.extend(cooking_methods)
            
            # Filter by taste preferences
            if taste_preferences and len(taste_preferences) > 0:
                joins.append("""
                    INNER JOIN recipe_taste_profiles rtp ON r.id = rtp.recipe_id
                    INNER JOIN taste_profiles tp ON rtp.taste_profile_id = tp.id
                """)
                placeholders = ','.join(['%s'] * len(taste_preferences))
                conditions.append(f"tp.nama IN ({placeholders})")
                params.extend(taste_preferences)
            
            # Filter by health conditions (recipes that ARE suitable)
            if health_conditions and len(health_conditions) > 0:
                joins.append("""
                    LEFT JOIN recipe_suitability rs ON r.id = rs.recipe_id
                    LEFT JOIN health_conditions hc ON rs.health_condition_id = hc.id
                """)
                placeholders = ','.join(['%s'] * len(health_conditions))
                conditions.append(f"""
                    (rs.is_suitable = TRUE AND hc.nama IN ({placeholders}))
                    OR rs.id IS NULL
                """)
                params.extend(health_conditions)
            
            # Filter by max time
            if max_time:
                conditions.append("r.waktu_masak <= %s")
                params.append(max_time)
            
            # Filter by difficulty
            if difficulty:
                conditions.append("r.tingkat_kesulitan = %s")
                params.append(difficulty)
            
            # Build final query
            if joins:
                query += " ".join(joins)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += f" LIMIT {limit}"
            
            # Execute
            self.cursor.execute(query, params)
            recipes = self.cursor.fetchall()
            
            # Enrich each recipe with details
            enriched_recipes = []
            for recipe in recipes:
                enriched = self.get_recipe_details(recipe['id'])
                enriched_recipes.append(enriched)
            
            return enriched_recipes
            
        except Error as e:
            print(f"Error searching recipes: {e}")
            return []
    
    def get_recipe_details(self, recipe_id: int) -> Dict:
        """
        Get complete recipe details including ingredients, methods, taste
        """
        try:
            # Basic recipe info
            self.cursor.execute("""
                SELECT * FROM recipes WHERE id = %s
            """, (recipe_id,))
            recipe = self.cursor.fetchone()
            
            if not recipe:
                return None
            
            # Get ingredients
            self.cursor.execute("""
                SELECT 
                    i.nama,
                    ri.is_main,
                    ri.jumlah,
                    i.kategori
                FROM recipe_ingredients ri
                JOIN ingredients i ON ri.ingredient_id = i.id
                WHERE ri.recipe_id = %s
            """, (recipe_id,))
            
            ingredients = self.cursor.fetchall()
            recipe['bahan_utama'] = [
                ing['nama'] for ing in ingredients if ing['is_main']
            ]
            recipe['bahan_tambahan'] = [
                ing['nama'] for ing in ingredients if not ing['is_main']
            ]
            
            # Get cooking methods
            self.cursor.execute("""
                SELECT cm.nama
                FROM recipe_cooking_methods rcm
                JOIN cooking_methods cm ON rcm.cooking_method_id = cm.id
                WHERE rcm.recipe_id = %s
            """, (recipe_id,))
            
            methods = self.cursor.fetchall()
            recipe['teknik_masak'] = [m['nama'] for m in methods]
            
            # Get taste profiles
            self.cursor.execute("""
                SELECT tp.nama
                FROM recipe_taste_profiles rtp
                JOIN taste_profiles tp ON rtp.taste_profile_id = tp.id
                WHERE rtp.recipe_id = %s
            """, (recipe_id,))
            
            tastes = self.cursor.fetchall()
            recipe['kategori_rasa'] = [t['nama'] for t in tastes]
            
            # Get suitability (cocok untuk / tidak cocok untuk)
            self.cursor.execute("""
                SELECT 
                    hc.nama,
                    rs.is_suitable
                FROM recipe_suitability rs
                JOIN health_conditions hc ON rs.health_condition_id = hc.id
                WHERE rs.recipe_id = %s
            """, (recipe_id,))
            
            suitability = self.cursor.fetchall()
            recipe['cocok_untuk'] = [
                s['nama'] for s in suitability if s['is_suitable']
            ]
            recipe['tidak_cocok_untuk'] = [
                s['nama'] for s in suitability if not s['is_suitable']
            ]
            
            return recipe
            
        except Error as e:
            print(f"Error getting recipe details: {e}")
            return None
    
    def get_restricted_ingredients(self, health_condition: str) -> List[str]:
        """
        Get list of ingredients to avoid for a health condition
        """
        try:
            self.cursor.execute("""
                SELECT i.nama
                FROM health_condition_restrictions hcr
                JOIN health_conditions hc ON hcr.health_condition_id = hc.id
                JOIN ingredients i ON hcr.ingredient_id = i.id
                WHERE hc.nama = %s AND hcr.severity = 'hindari'
            """, (health_condition,))
            
            restrictions = self.cursor.fetchall()
            return [r['nama'] for r in restrictions]
            
        except Error as e:
            print(f"Error getting restrictions: {e}")
            return []
    
    def log_user_query(
        self,
        query_text: str,
        intent: str,
        confidence: float,
        status: str,
        entities: Dict
    ) -> int:
        """
        Log user query for analytics
        
        Returns:
            query_id
        """
        try:
            self.cursor.execute("""
                INSERT INTO user_queries 
                (query_text, intent, confidence, status, entities)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                query_text,
                intent,
                confidence,
                status,
                json.dumps(entities, ensure_ascii=False)
            ))
            
            self.connection.commit()
            return self.cursor.lastrowid
            
        except Error as e:
            print(f"Error logging query: {e}")
            return None
    
    def log_matched_recipes(
        self,
        user_query_id: int,
        matched_recipes: List[Dict]
    ):
        """
        Log matched recipes for analytics
        """
        try:
            for i, match in enumerate(matched_recipes, 1):
                self.cursor.execute("""
                    INSERT INTO matched_recipes
                    (user_query_id, recipe_id, match_score, rank_position)
                    VALUES (%s, %s, %s, %s)
                """, (
                    user_query_id,
                    match['recipe']['id'],
                    match['score'],
                    i
                ))
            
            self.connection.commit()
            
        except Error as e:
            print(f"Error logging matched recipes: {e}")


if __name__ == "__main__":
    # Testing
    print("=== Database Connector Test ===\n")
    
    # Connect
    db = DatabaseConnector({
        'host': 'localhost',
        'user': 'root',
        'password': '',  
        'database': 'kala_rasa_jtv'
    })
    
    if db.connect():
        # Test 1: Search by ingredient
        print("\n[Test 1] Search recipes with 'ayam':")
        recipes = db.search_recipes(ingredients=['ayam'], limit=5)
        for r in recipes:
            print(f"  - {r['nama']}")
        
        # Test 2: Search by cooking method
        print("\n[Test 2] Search recipes with 'tumis':")
        recipes = db.search_recipes(cooking_methods=['tumis'], limit=5)
        for r in recipes:
            print(f"  - {r['nama']}")
        
        # Test 3: Get restrictions
        print("\n[Test 3] Get restrictions for diabetes:")
        restrictions = db.get_restricted_ingredients('diabetes')
        print(f"  Avoid: {', '.join(restrictions[:5])}")
        
        # Test 4: Get recipe details
        print("\n[Test 4] Get details for recipe ID 1:")
        details = db.get_recipe_details(1)
        if details:
            print(f"  Nama: {details['nama']}")
            print(f"  Bahan utama: {', '.join(details['bahan_utama'])}")
            print(f"  Teknik: {', '.join(details['teknik_masak'])}")
        
        db.disconnect()
    else:
        print("Failed to connect to database")
