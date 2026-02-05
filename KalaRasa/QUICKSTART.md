# Quick Start Guide - Recipe NLP Chatbot

## 🚀 5-Minute Setup

### Step 1: Install Dependencies (30 seconds)

```bash
pip install scikit-learn pandas numpy colorama
```

### Step 2: Train Model (1-2 minutes)

```bash
python train_model.py
```

Expected output:
```
Training accuracy: 1.0000
Testing accuracy: 0.8500
✓ Model saved successfully
```

### Step 3: Run Tests (1 minute)

```bash
python test_all.py
```

Expected output:
```
✓ PASS - Preprocessor
✓ PASS - NER Extractor
✓ PASS - Intent Classifier
✓ PASS - Full Pipeline
✓ PASS - JSON Output

🎉 All tests passed!
```

### Step 4: Try the Chatbot (2 minutes)

```bash
python chatbot.py
```

Try these examples:
```
You: mau masak ayam goreng yang crispy
You: aku diabetes ga boleh gula
You: cariin resep ikan yang cepat
You: exit
```

---

## 📖 Usage Examples

### Example 1: Basic Usage

```python
from src.nlp_pipeline import RecipeNLPPipeline

# Initialize
pipeline = RecipeNLPPipeline(load_models=True)

# Process query
result = pipeline.process("mau masak ayam goreng")

# Access results
print(f"Intent: {result['intent']['primary']}")
print(f"Ingredients: {result['entities']['ingredients']['main']}")
```

### Example 2: Batch Processing

```python
queries = [
    "mau masak ayam",
    "aku diabetes",
    "pengen yang pedas"
]

results = pipeline.process_batch(queries)

for result in results:
    print(result['intent']['primary'])
```

### Example 3: Get JSON Output

```python
import json

result = pipeline.process("masak ikan bakar untuk diet")
print(json.dumps(result, indent=2, ensure_ascii=False))
```

### Example 4: Extract Specific Entities

```python
result = pipeline.process("mau masak ayam tanpa santan untuk kolesterol")

# Get what to avoid
avoid = result['constraints']['must_exclude']
print(f"Must avoid: {avoid}")

# Get health restrictions
restrictions = result['constraints']['dietary_restrictions']
for restriction in restrictions:
    print(f"Condition: {restriction['condition']}")
    print(f"Avoid: {restriction['avoid']}")
```

---

## 🎯 Common Use Cases

### Use Case 1: Recipe Search Filter

```python
def search_recipes(user_query):
    result = pipeline.process(user_query)
    
    filters = {
        'ingredients': result['entities']['ingredients']['main'],
        'exclude': result['constraints']['must_exclude'],
        'cooking_methods': result['entities']['cooking_methods'],
        'health_safe': result['constraints']['dietary_restrictions']
    }
    
    return filters  # Send to recipe database
```

### Use Case 2: Dietary Restriction Checker

```python
def check_recipe_safe(recipe, user_query):
    result = pipeline.process(user_query)
    
    # Get all things to avoid
    avoid_list = result['constraints']['must_exclude']
    
    # Check if recipe is safe
    for ingredient in recipe['ingredients']:
        if ingredient in avoid_list:
            return False, f"Contains {ingredient}"
    
    return True, "Recipe is safe"
```

### Use Case 3: Smart Recipe Recommendation

```python
def get_recommendations(user_query):
    result = pipeline.process(user_query)
    
    recommendations = {
        'based_on_ingredients': result['entities']['ingredients']['main'],
        'cooking_style': result['entities']['cooking_methods'],
        'taste_profile': result['entities']['taste_preferences'],
        'health_considerations': result['entities']['health_conditions']
    }
    
    return recommendations
```

---

## 🐛 Troubleshooting

### Problem: "Model not found"
**Solution**: Run `python train_model.py` first

### Problem: "Low accuracy"
**Solution**: Add more training data to `data/intent_dataset.csv`

### Problem: "Entity not detected"
**Solution**: Add to knowledge base in `data/knowledge_base_*.json`

### Problem: Import errors
**Solution**: 
```bash
pip install scikit-learn pandas numpy colorama --upgrade
```

---

## 📊 Understanding Output

Sample output structure:
```json
{
  "intent": {
    "primary": "cari_resep_kompleks",
    "confidence": 0.87
  },
  "entities": {
    "ingredients": {
      "main": ["ayam"],
      "avoid": ["tepung", "santan"]
    },
    "cooking_methods": ["goreng"],
    "health_conditions": ["diabetes"]
  },
  "constraints": {
    "must_include": ["ayam"],
    "must_exclude": ["tepung", "santan", "gula"],
    "dietary_restrictions": [...]
  }
}
```

Key fields:
- **intent.primary**: What user wants to do
- **intent.confidence**: How sure (0-1)
- **entities.ingredients.main**: Main ingredients mentioned
- **entities.ingredients.avoid**: Ingredients to avoid
- **constraints.must_exclude**: All exclusions (compiled)

---

## 🔧 Customization

### Add New Ingredient

Edit `data/knowledge_base_ingredients.json`:
```json
{
  "protein": {
    "daging": ["ayam", "sapi", "YOUR_NEW_INGREDIENT"]
  }
}
```

### Add New Intent

1. Add to `config/config.py`:
```python
INTENT_LABELS = [
    'cari_resep',
    'your_new_intent'  # Add here
]
```

2. Add training data to `data/intent_dataset.csv`:
```csv
"example text for new intent",your_new_intent
```

3. Retrain:
```bash
python train_model.py
```

---

## 💾 Saving Results

### Save to JSON file

```python
import json
from datetime import datetime

result = pipeline.process("your query")

# Save with timestamp
filename = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(filename, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
```

### Save conversation history

```python
conversation = []

while True:
    user_input = input("You: ")
    result = pipeline.process(user_input)
    conversation.append(result)
    
# Save entire conversation
with open('conversation.json', 'w') as f:
    json.dump(conversation, f, indent=2, ensure_ascii=False)
```

---

## 🎓 Next Steps

1. **Improve Accuracy**: Add more training data
2. **Expand Knowledge Base**: Add more ingredients, conditions
3. **Build Recipe Matcher**: Use output to match with recipes
4. **Add API**: Create REST API with Flask/FastAPI
5. **Add Database**: Connect to real recipe database

---

## 📞 Support

- Run `python demo.py` for interactive examples
- Check `README.md` for full documentation
- Test with `python test_all.py`

**Happy Cooking! 🍳**
