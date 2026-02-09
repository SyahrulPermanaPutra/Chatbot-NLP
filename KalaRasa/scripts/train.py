#!/usr/bin/env python3
# scripts/generate_large_dataset.py
# Generate comprehensive training dataset

import csv
import random

# Template-based data generation
templates = {
    'cari_resep': [
        "mau masak {ingredient}",
        "pengen bikin {ingredient}",
        "cariin resep {ingredient}",
        "ada resep {ingredient} ga",
        "gimana cara masak {ingredient}",
        "resep {ingredient} yang enak",
        "mau masak {ingredient} yang {method}",
        "pengen {ingredient} {method}",
        "bikin {ingredient} pake {method}",
        "resep {ingredient} {taste}",
        "{ingredient} yang {taste} dong",
    ],
    'cari_resep_kompleks': [
        "mau masak {ingredient} yang {method} tapi tanpa {avoid}",
        "pengen {ingredient} {taste} ga pake {avoid}",
        "resep {ingredient} {method} yang {taste}",
        "{ingredient} {method} tanpa {avoid} untuk {condition}",
        "mau {ingredient} yang {taste} dan {method} ga boleh {avoid}",
        "bikin {ingredient} {method} buat {condition} tanpa {avoid}",
    ],
    'informasi_kondisi_kesehatan': [
        "aku {condition}",
        "saya punya {condition}",
        "lagi {condition} nih",
        "ada {condition} ga boleh apa",
        "{condition} pantangannya apa",
        "punya {condition} boleh makan apa",
    ],
    'tanya_alternatif': [
        "{ingredient} bisa diganti apa",
        "ga ada {ingredient} bisa pake apa",
        "substitusi {ingredient} dengan apa",
        "pengganti {ingredient}",
        "alternatif {ingredient} apa ya",
    ],
    'tanya_informasi': [
        "cara {method} yang benar gimana",
        "bedanya {method1} sama {method2}",
        "{ingredient} kandungannya apa",
        "kalori {ingredient} berapa",
        "nutrisi {ingredient}",
    ]
}

ingredients = [
    "ayam", "ikan", "sapi", "udang", "cumi", "tempe", "tahu",
    "bayam", "kangkung", "brokoli", "wortel", "kentang", "jagung",
    "nasi", "mie", "pasta", "bihun", "kwetiau",
    "telur", "telor", "salmon", "tuna", "daging", "kambing"
]

methods = [
    "goreng", "rebus", "kukus", "bakar", "panggang", "tumis",
    "oseng", "pepes", "rica rica", "balado", "rendang", "gulai"
]

tastes = [
    "pedas", "manis", "gurih", "asam", "asin", "segar",
    "renyah", "crispy", "lembut", "empuk"
]

avoids = [
    "santan", "gula", "garam", "msg", "minyak", "tepung",
    "mentega", "keju", "susu", "kacang"
]

conditions = [
    "diabetes", "kolesterol", "asam urat", "hipertensi", "maag",
    "diet", "alergi", "vegetarian"
]

def generate_dataset(output_file: str, num_samples: int = 500):
    """Generate large training dataset"""
    
    data = []
    
    # Generate from templates
    for intent, template_list in templates.items():
        for template in template_list:
            # Generate variations
            for _ in range(20):
                text = template.format(
                    ingredient=random.choice(ingredients),
                    method=random.choice(methods),
                    taste=random.choice(tastes),
                    avoid=random.choice(avoids),
                    condition=random.choice(conditions),
                    method1=random.choice(methods),
                    method2=random.choice(methods)
                )
                data.append((text, intent))
    
    # Add manual examples
    manual_data = [
        ("halo", "chitchat"),
        ("hai", "chitchat"),
        ("terima kasih", "chitchat"),
        ("thanks", "chitchat"),
        ("oke", "chitchat"),
        ("baik", "chitchat"),
        ("siap", "chitchat"),
        ("mau dong", "chitchat"),
        ("boleh", "chitchat"),
        ("gimana caranya", "tanya_informasi"),
        ("kenapa harus begitu", "tanya_informasi"),
        ("apa bedanya", "tanya_informasi"),
        ("berapa lama", "tanya_informasi"),
        ("sampai kapan", "tanya_informasi"),
    ]
    
    data.extend(manual_data)
    
    # Shuffle and limit
    random.shuffle(data)
    data = data[:num_samples]
    
    # Write to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'intent'])
        writer.writerows(data)
    
    print(f"Generated {len(data)} training examples")
    print(f"Saved to {output_file}")
    
    # Show distribution
    from collections import Counter
    intent_counts = Counter([intent for _, intent in data])
    print("\nIntent distribution:")
    for intent, count in intent_counts.most_common():
        print(f"  {intent}: {count}")

if __name__ == "__main__":
    generate_dataset('../data/large_intent_dataset.csv', num_samples=1000)
