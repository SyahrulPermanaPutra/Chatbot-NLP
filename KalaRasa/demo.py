#!/usr/bin/env python3
# demo.py
# Demo script untuk menunjukkan capabilities sistem

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
from src.nlp_pipeline import RecipeNLPPipeline
from colorama import init, Fore, Style

init(autoreset=True)


def print_header(text):
    """Print colored header"""
    print(f"\n{Fore.CYAN}{'='*80}")
    print(f"{text:^80}")
    print(f"{'='*80}{Style.RESET_ALL}\n")


def print_result(text, result):
    """Print formatted result"""
    intent = result['intent']['primary']
    confidence = result['intent']['confidence']
    entities = result['entities']
    
    print(f"{Fore.YELLOW}Input:{Style.RESET_ALL} {text}")
    print(f"{Fore.GREEN}Intent:{Style.RESET_ALL} {intent} ({confidence:.1%})")
    
    if entities['ingredients']['main']:
        print(f"{Fore.CYAN}Main Ingredients:{Style.RESET_ALL} {', '.join(entities['ingredients']['main'])}")
    
    if entities['ingredients']['avoid']:
        avoid_list = entities['ingredients']['avoid'][:5]  # Limit display
        more = f" (+{len(entities['ingredients']['avoid'])-5} more)" if len(entities['ingredients']['avoid']) > 5 else ""
        print(f"{Fore.RED}Must Avoid:{Style.RESET_ALL} {', '.join(avoid_list)}{more}")
    
    if entities['cooking_methods']:
        print(f"{Fore.MAGENTA}Cooking Methods:{Style.RESET_ALL} {', '.join(entities['cooking_methods'])}")
    
    if entities['health_conditions']:
        print(f"{Fore.YELLOW}Health Conditions:{Style.RESET_ALL} {', '.join(entities['health_conditions'])}")
    
    if entities['taste_preferences']:
        print(f"{Fore.GREEN}Taste Preferences:{Style.RESET_ALL} {', '.join(entities['taste_preferences'])}")
    
    print()


def demo_basic_queries():
    """Demo 1: Basic recipe queries"""
    print_header("DEMO 1: BASIC RECIPE QUERIES")
    
    pipeline = RecipeNLPPipeline(load_models=True)
    
    queries = [
        "mau masak ayam goreng",
        "cariin resep ikan bakar",
        "pengen bikin tumis kangkung",
        "ada resep sayur asem ga"
    ]
    
    for query in queries:
        result = pipeline.process(query)
        print_result(query, result)


def demo_health_conditions():
    """Demo 2: Health conditions and dietary restrictions"""
    print_header("DEMO 2: HEALTH CONDITIONS & DIETARY RESTRICTIONS")
    
    pipeline = RecipeNLPPipeline(load_models=True)
    
    queries = [
        "aku diabetes jadi ga boleh gula",
        "kolesterol tinggi ga bisa santan",
        "asam urat ga boleh bayam dan melinjo",
        "alergi seafood jadi ga bisa ikan dan udang",
        "vegetarian ga makan daging"
    ]
    
    for query in queries:
        result = pipeline.process(query)
        print_result(query, result)


def demo_complex_queries():
    """Demo 3: Complex multi-constraint queries"""
    print_header("DEMO 3: COMPLEX MULTI-CONSTRAINT QUERIES")
    
    pipeline = RecipeNLPPipeline(load_models=True)
    
    queries = [
        "gw pengen masak ayam goreng yang krispy tapi gak pake tepung untuk diet",
        "mau bikin pasta carbonara tapi dairy free gimana caranya",
        "cariin resep ikan yang cepat, ga pake santan, buat diabetes",
        "pengen masak sayur yang pedas gurih tapi ga boleh pake msg",
        "ada resep ayam yang direbus aja biar sehat, tanpa garam untuk hipertensi"
    ]
    
    for query in queries:
        result = pipeline.process(query)
        print_result(query, result)


def demo_informal_language():
    """Demo 4: Informal language and slang"""
    print_header("DEMO 4: INFORMAL LANGUAGE & SLANG HANDLING")
    
    pipeline = RecipeNLPPipeline(load_models=True)
    
    queries = [
        "gw mau masak aym gorng yg krispy bgt",
        "pgn bikin pasta tp dairy free gmn",
        "lg pengen yg pedes pedes deh",
        "ada resep yg simple aja ga ribet",
        "mau masak yg cpt doang, 15 mnit aja"
    ]
    
    for query in queries:
        result = pipeline.process(query)
        print_result(query, result)


def demo_json_output():
    """Demo 5: JSON output format"""
    print_header("DEMO 5: JSON OUTPUT FORMAT")
    
    pipeline = RecipeNLPPipeline(load_models=True)
    
    query = "mau masak ayam goreng yang krispy untuk diet tanpa tepung"
    result = pipeline.process(query)
    
    print(f"{Fore.YELLOW}Input:{Style.RESET_ALL} {query}\n")
    print(f"{Fore.CYAN}JSON Output:{Style.RESET_ALL}")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def demo_performance():
    """Demo 6: Performance metrics"""
    print_header("DEMO 6: PERFORMANCE METRICS")
    
    import time
    
    pipeline = RecipeNLPPipeline(load_models=True)
    
    test_queries = [
        "mau masak ayam goreng",
        "aku diabetes ga boleh gula",
        "pengen yang pedas gurih direbus",
        "cariin resep ikan yang cepat",
        "mau bikin pasta dairy free"
    ]
    
    times = []
    for query in test_queries:
        start = time.time()
        result = pipeline.process(query)
        elapsed = time.time() - start
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"{Fore.GREEN}Performance Statistics:{Style.RESET_ALL}")
    print(f"  Total queries: {len(test_queries)}")
    print(f"  Average time: {avg_time*1000:.2f}ms")
    print(f"  Min time: {min_time*1000:.2f}ms")
    print(f"  Max time: {max_time*1000:.2f}ms")
    print(f"  Throughput: {len(test_queries)/sum(times):.1f} queries/second")


def main():
    """Run all demos"""
    print(f"\n{Fore.CYAN}{'='*80}")
    print(f"{'RECIPE NLP CHATBOT - COMPREHENSIVE DEMO':^80}")
    print(f"{'='*80}{Style.RESET_ALL}")
    
    demos = [
        ("1", "Basic Recipe Queries", demo_basic_queries),
        ("2", "Health Conditions", demo_health_conditions),
        ("3", "Complex Queries", demo_complex_queries),
        ("4", "Informal Language", demo_informal_language),
        ("5", "JSON Output", demo_json_output),
        ("6", "Performance", demo_performance)
    ]
    
    print(f"\n{Fore.YELLOW}Available Demos:{Style.RESET_ALL}")
    for num, name, _ in demos:
        print(f"  {num}. {name}")
    print(f"  0. Run all demos")
    print(f"  q. Quit")
    
    while True:
        choice = input(f"\n{Fore.CYAN}Select demo (0-6, q to quit):{Style.RESET_ALL} ").strip()
        
        if choice.lower() == 'q':
            print(f"\n{Fore.YELLOW}Exiting demo. Goodbye!{Style.RESET_ALL}\n")
            break
        
        if choice == '0':
            for num, name, func in demos:
                func()
                input(f"\n{Fore.YELLOW}Press Enter to continue...{Style.RESET_ALL}")
        elif choice in ['1', '2', '3', '4', '5', '6']:
            idx = int(choice) - 1
            demos[idx][2]()
        else:
            print(f"{Fore.RED}Invalid choice. Please select 0-6 or q.{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
