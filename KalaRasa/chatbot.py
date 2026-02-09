#!/usr/bin/env python3
# chatbot.py
# Main chatbot interface

import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import json
from src.enhanced_nlp_engine import EnhancedNLPEngine
from src.recipe_matcher import RecipeMatcher
from src.recipe_matcher_mysql import RecipeMatcherMySQL
from colorama import init, Fore, Style

# Initialize colorama untuk colored terminal output
init(autoreset=True)


class RecipeChatbot:
    """
    Interactive Recipe Chatbot
    Menggunakan NLP Pipeline untuk memproses input user
    Dan Recipe Matcher untuk menemukan resep yang sesuai
    """
    
    def __init__(self):
        """Initialize chatbot dengan NLP pipeline dan recipe matcher"""
        print(f"{Fore.CYAN}Initializing Recipe Chatbot...{Style.RESET_ALL}")
        try:
            self.pipeline = EnhancedNLPEngine()
            # Gunakan salah satu matcher, bukan keduanya
            # self.matcher = RecipeMatcher()  # Non-MySQL
            self.matcher = RecipeMatcherMySQL()  # MySQL
            self.conversation_history = []
            print(f"{Fore.GREEN}✓ Chatbot ready!{Style.RESET_ALL}\n")
        except Exception as e:
            print(f"{Fore.RED}✗ Error initializing chatbot: {e}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Please run train_model.py first!{Style.RESET_ALL}")
            sys.exit(1)
    
    def print_welcome(self):
        """Print welcome message"""
        print("="*80)
        print(f"{Fore.CYAN}{'RECIPE CHATBOT':^80}{Style.RESET_ALL}")
        print("="*80)
        print(f"\n{Fore.YELLOW}Selamat datang di Recipe Chatbot!{Style.RESET_ALL}")
        print("Saya dapat membantu Anda menemukan resep berdasarkan:")
        print("  • Bahan yang ingin digunakan")
        print("  • Teknik memasak yang diinginkan")
        print("  • Kondisi kesehatan dan pantangan")
        print("  • Preferensi rasa")
        print(f"\n{Fore.CYAN}Contoh pertanyaan:{Style.RESET_ALL}")
        print('  - "mau masak ayam goreng yang krispy"')
        print('  - "aku diabetes jadi ga boleh makan manis"')
        print('  - "cariin resep ikan bakar yang cepat"')
        print(f'\n{Fore.YELLOW}Ketik "exit" atau "quit" untuk keluar{Style.RESET_ALL}')
        print("="*80)
        print()
    
    def format_response(self, result: dict, matched_recipes: list = None) -> str:
        """Format hasil NLP pipeline dan matched recipes menjadi response yang user-friendly"""
        
        # ===== HANDLE FORMAT BARU & LAMA =====
        if isinstance(result.get('intent'), dict):
            # Format BARU
            intent = result['intent'].get('primary', 'unknown')
            confidence = result['intent'].get('confidence', 0.0)
        else:
            # Format LAMA / fallback
            intent = result.get('intent', 'unknown')
            confidence = result.get('confidence', 0.0)

        entities = result.get('entities', {}) or {}
        status = result.get('status', 'ok')
        message = result.get('message', '')

        response = []

        # ===== PESAN FALLBACK =====
        if status != "ok":
            response.append(f"{Fore.YELLOW}{message}{Style.RESET_ALL}")
            return '\n'.join(response)
        
        # Header
        response.append(f"{Fore.GREEN}Saya mengerti!{Style.RESET_ALL}")
        
        # Intent detection
        if confidence > 0.7:
            conf_color = Fore.GREEN
        elif confidence > 0.4:
            conf_color = Fore.YELLOW
        else:
            conf_color = Fore.RED
        
        response.append(f"Intent terdeteksi: {conf_color}{intent}{Style.RESET_ALL} (confidence: {confidence:.2%})")
        
        # Main ingredients
        if entities.get('ingredients', {}).get('main'):
            ingredients_str = ', '.join(entities['ingredients']['main'])
            response.append(f"\n{Fore.CYAN}Bahan utama:{Style.RESET_ALL} {ingredients_str}")
        
        # Ingredients to avoid
        if entities.get('ingredients', {}).get('avoid'):
            avoid_list = entities['ingredients']['avoid']
            avoid_str = ', '.join(avoid_list[:5])
            more = f" (+{len(avoid_list)-5} more)" if len(avoid_list) > 5 else ""
            response.append(f"{Fore.RED}Hindari:{Style.RESET_ALL} {avoid_str}{more}")
        
        # Cooking methods
        if entities.get('cooking_methods'):
            methods_str = ', '.join(entities['cooking_methods'])
            response.append(f"{Fore.CYAN}Teknik memasak:{Style.RESET_ALL} {methods_str}")
        
        # Health conditions
        if entities.get('health_conditions'):
            conditions_str = ', '.join(entities['health_conditions'])
            response.append(f"{Fore.YELLOW}Kondisi kesehatan:{Style.RESET_ALL} {conditions_str}")
        
        # Taste preferences
        if entities.get('taste_preferences'):
            taste_str = ', '.join(entities['taste_preferences'])
            response.append(f"{Fore.MAGENTA}Preferensi rasa:{Style.RESET_ALL} {taste_str}")
        
        # Time constraint
        if entities.get('time_constraint'):
            response.append(f"{Fore.CYAN}Batasan waktu:{Style.RESET_ALL} {entities['time_constraint']}")
        
        # Matched recipes section
        response.append(f"\n{'='*80}")
        
        if matched_recipes and len(matched_recipes) > 0:
            response.append(f"{Fore.GREEN}✓ Menemukan {len(matched_recipes)} resep yang cocok:{Style.RESET_ALL}\n")
            
            for i, matched in enumerate(matched_recipes, 1):
                recipe = matched['recipe']
                score = matched['score']
                
                response.append(f"\n{Fore.CYAN}#{i}. {recipe['nama']}{Style.RESET_ALL} (Score: {score:.1f}/100)")
                response.append(f"   ⏱️  {recipe['waktu_masak']} menit | 👨‍🍳 {recipe['tingkat_kesulitan']} | 🔥 {recipe['kalori_per_porsi']} kal")
                
                # Show why it matched
                details = matched['match_details']
                if details.get('matched_ingredients'):
                    response.append(f"   ✓ Bahan: {', '.join(details['matched_ingredients'][:3])}")
                if details.get('matched_methods'):
                    response.append(f"   ✓ Teknik: {', '.join(details['matched_methods'])}")
                if details.get('matched_tastes'):
                    response.append(f"   ✓ Rasa: {', '.join(details['matched_tastes'])}")
        else:
            response.append(f"{Fore.YELLOW}⚠️  Tidak ada resep yang sesuai dengan kriteria Anda.{Style.RESET_ALL}")
            response.append(f"{Fore.YELLOW}   Coba ubah kriteria atau tambah lebih banyak resep ke database.{Style.RESET_ALL}")
        
        return '\n'.join(response)
    
    def save_output(self, result: dict, filename: str = None):
        """Save output JSON to file"""
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"output_{timestamp}.json"
        
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"{Fore.GREEN}Output saved to: {filepath}{Style.RESET_ALL}")
    
    def process_user_input(self, user_input: str):
        """Process one user input and return result"""
        print(f"\n{Fore.YELLOW}Processing...{Style.RESET_ALL}")
        result = self.pipeline.process(user_input)
        
        # Jika NLP gagal memahami, return error
        if result.get("status") != "ok":
            return {
                "success": False,
                "message": result.get("message", "Maaf, aku belum paham."),
                "raw_result": result
            }
        
        print(f"{Fore.YELLOW}Searching recipes...{Style.RESET_ALL}")
        matched_recipes = self.matcher.match_recipes(result, top_k=5)
        
        return {
            "success": True,
            "raw_result": result,
            "matched_recipes": matched_recipes
        }
    
    def run(self):
        """Main chatbot loop dengan flow yang lebih natural"""
        self.print_welcome()
        
        while True:
            try:
                # Get user input
                user_input = input(f"{Fore.CYAN}You:{Style.RESET_ALL} ").strip()
                
                # Check for exit commands
                if user_input.lower() in ['exit', 'quit', 'bye', 'keluar']:
                    print(f"\n{Fore.YELLOW}Terima kasih telah menggunakan Recipe Chatbot!{Style.RESET_ALL}")
                    print(f"{Fore.YELLOW}Sampai jumpa! 👋{Style.RESET_ALL}\n")
                    break
                
                # Skip empty input
                if not user_input:
                    continue
                
                # Special commands
                if user_input.lower() == 'help':
                    self.print_welcome()
                    continue
                
                if user_input.lower().startswith('save'):
                    if self.conversation_history:
                        self.save_output(self.conversation_history[-1])
                    else:
                        print(f"{Fore.RED}No conversation to save yet!{Style.RESET_ALL}")
                    continue
                
                # Proses input user
                process_result = self.process_user_input(user_input)
                
                if not process_result["success"]:
                    # Jika NLP tidak paham, tampilkan pesan error dan minta input ulang
                    print(f"\n{Fore.CYAN}Bot:{Style.RESET_ALL}")
                    print(process_result["message"])
                    print(f"{Fore.YELLOW}Coba tanyakan dengan cara lain, atau contoh di atas.{Style.RESET_ALL}")
                    print("\n" + "-"*80 + "\n")
                    continue
                
                # Jika berhasil dipahami, tampilkan hasil
                result = process_result["raw_result"]
                matched_recipes = process_result["matched_recipes"]
                
                # Store in history
                self.conversation_history.append({
                    'nlp_output': result,
                    'matched_recipes': matched_recipes
                })
                
                # Format and display response
                print(f"\n{Fore.CYAN}Bot:{Style.RESET_ALL}")
                print(self.format_response(result, matched_recipes))
                
                # Option to see detail of a recipe
                if matched_recipes and len(matched_recipes) > 0:
                    while True:
                        see_detail = input(f"\n{Fore.YELLOW}Lihat detail resep? (ketik nomor 1-{len(matched_recipes)}, 'ulang' untuk input baru, atau tekan Enter untuk skip):{Style.RESET_ALL} ").strip()
                        
                        if see_detail.lower() == 'ulang':
                            print(f"{Fore.CYAN}Silakan tanyakan lagi...{Style.RESET_ALL}")
                            break  # Keluar dari loop detail, kembali ke input utama
                        
                        if not see_detail:
                            # Skip melihat detail
                            break
                        
                        if see_detail.isdigit() and 1 <= int(see_detail) <= len(matched_recipes):
                            idx = int(see_detail) - 1
                            print(f"\n{Fore.CYAN}Detail Resep:{Style.RESET_ALL}")
                            print(self.matcher.format_recipe_display(matched_recipes[idx]))
                            
                            # Tanya apakah ingin melihat detail resep lain
                            another = input(f"\n{Fore.YELLOW}Lihat detail resep lain? (y/n):{Style.RESET_ALL} ").strip().lower()
                            if another != 'y':
                                break
                        else:
                            print(f"{Fore.RED}Input tidak valid. Masukkan angka 1-{len(matched_recipes)} atau 'ulang'.{Style.RESET_ALL}")
                
                # Option to see raw JSON
                see_json = input(f"\n{Fore.YELLOW}Lihat JSON output? (y/n):{Style.RESET_ALL} ").strip().lower()
                if see_json == 'y':
                    print(f"\n{Fore.CYAN}JSON Output:{Style.RESET_ALL}")
                    print(json.dumps(result, indent=2, ensure_ascii=False))
                
                print("\n" + "-"*80 + "\n")
                print(f"{Fore.CYAN}Apa lagi yang bisa saya bantu?{Style.RESET_ALL}")
                
            except KeyboardInterrupt:
                print(f"\n\n{Fore.YELLOW}Interrupted by user. Exiting...{Style.RESET_ALL}")
                break
            except Exception as e:
                print(f"\n{Fore.RED}Error: {e}{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}Silakan coba lagi.{Style.RESET_ALL}\n")


def main():
    """Main function"""
    chatbot = RecipeChatbot()
    chatbot.run()


if __name__ == "__main__":
    main()