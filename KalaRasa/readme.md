Setup : 
1. Aktivin venv di folder Chatbot-NLP 
Windows
venv\Scripts\activate
Linux :
source venv/bin/activate
2.  Cd Kalarasa terus pip install -r requirements.txt
3. Copy env.exampple
3. Lanjut ke Pre-Deploy

Pre-Deploy
•	Pastikan Flask service berjalan: curl http://localhost:5000/health
•	Jalankan CBR index build: php artisan cbr:rebuild-index di laravel
•	Validasi kamus NLP: python src/validate_dict.py --all
•	Validasi Engine dengan python test.py
•	Pastikan Redis berjalan dan dapat diakses Laravel
•	Set NLP_SERVICE_KEY di .env (sama di Laravel dan Flask)

// catatan : 
•   Update Index CBR php artisan nlp:retrain
•	Seed feedback historis ke CBR: php artisan cbr:sync-feedback --days=90


Post-Deploy
•	Cek health endpoint: GET /api/chatbot/health
•	Test pesan "Halo" → pastikan chit-chat response benar
•	Test pesan "Mau masak ayam" → pastikan intent cari_resep terdeteksi
•	Test topic switch: ayam → kambing → pastikan context di-replace


