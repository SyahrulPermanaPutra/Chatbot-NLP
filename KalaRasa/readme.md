Pre-Deploy
•	Pastikan Flask service berjalan: curl http://localhost:5000/health
•	Jalankan CBR index build: php artisan cbr:rebuild-index
•	Seed feedback historis ke CBR: php artisan cbr:sync-feedback --days=90
•	Validasi kamus NLP: python src/validate_dict.py
•	Pastikan Redis berjalan dan dapat diakses Laravel
•	Set NLP_SERVICE_KEY di .env (sama di Laravel dan Flask)

Post-Deploy
•	Cek health endpoint: GET /api/chatbot/health
•	Test pesan "Halo" → pastikan chit-chat response benar
•	Test pesan "Mau masak ayam" → pastikan intent cari_resep terdeteksi
•	Test topic switch: ayam → kambing → pastikan context di-replace


