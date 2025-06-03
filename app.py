"""
Yardım Çağrısı Sınıflandırma Web Uygulaması

Bu script, eğitilmiş model üzerinden web arayüzü ile tahmin yapılmasını sağlar.
"""

import os
import argparse
from flask import Flask, render_template, request, jsonify
from transformers import BertTokenizer

from src.model import TweetClassifier
from src.trainer import predict
from src.utils import get_device

app = Flask(__name__)

# Global değişkenler
MODEL = None
TOKENIZER = None
DEVICE = None

def parse_args():
    """Komut satırı argümanlarını işler."""
    parser = argparse.ArgumentParser(description='Yardım Çağrısı Sınıflandırıcı Web Uygulaması')
    
    parser.add_argument('--model_path', type=str, required=True,
                      help='Eğitilmiş model dosyası yolu')
    parser.add_argument('--pretrained_model', type=str, default='dbmdz/bert-base-turkish-cased',
                      help='Kullanılacak önceden eğitilmiş model')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                      help='Host adresi')
    parser.add_argument('--port', type=int, default=5000,
                      help='Port numarası')
    
    return parser.parse_args()

def load_model(model_path, pretrained_model, device):
    """Modeli yükler."""
    print(f"Model yükleniyor: {model_path}")
    model = TweetClassifier.load(model_path, device, pretrained_model)
    return model

@app.route('/')
def index():
    """Ana sayfa."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_api():
    """API tahmini."""
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({
            'error': 'Metin girişi boş olamaz'
        }), 400
    
    label, score = predict(MODEL, TOKENIZER, text, DEVICE)
    
    return jsonify({
        'text': text,
        'prediction': label,
        'score': score,
        'is_help_call': label == "Yardım çağrısı"
    })

def create_template_dir():
    """Template dizinini ve dosyalarını oluşturur."""
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # HTML şablonu oluştur
    html_content = """
    <!DOCTYPE html>
    <html lang="tr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Yardım Çağrısı Sınıflandırıcı</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                line-height: 1.6;
            }
            h1 {
                color: #2c3e50;
                text-align: center;
            }
            .container {
                background-color: #f9f9f9;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            textarea {
                width: 100%;
                height: 120px;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 4px;
                resize: vertical;
                margin-bottom: 10px;
            }
            button {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px 15px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 14px;
            }
            button:hover {
                background-color: #2980b9;
            }
            #result {
                margin-top: 20px;
                padding: 15px;
                border-radius: 4px;
                display: none;
            }
            .help-call {
                background-color: #f8d7da;
                color: #721c24;
            }
            .normal-tweet {
                background-color: #d4edda;
                color: #155724;
            }
            .confidence {
                font-size: 14px;
                margin-top: 5px;
            }
        </style>
    </head>
    <body>
        <h1>Yardım Çağrısı Sınıflandırıcı</h1>
        <div class="container">
            <p>Lütfen aşağıya analiz edilecek metni girin:</p>
            <textarea id="text-input" placeholder="Twitter mesajı, sosyal medya gönderisi veya metin girin..."></textarea>
            <button id="predict-btn">Analiz Et</button>
            
            <div id="result">
                <h3>Sonuç: <span id="prediction-label"></span></h3>
                <p id="prediction-text"></p>
                <p class="confidence">Güven skoru: <span id="confidence-score"></span></p>
            </div>
        </div>

        <script>
            document.getElementById('predict-btn').addEventListener('click', async () => {
                const text = document.getElementById('text-input').value.trim();
                
                if (!text) {
                    alert('Lütfen bir metin girin!');
                    return;
                }
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ text }),
                    });
                    
                    const result = await response.json();
                    
                    const resultElement = document.getElementById('result');
                    const predictionLabel = document.getElementById('prediction-label');
                    const predictionText = document.getElementById('prediction-text');
                    const confidenceScore = document.getElementById('confidence-score');
                    
                    if (result.is_help_call) {
                        resultElement.className = 'help-call';
                        predictionLabel.textContent = 'Yardım Çağrısı';
                        predictionText.textContent = 'Bu metin, bir yardım çağrısı içeriyor olabilir. Acil durum veya yardım talebi gösteriyor.';
                    } else {
                        resultElement.className = 'normal-tweet';
                        predictionLabel.textContent = 'Normal Tweet';
                        predictionText.textContent = 'Bu metin, normal bir sosyal medya paylaşımıdır. Yardım çağrısı içermiyor.';
                    }
                    
                    confidenceScore.textContent = `${(result.score * 100).toFixed(2)}%`;
                    resultElement.style.display = 'block';
                    
                } catch (error) {
                    console.error('Hata:', error);
                    alert('Bir hata oluştu. Lütfen tekrar deneyin.');
                }
            });
        </script>
    </body>
    </html>
    """
    
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

def main():
    """Ana uygulama işlevi."""
    global MODEL, TOKENIZER, DEVICE
    
    # Argümanları işle
    args = parse_args()
    
    # Template dizinini ve dosyalarını oluştur
    create_template_dir()
    
    # Cihazı belirle
    DEVICE = get_device()
    print(f"Cihaz: {DEVICE}")
    
    # Tokenizer oluştur
    TOKENIZER = BertTokenizer.from_pretrained(args.pretrained_model)
    
    # Modeli yükle
    MODEL = load_model(args.model_path, args.pretrained_model, DEVICE)
    
    # Uygulamayı başlat
    print(f"Web uygulaması başlatılıyor: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)

if __name__ == "__main__":
    main() 