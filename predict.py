"""
Yardım Çağrısı Sınıflandırma Modeli Tahmin Kodu

Bu script, eğitilmiş modeli kullanarak metinlerin yardım çağrısı olup olmadığını tahmin eder.
"""

import os
import argparse
import json
from transformers import BertTokenizer

from src.model import TweetClassifier
from src.trainer import predict
from src.utils import get_device

def parse_args():
    """Komut satırı argümanlarını işler."""
    parser = argparse.ArgumentParser(description='Yardım Çağrısı Sınıflandırıcı Tahmini')
    
    # Model parametreleri
    parser.add_argument('--model_path', type=str, required=True,
                      help='Eğitilmiş model dosyası yolu')
    parser.add_argument('--pretrained_model', type=str, default='dbmdz/bert-base-turkish-cased',
                      help='Kullanılacak önceden eğitilmiş model')
    
    # Tahmin parametreleri
    parser.add_argument('--text', type=str,
                      help='Tahmin edilecek metin')
    parser.add_argument('--input_file', type=str,
                      help='Metin içeren satırlar halindeki dosya')
    parser.add_argument('--output_file', type=str,
                      help='Tahmin sonuçlarının kaydedileceği dosya')
    
    # GPU kullanımı
    parser.add_argument('--use_gpu', action='store_true',
                      help='GPU kullanımını zorunlu hale getirir')
    
    return parser.parse_args()

def load_model(model_path, pretrained_model, device):
    """Modeli yükler."""
    print(f"Model yükleniyor: {model_path}")
    model = TweetClassifier.load(model_path, device, pretrained_model)
    return model

def main():
    """Ana tahmin işlevi."""
    # Argümanları işle
    args = parse_args()
    
    # Cihazı belirle
    device = get_device(force_gpu=args.use_gpu)
    print(f"Cihaz: {device}")
    
    # Tokenizer oluştur
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
    
    # Modeli yükle
    model = load_model(args.model_path, args.pretrained_model, device)
    
    # Tek metin tahmini
    if args.text:
        label, score = predict(model, tokenizer, args.text, device)
        print(f"Metin: {args.text}")
        print(f"Tahmin: {label}")
        print(f"Skor: {score:.4f}")
    
    # Dosyadan metin tahmini
    elif args.input_file:
        results = []
        
        print(f"Dosyadan metinler okunuyor: {args.input_file}")
        with open(args.input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines):
            text = line.strip()
            if not text:
                continue
                
            label, score = predict(model, tokenizer, text, device)
            
            result = {
                'id': i,
                'text': text,
                'prediction': label,
                'score': score
            }
            
            results.append(result)
            
            # İlerleme gösterimi
            if (i + 1) % 10 == 0:
                print(f"{i + 1}/{len(lines)} metin işlendi...")
        
        # Sonuçları kaydet
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            print(f"Sonuçlar {args.output_file} dosyasına kaydedildi.")
        else:
            for result in results:
                print(f"Metin: {result['text']}")
                print(f"Tahmin: {result['prediction']}")
                print(f"Skor: {result['score']:.4f}")
                print("-" * 50)
    
    else:
        print("Lütfen tahmin için bir metin veya giriş dosyası belirtin.")

if __name__ == "__main__":
    main() 