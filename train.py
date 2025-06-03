"""
Yardım Çağrısı Sınıflandırma Modeli Eğitim Kodu

Bu script, BERT tabanlı bir modeli kullanarak yardım çağrısı sınıflandırması için eğitir.
"""

import os
import argparse
import json
import torch
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
import matplotlib.pyplot as plt

from src.model import TweetClassifier
from src.trainer import Trainer
from src.data import load_data, create_data_loaders
from src.utils import get_device, set_seed, create_experiment_dir, save_config, plot_training_history

def parse_args():
    """Komut satırı argümanlarını işler."""
    parser = argparse.ArgumentParser(description='Yardım Çağrısı Sınıflandırıcı Eğitimi')
    
    # Veri parametreleri
    parser.add_argument('--data_path', type=str, required=True,
                      help='Eğitim veri seti dosyası yolu')
    parser.add_argument('--val_size', type=float, default=0.2,
                      help='Validasyon seti oranı')
    
    # Model parametreleri
    parser.add_argument('--pretrained_model', type=str, default='dbmdz/bert-base-turkish-cased',
                      help='Kullanılacak önceden eğitilmiş model')
    parser.add_argument('--max_len', type=int, default=128,
                      help='Maksimum metin uzunluğu')
    
    # Eğitim parametreleri
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Batch boyutu')
    parser.add_argument('--epochs', type=int, default=4,
                      help='Eğitim epoch sayısı')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                      help='Öğrenme hızı')
    parser.add_argument('--seed', type=int, default=42,
                      help='Rastgele tohum değeri')
    parser.add_argument('--num_workers', type=int, default=0,
                      help='Veri yükleme için işçi sayısı')
    
    # Çıktı parametreleri
    parser.add_argument('--output_dir', type=str, default='experiments',
                      help='Sonuçların kaydedileceği dizin')
    
    # GPU kullanımı
    parser.add_argument('--use_gpu', action='store_true',
                      help='GPU kullanımını zorunlu hale getirir (GPU varsa)')
    
    # Performans optimizasyonları
    parser.add_argument('--fp16', action='store_true',
                      help='16-bit hassasiyet kullanarak eğitimi hızlandırır (GPU gerektirir)')
    parser.add_argument('--warmup_steps', type=int, default=0,
                      help='Isınma adımları sayısı')
    
    return parser.parse_args()

def main():
    """Ana eğitim işlevi."""
    # Argümanları işle
    args = parse_args()
    
    # Rastgele tohum değerini ayarla
    set_seed(args.seed)
    
    # Çıktı dizinlerini oluştur
    experiment_dir = create_experiment_dir(args.output_dir)
    models_dir = os.path.join(experiment_dir, "models")
    plots_dir = os.path.join(experiment_dir, "plots")
    
    print(f"Deney dizini: {experiment_dir}")
    
    # Konfigürasyonu kaydet
    config_path = os.path.join(experiment_dir, "config.json")
    save_config(vars(args), config_path)
    print(f"Konfigürasyon kaydedildi: {config_path}")
    
    # Cihazı belirle
    try:
        device = get_device(force_gpu=args.use_gpu)
    except RuntimeError:
        print("UYARI: GPU isteği yapılmış ancak GPU bulunamadı. CPU kullanılacak.")
        device = torch.device('cpu')
    
    print(f"Kullanılan cihaz: {device}")
    
    # FP16 ayarla
    fp16 = False
    if args.fp16:
        if device.type == 'cuda' and torch.cuda.is_available():
            fp16 = True
            print("FP16 hassasiyeti etkinleştirildi")
        else:
            print("UYARI: FP16 hassasiyeti sadece GPU ile kullanılabilir. Devre dışı bırakıldı.")
    
    # Veriyi yükle
    print("Veri yükleniyor...")
    train_data, val_data = load_data(args.data_path, test_size=args.val_size, random_state=args.seed)
    
    # Tokenizer oluştur
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
    
    # Veri yükleyicilerini oluştur
    print("Veri yükleyicileri oluşturuluyor...")
    train_loader, val_loader = create_data_loaders(
        train_data, 
        val_data, 
        tokenizer, 
        batch_size=args.batch_size, 
        max_len=args.max_len,
        num_workers=args.num_workers
    )
    
    # Modeli oluştur
    print("Model oluşturuluyor...")
    model = TweetClassifier(
        pretrained_model=args.pretrained_model, 
        n_classes=2
    )
    model.to(device)
    
    # Eğitim için hazırlık
    total_steps = len(train_loader) * args.epochs
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, correct_bias=False)
    
    # Öğrenme hızı planlayıcısı
    scheduler = None
    if args.warmup_steps > 0:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_steps
        )
        print(f"Öğrenme hızı planlayıcısı oluşturuldu. Isınma adımları: {args.warmup_steps}")
    
    # Eğiticiyi oluştur
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        model_save_path=models_dir,
        optimizer=optimizer,
        scheduler=scheduler,
        fp16=fp16
    )
    
    # Modeli eğit
    print(f"Eğitim başlıyor... ({args.epochs} epoch)")
    trainer.train(epochs=args.epochs, save_best=True)
    
    # Eğitim geçmişini çiz
    history_path = os.path.join(models_dir, "training_history.json")
    if os.path.exists(history_path):
        with open(history_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        plot_path = os.path.join(plots_dir, "training_history.png")
        plot_training_history(history, save_path=plot_path)
        print(f"Eğitim geçmişi grafiği kaydedildi: {plot_path}")
    
    print(f"Eğitim tamamlandı! Sonuçlar {experiment_dir} dizinine kaydedildi.")

if __name__ == "__main__":
    main() 