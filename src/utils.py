import os
import json
import torch
import matplotlib.pyplot as plt
from datetime import datetime

def set_seed(seed=42):
    """
    Çoğaltılabilirlik için rastgele sayı üretecini sabitler.
    
    Args:
        seed (int): Rastgele sayı üreteci için tohum değeri.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_device(force_gpu=False):
    """
    Kullanılabilir en iyi cihazı döndürür.
    
    Args:
        force_gpu (bool): GPU kullanımını zorunlu kılmak için True değerini verin.
        
    Returns:
        torch.device: 'cuda' veya 'cpu'
    """
    if force_gpu:
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            raise RuntimeError("GPU (CUDA) kullanım için seçildi fakat bu sistemde CUDA etkin bir GPU bulunamadı.")
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_config(config, path='config.json'):
    """
    Konfigürasyon parametrelerini bir JSON dosyasına kaydeder.
    
    Args:
        config (dict): Konfigürasyon sözlüğü.
        path (str): Kaydedilecek dosya yolu.
    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

def load_config(path='config.json'):
    """
    Konfigürasyon parametrelerini bir JSON dosyasından yükler.
    
    Args:
        path (str): Yüklenecek dosya yolu.
        
    Returns:
        dict: Konfigürasyon sözlüğü.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_training_history(history, save_path=None):
    """
    Eğitim ve validasyon kayıplarını ve doğruluk değerlerini çizer.
    
    Args:
        history (dict): Eğitim geçmişi.
        save_path (str, optional): Grafiğin kaydedileceği yol.
    """
    plt.figure(figsize=(12, 5))
    
    # Kayıp grafiği
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Eğitim')
    plt.plot(history['val_loss'], label='Validasyon')
    plt.title('Eğitim ve Validasyon Kaybı')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp')
    plt.legend()
    
    # Doğruluk grafiği
    plt.subplot(1, 2, 2)
    plt.plot(history['val_accuracy'], label='Validasyon')
    plt.title('Validasyon Doğruluğu')
    plt.xlabel('Epoch')
    plt.ylabel('Doğruluk')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def create_experiment_dir(base_dir='experiments'):
    """
    Zaman damgalı bir deney dizini oluşturur.
    
    Args:
        base_dir (str): Temel dizin.
        
    Returns:
        str: Oluşturulan dizinin yolu.
    """
    # Zaman damgası oluştur
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Dizin yolu oluştur
    experiment_dir = os.path.join(base_dir, f"experiment_{timestamp}")
    
    # Dizini oluştur
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Alt dizinleri oluştur
    models_dir = os.path.join(experiment_dir, "models")
    plots_dir = os.path.join(experiment_dir, "plots")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    return experiment_dir 