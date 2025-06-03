import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import os
import json

class Trainer:
    def __init__(
        self, 
        model, 
        train_loader, 
        val_loader, 
        device, 
        learning_rate=2e-5, 
        model_save_path='models',
        optimizer=None,
        scheduler=None,
        fp16=False
    ):
        """
        Model eğitici sınıfı.
        
        Args:
            model: Eğitilecek model.
            train_loader: Eğitim veri yükleyicisi.
            val_loader: Validasyon veri yükleyicisi.
            device: Eğitimin yapılacağı cihaz ('cpu' veya 'cuda').
            learning_rate (float): Öğrenme hızı.
            model_save_path (str): Model ve sonuçların kaydedileceği dizin.
            optimizer: Özel bir optimizer (belirtilmezse AdamW kullanılır).
            scheduler: Öğrenme hızı planlayıcısı (belirtilmezse kullanılmaz).
            fp16 (bool): FP16 hassasiyeti kullanılsın mı (GPU gerektirir).
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer = optimizer if optimizer is not None else torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.scheduler = scheduler
        self.loss_fn = nn.CrossEntropyLoss()
        self.model_save_path = model_save_path
        self.fp16 = fp16
        
        # FP16 için scaler oluştur
        self.scaler = torch.cuda.amp.GradScaler() if fp16 else None
        
        # Model kaydetme dizinini oluştur
        os.makedirs(model_save_path, exist_ok=True)
        
        # Eğitim geçmişi
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
    
    def train_epoch(self):
        """
        Bir epoch için eğitim yapar.
        
        Returns:
            float: Ortalama eğitim kaybı.
        """
        self.model.train()
        total_loss = 0
        
        # Progress bar oluştur
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Eğitim")
        
        for step, batch in progress_bar:
            # Veriyi cihaza taşı
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Gradyanları sıfırla
            self.optimizer.zero_grad()
            
            # Forward pass (FP16 için)
            if self.fp16:
                with torch.cuda.amp.autocast():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = self.loss_fn(outputs, labels)
                
                # Geriye yayılım (FP16)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Normal forward pass ve geriye yayılım
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            # Scheduler'ı güncelle
            if self.scheduler is not None:
                self.scheduler.step()
            
            total_loss += loss.item()
            
            # Progress bar güncelle
            progress_bar.set_postfix({"Kayıp": f"{loss.item():.4f}"})
            
        return total_loss / len(self.train_loader)
    
    def evaluate(self):
        """
        Modeli validasyon veri seti üzerinde değerlendirir.
        
        Returns:
            tuple: (val_loss, accuracy, classification_report_dict)
        """
        self.model.eval()
        total_loss = 0
        predictions = []
        actual_labels = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.loss_fn(outputs, labels)
                total_loss += loss.item()
                
                _, preds = torch.max(outputs, dim=1)
                
                predictions.extend(preds.cpu().tolist())
                actual_labels.extend(labels.cpu().tolist())
        
        # Metrikleri hesapla
        val_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(actual_labels, predictions)
        report_dict = classification_report(actual_labels, predictions, output_dict=True)
        
        return val_loss, accuracy, report_dict
    
    def train(self, epochs, save_best=True):
        """
        Modeli belirtilen sayıda epoch için eğitir.
        
        Args:
            epochs (int): Eğitim epoch sayısı.
            save_best (bool): Sadece en iyi modeli mi kaydet.
            
        Returns:
            Eğitilmiş model.
        """
        best_accuracy = 0
        best_epoch = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Eğitim
            train_loss = self.train_epoch()
            
            # Değerlendirme
            val_loss, accuracy, report = self.evaluate()
            
            # Sonuçları yazdır
            print(f"Eğitim Kaybı: {train_loss:.4f}")
            print(f"Validasyon Kaybı: {val_loss:.4f}")
            print(f"Doğruluk: {accuracy:.4f}")
            print(f"F1 (Yardım Çağrısı Sınıfı): {report['1']['f1-score']:.4f}")
            
            # Geçmişi güncelle
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(accuracy)
            
            # En iyi modeli kaydet
            if save_best and accuracy > best_accuracy:
                best_accuracy = accuracy
                best_epoch = epoch
                self._save_model(epoch, accuracy, report)
            elif not save_best:
                self._save_model(epoch, accuracy, report)
        
        # Eğitim sonuçlarını kaydet
        self._save_history()
        
        if save_best:
            print(f"\nEn iyi model (epoch {best_epoch+1}) kaydedildi. Doğruluk: {best_accuracy:.4f}")
        
        return self.model
    
    def _save_model(self, epoch, accuracy, report):
        """Model ve etiketlerini kaydeder."""
        # Modeli kaydet
        model_filename = os.path.join(self.model_save_path, f"model_epoch_{epoch+1}_acc_{accuracy:.4f}.pt")
        torch.save(self.model.state_dict(), model_filename)
        print(f"Model kaydedildi: {model_filename}")
        
        # Model bilgilerini kaydet
        model_info = {
            'epoch': epoch + 1,
            'accuracy': accuracy,
            'report': report,
        }
        
        model_info_filename = os.path.join(
            self.model_save_path, 
            f"model_info_epoch_{epoch+1}_acc_{accuracy:.4f}.json"
        )
        
        with open(model_info_filename, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=4)
        print(f"Model bilgisi kaydedildi: {model_info_filename}")
    
    def _save_history(self):
        """Eğitim geçmişini kaydeder."""
        history_filename = os.path.join(self.model_save_path, "training_history.json")
        with open(history_filename, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=4)


def predict(model, tokenizer, text, device):
    """
    Verilen metin için tahmin yapar.
    
    Args:
        model: Tahmin yapacak model.
        tokenizer: Tokenizer.
        text (str): Tahmin edilecek metin.
        device: Model cihazı ('cpu' veya 'cuda').
        
    Returns:
        tuple: (tahmin_etiketi, tahmin_skoru)
    """
    model.eval()
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        probs = torch.softmax(outputs, dim=1)
        _, prediction = torch.max(outputs, dim=1)
        
    label = "Yardım çağrısı" if prediction.item() == 1 else "Normal tweet"
    score = probs[0][prediction.item()].item()
    
    return label, score 