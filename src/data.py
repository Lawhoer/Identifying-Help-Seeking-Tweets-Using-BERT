import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        """
        Tweet veri seti sınıfı.
        
        Args:
            texts (list): Tweet metinleri listesi.
            labels (list): Etiketler listesi (0: normal tweet, 1: yardım çağrısı).
            tokenizer (BertTokenizer): Metinleri tokenize etmek için kullanılacak tokenizer.
            max_len (int): Maksimum metin uzunluğu.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        """Veri seti boyutunu döndürür."""
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        Belirtilen indeksteki örneği döndürür.
        
        Args:
            idx (int): Örnek indeksi.
            
        Returns:
            dict: Tokenize edilmiş metin ve etiketi içeren sözlük.
        """
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def load_data(data_path, test_size=0.2, random_state=42):
    """
    Veri setini yükler ve eğitim/test setlerine ayırır.
    
    Args:
        data_path (str): Veri seti dosya yolu.
        test_size (float): Test seti oranı.
        random_state (int): Rastgele durum sayısı.
        
    Returns:
        tuple: (train_data, val_data) pandas DataFrame'leri.
    """
    data = pd.read_csv(data_path)
    
    print(f"Veri şekli: {data.shape}")
    print(f"Sınıf dağılımı:\n{data['label'].value_counts()}")
    
    # Veriyi eğitim ve test setlerine ayırma
    train_data, val_data = train_test_split(
        data, test_size=test_size, random_state=random_state, stratify=data['label']
    )
    
    return train_data, val_data

def create_data_loaders(train_data, val_data, tokenizer, batch_size=16, max_len=128, num_workers=0):
    """
    Eğitim ve validasyon veri yükleyicilerini oluşturur.
    
    Args:
        train_data (pd.DataFrame): Eğitim verileri.
        val_data (pd.DataFrame): Validasyon verileri.
        tokenizer (BertTokenizer): BERT tokenizer.
        batch_size (int): Batch boyutu.
        max_len (int): Maksimum metin uzunluğu.
        num_workers (int): Veri yükleme için çalışan sayısı.
        
    Returns:
        tuple: (train_loader, val_loader) DataLoader'ları.
    """
    # Veri setlerini oluşturma
    train_dataset = TweetDataset(
        texts=train_data['text'].values,
        labels=train_data['label'].values,
        tokenizer=tokenizer,
        max_len=max_len
    )
    
    val_dataset = TweetDataset(
        texts=val_data['text'].values,
        labels=val_data['label'].values,
        tokenizer=tokenizer,
        max_len=max_len
    )
    
    # DataLoader'ları oluşturma
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader 