import torch
from torch import nn
from transformers import BertModel

class TweetClassifier(nn.Module):
    def __init__(self, pretrained_model='dbmdz/bert-base-turkish-cased', n_classes=2, dropout=0.3):
        """
        Türkçe dili için özel BERT tabanlı sınıflandırıcı model.
        
        Args:
            pretrained_model (str): Kullanılacak önceden eğitilmiş BERT modeli.
            n_classes (int): Sınıf sayısı, varsayılan olarak 2 (yardım çağrısı veya normal tweet).
            dropout (float): Dropout oranı.
        """
        super(TweetClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.drop = nn.Dropout(p=dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.pretrained_model = pretrained_model  # Modelin orijinal adını kaydet
    
    def forward(self, input_ids, attention_mask):
        """
        Model için ileri geçiş fonksiyonu.
        
        Args:
            input_ids (torch.Tensor): BERT token ID'leri.
            attention_mask (torch.Tensor): Dikkat maskeleri.
            
        Returns:
            torch.Tensor: Sınıf logit'leri.
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs[1]  # CLS token çıktısı
        x = self.drop(pooled_output)
        return self.fc(x)
    
    def save(self, path):
        """
        Modeli belirtilen yola kaydeder.
        
        Args:
            path (str): Modelin kaydedileceği dosya yolu.
        """
        torch.save({
            'state_dict': self.state_dict(),
            'pretrained_model': self.pretrained_model,
            'n_classes': self.fc.out_features,
        }, path)
        print(f"Model kaydedildi: {path}")
    
    @classmethod
    def load(cls, path, device='cpu', pretrained_model=None, n_classes=None):
        """
        Kaydedilmiş modeli yükler.
        
        Args:
            path (str): Modelin yükleneceği dosya yolu.
            device (str): Modelin çalışacağı cihaz ('cpu' veya 'cuda').
            pretrained_model (str, optional): BERT modeli adı (kaydedilmişse kullanılmaz).
            n_classes (int, optional): Sınıf sayısı (kaydedilmişse kullanılmaz).
            
        Returns:
            TweetClassifier: Yüklenmiş model.
        """
        checkpoint = torch.load(path, map_location=device)
        
        # Eski model formatı kontrolü
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            # Yeni format
            state_dict = checkpoint['state_dict']
            pretrained_model = checkpoint.get('pretrained_model', pretrained_model)
            n_classes = checkpoint.get('n_classes', n_classes or 2)
        else:
            # Eski format (sadece state_dict)
            state_dict = checkpoint
            if pretrained_model is None:
                pretrained_model = 'dbmdz/bert-base-turkish-cased'
            if n_classes is None:
                n_classes = 2
        
        model = cls(pretrained_model=pretrained_model, n_classes=n_classes)
        model.load_state_dict(state_dict)
        model.to(device)
        return model 