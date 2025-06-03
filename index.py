import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Veri seti sınıfı
class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
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

# BERT model sınıfı
class TweetClassifier(nn.Module):
    def __init__(self, n_classes=2):
        super(TweetClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('dbmdz/bert-base-turkish-cased')
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(outputs[1])
        return self.fc(output)

def train_model(model, train_loader, val_loader, device, epochs=3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}")
        
        # Validasyon
        val_accuracy, val_report = evaluate_model(model, val_loader, device)
        print(f"Validation Accuracy: {val_accuracy}")
        print(f"Classification Report:\n{val_report}")
    
    return model

def evaluate_model(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    
    accuracy = accuracy_score(actual_labels, predictions)
    report = classification_report(actual_labels, predictions)
    
    return accuracy, report

def predict_tweet(model, tokenizer, text, device):
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
        _, prediction = torch.max(outputs, dim=1)
        
    return "Yardım çağrısı" if prediction.item() == 1 else "Normal tweet"

def main():
    # Veri yükleme
    data = pd.read_csv('data/database.csv')
    
    # Veriyi inceleme
    print(f"Veri şekli: {data.shape}")
    print(f"Sınıf dağılımı:\n{data['label'].value_counts()}")
    
    # Cihaz belirleme
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan cihaz: {device}")
    
    # Verileri ayırma
    train_data, val_data = train_test_split(
        data, test_size=0.2, random_state=42, stratify=data['label']
    )
    
    # Tokenizer ve model oluşturma
    tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased')
    model = TweetClassifier().to(device)
    
    # Veri setlerini oluşturma
    train_dataset = TweetDataset(
        texts=train_data['text'].values,
        labels=train_data['label'].values,
        tokenizer=tokenizer
    )
    
    val_dataset = TweetDataset(
        texts=val_data['text'].values,
        labels=val_data['label'].values,
        tokenizer=tokenizer
    )
    
    # DataLoader oluşturma
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False
    )
    
    # Modeli eğitme
    trained_model = train_model(model, train_loader, val_loader, device, epochs=3)
    
    # Modeli kaydetme
    torch.save(trained_model.state_dict(), 'yardim_cagrisi_model.pt')
    
    # Örnek tahmin
    ornek_tweet = "Adres: Malatya Battalgazi İnönü Caddesi No:35, yardım bekliyoruz, 3 gündür kimse gelmedi."
    tahmin = predict_tweet(trained_model, tokenizer, ornek_tweet, device)
    print(f"\nÖrnek tweet: {ornek_tweet}")
    print(f"Tahmin: {tahmin}")
    
    ornek_tweet2 = "Galatasaray, Trabzon'dan galibiyetle dönüyor"
    tahmin2 = predict_tweet(trained_model, tokenizer, ornek_tweet2, device)
    print(f"\nÖrnek tweet: {ornek_tweet2}")
    print(f"Tahmin: {tahmin2}")

if __name__ == "__main__":
    main()
