# Türkçe Yardım Çağrısı Analiz Modeli

Bu proje, Twitter'dan alınan Türkçe metinleri analiz ederek, bir metnin yardım çağrısı olup olmadığını sınıflandıran bir BERT modeli içermektedir.

## Proje Hakkında

Bu model, deprem gibi afet durumlarında sosyal medyadan gelen yardım çağrılarını tespit etmek üzere tasarlanmıştır. Twitter veya benzeri platformlardan alınan metinleri analiz ederek, acil durumları ve yardım çağrılarını normal sosyal medya paylaşımlarından ayırt edebilir.

## Veri Seti

Projede kullanılan veri seti şu bilgileri içerir:
- `id`: Metin için benzersiz tanımlayıcı
- `text`: Analiz edilecek Türkçe metin
- `label`: Yardım çağrısı (1) veya normal tweet (0)

## Proje Yapısı

```
├── data/
│   └── database.csv                # Eğitim verileri
├── src/
│   ├── __init__.py                 # Modül tanımı
│   ├── model.py                    # Model tanımları
│   ├── data.py                     # Veri yükleme ve işleme
│   ├── trainer.py                  # Model eğitimi ve değerlendirme
│   └── utils.py                    # Yardımcı fonksiyonlar
├── experiments/                    # Eğitim deneyleri (otomatik oluşturulur)
│   └── experiment_YYYYMMDD_HHMMSS/ # Zaman damgalı deney dizini
│       ├── models/                 # Eğitilmiş modeller
│       ├── plots/                  # Eğitim grafikleri
│       └── config.json             # Deney konfigürasyonu
├── templates/                      # Web arayüzü şablonları
│   └── index.html                  # Ana sayfa
├── train.py                        # Eğitim kodu
├── predict.py                      # Tahmin kodu
├── app.py                          # Web uygulaması
├── requirements.txt                # Proje bağımlılıkları
└── README.md                       # Proje açıklaması
```

## Model

Model, Türkçe metinleri analiz etmek için özel olarak eğitilmiş `dbmdz/bert-base-turkish-cased` BERT modeli temel alınarak oluşturulmuştur.

## Kurulum

Gerekli kütüphaneleri yüklemek için:

```bash
pip install -r requirements.txt
```

## Kullanım

### Model Eğitimi

Modeli eğitmek için:

```bash
python train.py --epochs 3 --batch_size 16 --save_best
```

Diğer parametreler:

```
--data_path        : Veri seti dosya yolu (varsayılan: data/database.csv)
--test_size        : Test seti oranı (varsayılan: 0.2)
--pretrained_model : Kullanılacak önceden eğitilmiş model (varsayılan: dbmdz/bert-base-turkish-cased)
--max_len          : Maksimum metin uzunluğu (varsayılan: 128)
--learning_rate    : Öğrenme hızı (varsayılan: 2e-5)
--seed             : Rastgele sayı üreteci için tohum değeri (varsayılan: 42)
--save_best        : Sadece en iyi modeli kaydet (flag)
```

### Tahmin Yapma

Komut satırından tahmin yapmak için:

```bash
python predict.py --model_path models/model.pt --text "Adres: Malatya Battalgazi İnönü Caddesi No:35, yardım bekliyoruz, 3 gündür kimse gelmedi."
```

Metin dosyasından tahminler:

```bash
python predict.py --model_path models/model.pt --input_file metinler.txt --output_file sonuclar.json
```

### Web Arayüzü

Web tabanlı arayüz ile kullanmak için:

```bash
python app.py --model_path expr --port 5000
```

Tarayıcınızda `http://localhost:5000` adresine giderek web arayüzünü kullanabilirsiniz.

![Web Arayüzü](docs/web_interface.png)

## API Kullanımı

Web uygulaması üzerinden API çağrıları da yapabilirsiniz:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"text":"Adres: Malatya Battalgazi İnönü Caddesi No:35, yardım bekliyoruz, 3 gündür kimse gelmedi."}' http://localhost:5000/predict
```

## Örnek Python Kullanımı

```python
from transformers import BertTokenizer
from src.model import TweetClassifier
from src.trainer import predict
from src.utils import get_device

# Cihazı belirle
device = get_device()

# Model ve tokenizer yükleme
model = TweetClassifier.load('models/model.pt', device)
tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased')

# Tahmin yapma
text = "Adres: Malatya Battalgazi İnönü Caddesi No:35, yardım bekliyoruz, 3 gündür kimse gelmedi."
label, score = predict(model, tokenizer, text, device)
print(f"Tahmin: {label}, Skor: {score:.4f}")
``` 