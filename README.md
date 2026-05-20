#  Morfolojik Çözümleme (Morphological Disambiguation)

Bursa Teknik Üniversitesi — Bilgisayar Mühendisliği Bölümü  
**Doğal Dil İşleme (NLP) Dersi Projesi

---

##  Proje Hakkında

Türkçe gibi sondan eklemeli (agglutinative) dillerde bir kelime birden fazla morfolojik çözümlemeye sahip olabilir. Örneğin **"yüz"** kelimesi isim (sayı), sıfat (yüz ifadesi) veya fiil (yüzmek) olarak çözümlenebilir.

Bu proje, verilen Türkçe cümlelerdeki her kelimenin bağlamına uygun **doğru morfolojik analizini** (kök, kelime türü, ek bilgisi) otomatik olarak belirleyen bir sistem geliştirmeyi amaçlamaktadır.

## Proje Yapısı

```
├── src/
│   ├── preprocessing.py       # Veri ön işleme ve CoNLL-U okuma
│   ├── feature_extraction.py  # Özellik çıkarımı (323 özellik)
│   ├── models.py              # Model tanımları (NB, MLP, LR, HMM)
│   ├── train.py               # Model eğitimi
│   ├── evaluate.py            # Değerlendirme ve metrikler
│   └── utils.py               # Yardımcı fonksiyonlar
├── data/
│   ├── train/                 # Eğitim verisi (CoNLL-U formatı)
│   └── test/                  # Test verisi (CoNLL-U formatı)
├── models/                    # Kaydedilmiş model dosyaları
├── results/                   # Sonuç grafikleri ve confusion matrix'ler
├── docs/                      # Proje raporu
├── requirements.txt
└── README.md
```

##  Veri Seti

Tüm işaretlemeler **Universal Dependencies (UD)** standartlarında CoNLL-U formatında yapılmıştır.

|  | Eğitim Seti | Test Seti |
|---|---|---|
| Cümle Sayısı | 500 | 100 |
| Token Sayısı | 2199 | 449 |
| Ort. Token/Cümle | 4.4 | 4.5 |
| UPOS Sınıf Sayısı | 9 | 9 |

**UPOS Etiket Dağılımı (Eğitim Seti):**

| UPOS | NOUN | PUNCT | VERB | ADJ | ADV | PRON | DET | ADP | CCONJ |
|---|---|---|---|---|---|---|---|---|---|
| Sayı | 792 | 500 | 480 | 145 | 135 | 65 | 38 | 25 | 19 |
| Oran | %36.0 | %22.7 | %21.8 | %6.6 | %6.1 | %3.0 | %1.7 | %1.1 | %0.9 |

##  Yöntem

### Özellik Çıkarımı
Her token için toplam **323 özellik** üretilmiştir. Bağlamsal pencere yöntemi (window size = 2) kullanılarak hedef kelimenin kendisi ve ±2 komşu kelimesinin bilgileri birleştirilmiştir.

| Özellik Grubu | Açıklama |
|---|---|
| **Kelime Düzeyi** | Uzunluk, büyük harf, rakam kontrolü, ünlü-ünsüz oranı |
| **Morfolojik** | 36 yaygın Türkçe ek için binary özellikler (-lar, -ler, -dan, -yor, -mış vb.) |
| **N-gram** | İlk/son 1-2 karakter kodları |
| **Bağlamsal** | ±2 pencere içindeki kelimelerin özellikleri |
| **Konumsal** | Cümle içi pozisyon, baş/son kontrolü |

### Zemberek Entegrasyonu
[Zemberek](https://github.com/ahmetaa/zemberek-nlp), Türkçe için geliştirilmiş açık kaynaklı bir NLP kütüphanesidir. Her kelime için olası tüm morfolojik çözümlemeleri üretir; sistem bu adaylar arasında bağlamsal bilgiyi kullanarak doğru olanı seçer.

## Modeller

| Model | Tür | Parametreler | Avantaj | Dezavantaj |
|---|---|---|---|---|
| **Naive Bayes** (Gaussian) | Olasılıksal | Otomatik | Hızlı eğitim | Bağımsızlık varsayımı |
| **MLP** (128-64-32) | Sinir Ağı | 3 gizli katman | Doğrusal olmayan ilişkiler | Çok veri gerektirir |
| **Logistic Regression** | Doğrusal | C=1.0, L-BFGS | Yorumlanabilir | Doğrusal sınır |
| **HMM** | Dizilim Modeli | Geçiş + Emisyon | Dizilim bilgisi | Kelime bazlı |

## Sonuçlar

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
|---|---|---|---|---|
| Naive Bayes | %64.6 | 0.459 | 0.680 | 0.478 |
| MLP (128-64-32) | %87.8 | 0.590 | 0.531 | 0.553 |
| Logistic Regression | %89.5 | 0.770 | 0.666 | 0.704 |
| **HMM** | **%99.6** | **0.996** | **0.943** | **0.960** |

>  **En iyi sonuç: HMM modeli** — %99.6 accuracy, 0.960 F1-Score

### HMM Sınıf Bazında Sonuçlar

| Sınıf | Precision | Recall | F1-Score | Destek |
|---|---|---|---|---|
| ADJ | 0.971 | 1.000 | 0.986 | 34 |
| ADP | 1.000 | 0.500 | 0.667 | 2 |
| ADV | 1.000 | 1.000 | 1.000 | 23 |
| CCONJ | 1.000 | 1.000 | 1.000 | 2 |
| DET | 1.000 | 1.000 | 1.000 | 7 |
| NOUN | 0.994 | 1.000 | 0.997 | 179 |
| PRON | 1.000 | 1.000 | 1.000 | 9 |
| PUNCT | 1.000 | 1.000 | 1.000 | 100 |
| VERB | 1.000 | 0.989 | 0.995 | 93 |

## Kurulum ve Çalıştırma

```bash
# Repoyu klonla
git clone https://github.com/KULLANICI_ADI/nlp-morphological-disambiguation.git
cd nlp-morphological-disambiguation

# Bağımlılıkları yükle
pip install -r requirements.txt

# Modeli eğit
python src/train.py

# Değerlendirme
python src/evaluate.py
```

## Gelecek Çalışmalar

- UD Turkish-IMST gibi büyük ölçekli gerçek corpus'lar ile eğitim
- CRF (Conditional Random Fields) modeli eklenmesi
- Zemberek ile gerçek aday çözümleme üretimi
- Derin öğrenme modelleri (BiLSTM-CRF, BERT) denemesi

##  Kaynakça

1. Jurafsky, D., & Martin, J.H. (2024). *Speech and Language Processing* (3rd ed.). Stanford University.
2. Oflazer, K. (1994). *Two-level Description of Turkish Morphology*. Literary and Linguistic Computing, 9(2).
3. Sak, H., Güngör, T., & Saraçlar, M. (2007). *Morphological Disambiguation of Turkish Text with Perceptron Algorithm*. CICLing 2007.
4. Yüret, D., & Türe, F. (2006). *Learning Morphological Disambiguation Rules for Turkish*. HLT-NAACL 2006.
5. Akbik, A., et al. (2018). *Contextual String Embeddings for Sequence Labeling*. COLING 2018.
6. [Zemberek NLP](https://github.com/ahmetaa/zemberep-nlp)
7. [Universal Dependencies Turkish Treebank](https://universaldependencies.org/treebanks/tr_imst/)
8. Pedregosa et al. (2011). *scikit-learn: Machine Learning in Python*. JMLR 12.

##  Lisans

Bu proje akademik amaçlı geliştirilmiştir.

---

**Öğrenci:** Büşra Ünal — 22360859084  
**Ders:** Doğal Dil İşleme (NLP)  
**Kurum:** Bursa Teknik Üniversitesi — Bilgisayar Mühendisliği Bölümü
