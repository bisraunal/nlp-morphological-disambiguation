"""
Model Eğitim Betiği
-------------------
CoNLL-U formatındaki eğitim verisini okur, özellik çıkarımı yapar
ve dört modeli (NB, MLP, LR, HMM) eğitir.
Eğitilmiş modeller models/ dizinine kaydedilir.
"""

import os
import sys
import pickle
import warnings
import numpy as np
from datetime import datetime

from preprocessing import conllu_oku, tokenleri_ve_etiketleri_ayir, veri_seti_istatistikleri
from feature_extraction import veri_setini_ozelliklendir
from models import naive_bayes_olustur, mlp_olustur, logistic_regression_olustur, GizliMarkovModeli

warnings.filterwarnings('ignore')

# ── Varsayılan Yollar ──
EGITIM_DOSYASI = os.path.join('data', 'train', 'train.conllu')
MODEL_DIZINI = 'models'


def modelleri_egit(egitim_dosyasi: str = EGITIM_DOSYASI):
    """
    Tüm modelleri eğitir ve kaydeder.
    
    Adımlar:
        1. CoNLL-U verisini oku
        2. Özellik çıkarımı yap
        3. Her modeli eğit
        4. Eğitilmiş modelleri kaydet
    """
    print("=" * 60)
    print("  MORFOLOJIK ÇÖZÜMLEME — MODEL EĞİTİMİ")
    print("=" * 60)
    print(f"  Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()

    # ── 1. Veriyi Oku ──
    print("[1/4] Eğitim verisi okunuyor...")
    if not os.path.exists(egitim_dosyasi):
        print(f"  HATA: '{egitim_dosyasi}' dosyası bulunamadı!")
        print("  Lütfen eğitim verinizi data/train/ dizinine yerleştirin.")
        sys.exit(1)

    cumleler = conllu_oku(egitim_dosyasi)
    istatistik = veri_seti_istatistikleri(cumleler)
    print(f"  → {istatistik['cumle_sayisi']} cümle, {istatistik['token_sayisi']} token yüklendi")
    print(f"  → UPOS sınıf sayısı: {istatistik['sinif_sayisi']}")
    print()

    # ── 2. Özellik Çıkarımı ──
    print("[2/4] Özellik çıkarımı yapılıyor...")
    tokenler, etiketler = tokenleri_ve_etiketleri_ayir(cumleler)
    X_egitim, y_egitim = veri_setini_ozelliklendir(tokenler, etiketler)
    print(f"  → Özellik matrisi boyutu: {X_egitim.shape}")
    print(f"  → Token başına özellik  : {X_egitim.shape[1]}")
    print()

    # ── 3. Modelleri Eğit ──
    print("[3/4] Modeller eğitiliyor...")
    os.makedirs(MODEL_DIZINI, exist_ok=True)

    # 3a. Naive Bayes
    print("  ▸ Naive Bayes (Gaussian)...")
    nb_modeli = naive_bayes_olustur()
    nb_modeli.fit(X_egitim, y_egitim)
    print("    ✓ Eğitim tamamlandı")

    # 3b. MLP
    print("  ▸ MLP (128-64-32)...")
    mlp_modeli = mlp_olustur()
    mlp_modeli.fit(X_egitim, y_egitim)
    print(f"    ✓ Eğitim tamamlandı ({mlp_modeli.n_iter_} epoch)")

    # 3c. Logistic Regression
    print("  ▸ Logistic Regression (Multinomial)...")
    lr_modeli = logistic_regression_olustur()
    lr_modeli.fit(X_egitim, y_egitim)
    print("    ✓ Eğitim tamamlandı")

    # 3d. HMM (dizilim tabanlı — doğrudan token/etiket ile eğitilir)
    print("  ▸ HMM (Gizli Markov Modeli)...")
    hmm_modeli = GizliMarkovModeli()
    hmm_modeli.egit(tokenler, etiketler)
    print("    ✓ Eğitim tamamlandı")
    print()

    # ── 4. Modelleri Kaydet ──
    print("[4/4] Modeller kaydediliyor...")
    modeller = {
        'naive_bayes': nb_modeli,
        'mlp': mlp_modeli,
        'logistic_regression': lr_modeli,
        'hmm': hmm_modeli,
    }

    for isim, model in modeller.items():
        dosya_yolu = os.path.join(MODEL_DIZINI, f'{isim}.pkl')
        with open(dosya_yolu, 'wb') as f:
            pickle.dump(model, f)
        print(f"  ✓ {dosya_yolu}")

    print()
    print("=" * 60)
    print("  Tüm modeller başarıyla eğitildi ve kaydedildi!")
    print("  Değerlendirme için: python evaluate.py")
    print("=" * 60)


if __name__ == '__main__':
    dosya = sys.argv[1] if len(sys.argv) > 1 else EGITIM_DOSYASI
    modelleri_egit(dosya)
