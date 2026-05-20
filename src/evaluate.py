"""
Model Değerlendirme Betiği
--------------------------
Eğitilmiş modelleri test verisi üzerinde değerlendirir.
Accuracy, Precision, Recall, F1-Score ve Confusion Matrix üretir.
Sonuçlar results/ dizinine kaydedilir.
"""

import os
import sys
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from preprocessing import conllu_oku, tokenleri_ve_etiketleri_ayir
from feature_extraction import veri_setini_ozelliklendir

warnings.filterwarnings('ignore')

# ── Varsayılan Yollar ──
TEST_DOSYASI = os.path.join('data', 'test', 'test.conllu')
MODEL_DIZINI = 'models'
SONUC_DIZINI = 'results'


def modeli_yukle(model_adi: str):
    """Kaydedilmiş modeli diskten yükler."""
    dosya_yolu = os.path.join(MODEL_DIZINI, f'{model_adi}.pkl')
    with open(dosya_yolu, 'rb') as f:
        return pickle.load(f)


def karisiklik_matrisi_ciz(
    gercek: np.ndarray,
    tahmin: np.ndarray,
    siniflar: list,
    baslik: str,
    kayit_yolu: str
):
    """
    Karışıklık matrisini ısı haritası olarak çizer ve kaydeder.
    
    Parametreler:
        gercek: Gerçek etiketler
        tahmin: Tahmin edilen etiketler
        siniflar: UPOS sınıf isimleri
        baslik: Grafik başlığı
        kayit_yolu: Görselin kaydedileceği dosya yolu
    """
    cm = confusion_matrix(gercek, tahmin, labels=siniflar)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=siniflar,
        yticklabels=siniflar,
    )
    plt.title(f'Karışıklık Matrisi - {baslik}', fontsize=14, fontweight='bold')
    plt.xlabel('Tahmin Edilen', fontsize=12)
    plt.ylabel('Gerçek', fontsize=12)
    plt.tight_layout()
    plt.savefig(kayit_yolu, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Karışıklık matrisi kaydedildi: {kayit_yolu}")


def model_karsilastirma_grafigi(sonuclar: dict, kayit_yolu: str):
    """
    Tüm modellerin metriklerini yan yana çubuk grafik olarak çizer.
    
    Parametreler:
        sonuclar: {model_adi: {metrik: deger}} sözlüğü
        kayit_yolu: Görselin kaydedileceği dosya yolu
    """
    model_adlari = list(sonuclar.keys())
    metrikler = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    renkler = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']

    x = np.arange(len(model_adlari))
    genislik = 0.2

    plt.figure(figsize=(12, 6))
    for i, (metrik, renk) in enumerate(zip(metrikler, renkler)):
        degerler = [sonuclar[m][metrik] for m in model_adlari]
        cubuklar = plt.bar(x + i * genislik, degerler, genislik, label=metrik, color=renk)
        # Çubuk üstüne değer yaz
        for cubuk, deger in zip(cubuklar, degerler):
            plt.text(
                cubuk.get_x() + cubuk.get_width() / 2, cubuk.get_height() + 0.01,
                f'{deger:.3f}', ha='center', va='bottom', fontsize=8
            )

    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Skor', fontsize=12)
    plt.title('Model Karşılaştırması', fontsize=14, fontweight='bold')
    plt.xticks(x + genislik * 1.5, model_adlari, rotation=15)
    plt.ylim(0, 1.15)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(kayit_yolu, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Karşılaştırma grafiği kaydedildi: {kayit_yolu}")


def degerlendirme_yap(test_dosyasi: str = TEST_DOSYASI):
    """
    Tüm modelleri test verisi üzerinde değerlendirir ve sonuçları raporlar.
    """
    print("=" * 60)
    print("  MORFOLOJIK ÇÖZÜMLEME — MODEL DEĞERLENDİRME")
    print("=" * 60)
    print(f"  Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()

    # ── Test Verisini Oku ──
    print("[1/3] Test verisi okunuyor...")
    if not os.path.exists(test_dosyasi):
        print(f"  HATA: '{test_dosyasi}' dosyası bulunamadı!")
        sys.exit(1)

    cumleler = conllu_oku(test_dosyasi)
    tokenler, etiketler = tokenleri_ve_etiketleri_ayir(cumleler)
    X_test, y_test = veri_setini_ozelliklendir(tokenler, etiketler)
    print(f"  → {len(cumleler)} cümle, {len(y_test)} token yüklendi")
    print()

    # ── Sınıf İsimleri ──
    siniflar = sorted(set(y_test))
    os.makedirs(SONUC_DIZINI, exist_ok=True)

    # ── Modelleri Değerlendir ──
    print("[2/3] Modeller değerlendiriliyor...")
    sonuclar = {}

    # Sklearn modelleri (NB, MLP, LR)
    sklearn_modeller = {
        'Naive Bayes': 'naive_bayes',
        'MLP (128-64-32)': 'mlp',
        'Logistic Regression': 'logistic_regression',
    }

    for gorunen_ad, dosya_adi in sklearn_modeller.items():
        print(f"\n  ▸ {gorunen_ad}")
        model = modeli_yukle(dosya_adi)
        y_tahmin = model.predict(X_test)

        acc = accuracy_score(y_test, y_tahmin)
        pre = precision_score(y_test, y_tahmin, average='macro', zero_division=0)
        rec = recall_score(y_test, y_tahmin, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_tahmin, average='macro', zero_division=0)

        sonuclar[gorunen_ad] = {
            'Accuracy': acc, 'Precision': pre, 'Recall': rec, 'F1-Score': f1
        }

        print(f"    Accuracy : %{acc * 100:.1f}")
        print(f"    Precision: {pre:.3f}")
        print(f"    Recall   : {rec:.3f}")
        print(f"    F1-Score : {f1:.3f}")

        karisiklik_matrisi_ciz(
            y_test, y_tahmin, siniflar, gorunen_ad,
            os.path.join(SONUC_DIZINI, f'cm_{dosya_adi}.png')
        )

    # HMM (dizilim tabanlı — ayrı değerlendirme)
    print(f"\n  ▸ HMM (Gizli Markov Modeli)")
    hmm_modeli = modeli_yukle('hmm')
    hmm_tahminleri = hmm_modeli.tahmin_et(tokenler)

    # Düzleştir: cümle listesi → tek liste
    y_hmm = [etiket for cumle in hmm_tahminleri for etiket in cumle]
    y_gercek_hmm = [etiket for cumle in etiketler for etiket in cumle]

    acc = accuracy_score(y_gercek_hmm, y_hmm)
    pre = precision_score(y_gercek_hmm, y_hmm, average='macro', zero_division=0)
    rec = recall_score(y_gercek_hmm, y_hmm, average='macro', zero_division=0)
    f1 = f1_score(y_gercek_hmm, y_hmm, average='macro', zero_division=0)

    sonuclar['HMM'] = {
        'Accuracy': acc, 'Precision': pre, 'Recall': rec, 'F1-Score': f1
    }

    print(f"    Accuracy : %{acc * 100:.1f}")
    print(f"    Precision: {pre:.3f}")
    print(f"    Recall   : {rec:.3f}")
    print(f"    F1-Score : {f1:.3f}")

    karisiklik_matrisi_ciz(
        np.array(y_gercek_hmm), np.array(y_hmm), siniflar, 'HMM',
        os.path.join(SONUC_DIZINI, 'cm_hmm.png')
    )

    # ── Sonuç Grafikleri ──
    print(f"\n[3/3] Grafikler oluşturuluyor...")
    model_karsilastirma_grafigi(
        sonuclar,
        os.path.join(SONUC_DIZINI, 'model_karsilastirma.png')
    )

    # ── HMM Sınıf Bazında Rapor ──
    print("\n" + "─" * 60)
    print("  EN İYİ MODEL (HMM) — SINIF BAZINDA SONUÇLAR")
    print("─" * 60)
    print(classification_report(y_gercek_hmm, y_hmm, target_names=siniflar, zero_division=0))

    # ── Özet Tablo ──
    print("─" * 60)
    print("  GENEL SONUÇ TABLOSU")
    print("─" * 60)
    print(f"  {'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("  " + "─" * 65)
    for model_adi, metrikler in sonuclar.items():
        print(f"  {model_adi:<25} {metrikler['Accuracy']:>9.1%}"
              f" {metrikler['Precision']:>10.3f}"
              f" {metrikler['Recall']:>10.3f}"
              f" {metrikler['F1-Score']:>10.3f}")
    print("=" * 60)


if __name__ == '__main__':
    dosya = sys.argv[1] if len(sys.argv) > 1 else TEST_DOSYASI
    degerlendirme_yap(dosya)
