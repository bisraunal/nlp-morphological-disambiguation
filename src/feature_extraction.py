"""
Özellik Çıkarımı Modülü
-----------------------
Her token için toplam 323 özellik üretir.
Bağlamsal pencere yöntemi (window size = 2) ile hedef kelimenin
kendisi ve ±2 komşu kelimesinin bilgileri birleştirilir.

Özellik grupları:
    - Kelime Düzeyi : Uzunluk, büyük harf, rakam, noktalama, ünlü-ünsüz oranı
    - Morfolojik    : 36 yaygın Türkçe ek için binary tespitler
    - N-gram        : İlk/son 1-2 karakter kodları
    - Bağlamsal     : ±2 pencere içindeki kelimelerin özellikleri
    - Konumsal      : Cümle içi pozisyon, baş/son kontrolü
"""

import numpy as np
from typing import List

# ── Türkçe Ek Listesi (36 adet) ──────────────────────────────
TURKCE_EKLER = [
    'lar', 'ler', 'dan', 'den', 'tan', 'ten',
    'da', 'de', 'ta', 'te', 'ın', 'in', 'un', 'ün',
    'nın', 'nin', 'nun', 'nün', 'ya', 'ye',
    'yor', 'mış', 'miş', 'muş', 'müş',
    'dı', 'di', 'du', 'dü', 'tı', 'ti', 'tu', 'tü',
    'acak', 'ecek', 'arak', 'erek',
]

UNLULER = set('aeıioöuüAEIİOÖUÜ')
UNSUZLER = set('bcçdfgğhjklmnprsştvyzBCÇDFGĞHJKLMNPRSŞTVYZ')


def kelime_ozellikleri(kelime: str) -> List[float]:
    """
    Tek bir kelime için temel düzey özellikleri çıkarır.
    
    Çıkarılan özellikler:
        - Normalize uzunluk
        - Büyük harfle başlama / tamamen büyük harf
        - Rakam içerme / tamamen rakam olma
        - Noktalama işareti olma
        - Ünlü ve ünsüz oranları
        - Son harf ünlü mü kontrolü
        - İlk/son 1-2 karakter n-gram kodları
        - 36 Türkçe ek binary tespiti
    
    Döndürür:
        Özellik değerleri listesi (float)
    """
    ozellikler = []

    # ── Temel Özellikler ──
    ozellikler.append(len(kelime) / 20.0)                                    # Normalize uzunluk
    ozellikler.append(1.0 if kelime[0].isupper() else 0.0)                   # Büyük harfle başlıyor mu
    ozellikler.append(1.0 if kelime.isupper() else 0.0)                      # Tamamen büyük harf mi
    ozellikler.append(1.0 if any(c.isdigit() for c in kelime) else 0.0)      # Rakam içeriyor mu
    ozellikler.append(1.0 if kelime.isdigit() else 0.0)                      # Tamamen rakam mı
    ozellikler.append(1.0 if not kelime.isalnum() else 0.0)                  # Noktalama işareti mi

    # ── Ünlü / Ünsüz Oranları ──
    unlu_sayisi = sum(1 for c in kelime if c in UNLULER)
    unsuz_sayisi = sum(1 for c in kelime if c in UNSUZLER)
    ozellikler.append(unlu_sayisi / max(len(kelime), 1))                     # Ünlü oranı
    ozellikler.append(unsuz_sayisi / max(len(kelime), 1))                    # Ünsüz oranı
    ozellikler.append(1.0 if kelime[-1] in UNLULER else 0.0)                 # Son harf ünlü mü

    # ── N-gram Özellikleri ──
    ozellikler.append(ord(kelime[0]) / 256.0)                                # İlk karakter kodu
    ozellikler.append(ord(kelime[-1]) / 256.0)                               # Son karakter kodu
    ozellikler.append((ord(kelime[0]) + ord(kelime[1])) / 512.0
                      if len(kelime) > 1 else 0.0)                           # İlk 2 karakter
    ozellikler.append((ord(kelime[-2]) + ord(kelime[-1])) / 512.0
                      if len(kelime) > 1 else 0.0)                           # Son 2 karakter

    # ── Türkçe Ek Tespiti (Binary) ──
    kelime_kucuk = kelime.lower()
    for ek in TURKCE_EKLER:
        ozellikler.append(1.0 if kelime_kucuk.endswith(ek) else 0.0)

    return ozellikler


def token_icin_ozellik_cikar(
    tokenler: List[str],
    indeks: int,
    pencere_boyutu: int = 2
) -> np.ndarray:
    """
    Bir token için bağlamsal pencere dahil tüm özellikleri çıkarır.
    
    Pencere yöntemi: Hedef token ve ±pencere_boyutu kadar komşu kelimelerin
    kelime düzeyindeki özellikleri birleştirilerek tek bir vektör oluşturulur.
    Pencere dışına taşan pozisyonlar sıfır ile doldurulur (padding).
    
    Parametreler:
        tokenler: Cümledeki tüm kelimeler
        indeks: Hedef tokenın cümle içindeki konumu
        pencere_boyutu: Kaç komşu kelime dahil edilecek (varsayılan: 2)
    
    Döndürür:
        Özellik vektörü (numpy dizisi)
    """
    tum_ozellikler = []

    # ── Bağlamsal Pencere ──
    # Her pozisyon (-2, -1, 0, +1, +2) için kelime özelliklerini topla
    bos_vektor_uzunlugu = len(kelime_ozellikleri("test"))

    for kayma in range(-pencere_boyutu, pencere_boyutu + 1):
        konum = indeks + kayma

        if 0 <= konum < len(tokenler):
            tum_ozellikler.extend(kelime_ozellikleri(tokenler[konum]))
        else:
            # Cümle sınırı dışı: sıfır vektörle doldur
            tum_ozellikler.extend([0.0] * bos_vektor_uzunlugu)

    # ── Konumsal Özellikler ──
    tum_ozellikler.append(indeks / max(len(tokenler) - 1, 1))   # Göreli konum (0..1)
    tum_ozellikler.append(1.0 if indeks == 0 else 0.0)          # Cümle başı mı
    tum_ozellikler.append(1.0 if indeks == len(tokenler) - 1 else 0.0)  # Cümle sonu mu

    return np.array(tum_ozellikler, dtype=np.float64)


def veri_setini_ozelliklendir(
    cumle_tokenleri: List[List[str]],
    cumle_etiketleri: List[List[str]],
    pencere_boyutu: int = 2
) -> tuple:
    """
    Tüm veri seti için özellik matrisi ve etiket vektörü oluşturur.
    
    Parametreler:
        cumle_tokenleri: Her cümle için token listesi
        cumle_etiketleri: Her cümle için UPOS etiket listesi
        pencere_boyutu: Bağlamsal pencere boyutu
    
    Döndürür:
        (X, y) ikilisi — X: özellik matrisi, y: etiket dizisi
    """
    X_listesi = []
    y_listesi = []

    for tokenler, etiketler in zip(cumle_tokenleri, cumle_etiketleri):
        for i, etiket in enumerate(etiketler):
            ozellik_vektoru = token_icin_ozellik_cikar(tokenler, i, pencere_boyutu)
            X_listesi.append(ozellik_vektoru)
            y_listesi.append(etiket)

    X = np.array(X_listesi)
    y = np.array(y_listesi)

    return X, y


if __name__ == '__main__':
    # Hızlı test
    ornek_tokenler = ["Dün", "akşam", "toplantıdan", "erken", "çıktı", "."]
    ornek_etiketler = ["ADV", "ADV", "NOUN", "ADV", "VERB", "PUNCT"]

    X, y = veri_setini_ozelliklendir([ornek_tokenler], [ornek_etiketler])
    print(f"Özellik matrisi boyutu: {X.shape}")
    print(f"Etiket sayısı         : {len(y)}")
    print(f"Örnek etiketler       : {y.tolist()}")
    print(f"Token başına özellik  : {X.shape[1]}")
