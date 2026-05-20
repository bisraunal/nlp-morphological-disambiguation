"""
Veri Ön İşleme Modülü
---------------------
CoNLL-U formatındaki veriyi okuma, ayrıştırma ve istatistik çıkarma işlemleri.
Universal Dependencies (UD) standartlarına uygun şekilde çalışır.
"""

import os
from typing import List, Dict, Tuple


def conllu_oku(dosya_yolu: str) -> List[List[Dict]]:
    """
    CoNLL-U formatındaki dosyayı okur ve cümle listesi döndürür.
    
    Her cümle, token sözlüklerinden oluşan bir listedir.
    Her token sözlüğü şu anahtarları içerir:
        - id: Kelimenin cümledeki sırası
        - form: Kelimenin orijinal hali
        - lemma: Kelimenin kökü
        - upos: Universal POS etiketi (NOUN, VERB, ADJ...)
        - xpos: Dile özgü POS etiketi
        - feats: Morfolojik özellikler (Case, Number, Tense...)
    
    Parametreler:
        dosya_yolu: CoNLL-U dosyasının tam yolu
    
    Döndürür:
        Cümle listesi (her cümle bir token listesi)
    """
    cumleler = []
    mevcut_cumle = []

    with open(dosya_yolu, 'r', encoding='utf-8') as f:
        for satir in f:
            satir = satir.strip()

            # Yorum satırlarını atla (# ile başlayan)
            if satir.startswith('#'):
                continue

            # Boş satır cümle sonunu belirtir
            if not satir:
                if mevcut_cumle:
                    cumleler.append(mevcut_cumle)
                    mevcut_cumle = []
                continue

            alanlar = satir.split('\t')
            if len(alanlar) >= 6:
                token = {
                    'id': alanlar[0],
                    'form': alanlar[1],
                    'lemma': alanlar[2],
                    'upos': alanlar[3],
                    'xpos': alanlar[4] if len(alanlar) > 4 else '_',
                    'feats': alanlar[5] if len(alanlar) > 5 else '_',
                }
                mevcut_cumle.append(token)

        # Dosya sonundaki son cümleyi de ekle
        if mevcut_cumle:
            cumleler.append(mevcut_cumle)

    return cumleler


def tokenleri_ve_etiketleri_ayir(cumleler: List[List[Dict]]) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Cümlelerden token (kelime) ve UPOS etiket listelerini ayrı ayrı çıkarır.
    
    Döndürür:
        (token_listesi, etiket_listesi) ikilisi
    """
    tum_tokenler = []
    tum_etiketler = []

    for cumle in cumleler:
        tokenler = [t['form'] for t in cumle]
        etiketler = [t['upos'] for t in cumle]
        tum_tokenler.append(tokenler)
        tum_etiketler.append(etiketler)

    return tum_tokenler, tum_etiketler


def veri_seti_istatistikleri(cumleler: List[List[Dict]]) -> Dict:
    """
    Veri seti hakkında temel istatistikleri hesaplar.
    
    Döndürür:
        Sözlük: cumle_sayisi, token_sayisi, ortalama_token, sinif_sayisi, upos_dagilimi
    """
    toplam_token = sum(len(c) for c in cumleler)
    upos_sayilari = {}

    for cumle in cumleler:
        for token in cumle:
            etiket = token['upos']
            upos_sayilari[etiket] = upos_sayilari.get(etiket, 0) + 1

    return {
        'cumle_sayisi': len(cumleler),
        'token_sayisi': toplam_token,
        'ortalama_token': round(toplam_token / len(cumleler), 1) if cumleler else 0,
        'sinif_sayisi': len(upos_sayilari),
        'upos_dagilimi': dict(sorted(upos_sayilari.items(), key=lambda x: -x[1])),
    }


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Kullanım: python preprocessing.py <conllu_dosyası>")
        sys.exit(1)

    dosya = sys.argv[1]
    cumleler = conllu_oku(dosya)
    istatistik = veri_seti_istatistikleri(cumleler)

    print(f"Cümle sayısı     : {istatistik['cumle_sayisi']}")
    print(f"Token sayısı     : {istatistik['token_sayisi']}")
    print(f"Ort. token/cümle : {istatistik['ortalama_token']}")
    print(f"UPOS sınıf sayısı: {istatistik['sinif_sayisi']}")
    print("\nUPOS Dağılımı:")
    for etiket, sayi in istatistik['upos_dagilimi'].items():
        yuzde = sayi / istatistik['token_sayisi'] * 100
        print(f"  {etiket:8s}: {sayi:4d} (%{yuzde:.1f})")
