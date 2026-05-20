"""
Yardımcı Fonksiyonlar
---------------------
Proje genelinde kullanılan ortak araçlar.
"""

import os
import pickle
from typing import Any


def model_kaydet(model: Any, dosya_yolu: str):
    """Modeli pickle formatında diske kaydeder."""
    dizin = os.path.dirname(dosya_yolu)
    if dizin:
        os.makedirs(dizin, exist_ok=True)
    with open(dosya_yolu, 'wb') as f:
        pickle.dump(model, f)


def model_yukle(dosya_yolu: str) -> Any:
    """Daha önce kaydedilmiş bir modeli diskten yükler."""
    with open(dosya_yolu, 'rb') as f:
        return pickle.load(f)


def dizin_olustur(*yollar: str):
    """Verilen dizin yollarını oluşturur (varsa atlar)."""
    for yol in yollar:
        os.makedirs(yol, exist_ok=True)


def upos_renkleri() -> dict:
    """Grafiklerde kullanılmak üzere UPOS etiketleri için renk paleti döndürür."""
    return {
        'NOUN':  '#2196F3',
        'VERB':  '#4CAF50',
        'ADJ':   '#FF9800',
        'ADV':   '#9C27B0',
        'PRON':  '#E91E63',
        'DET':   '#00BCD4',
        'ADP':   '#795548',
        'CCONJ': '#607D8B',
        'PUNCT': '#9E9E9E',
    }
