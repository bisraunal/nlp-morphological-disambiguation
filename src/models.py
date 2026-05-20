"""
Makine Öğrenmesi Modelleri
--------------------------
Dört farklı model tanımlanmıştır:
    1. Naive Bayes (Gaussian)        — Olasılıksal sınıflandırıcı
    2. MLP (128-64-32)               — Çok katmanlı algılayıcı
    3. Logistic Regression           — Çok sınıflı doğrusal model
    4. HMM (Hidden Markov Model)     — Gizli Markov modeli (özel uygulama)

İlk üç model scikit-learn ile, HMM ise sıfırdan yazılmıştır.
"""

import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple

from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression


# ╔══════════════════════════════════════════════════════════════╗
# ║  1. Naive Bayes (Gaussian)                                  ║
# ╚══════════════════════════════════════════════════════════════╝

def naive_bayes_olustur() -> GaussianNB:
    """
    Gaussian Naive Bayes sınıflandırıcısı oluşturur.
    
    Bayes teoremine dayalı olasılıksal bir modeldir.
    Özellikler arası bağımsızlık varsayımı yapar.
    Gaussian varyantı sürekli değişkenler için normal dağılım varsayar.
    
    Avantaj : Hızlı eğitim
    Dezavantaj: Bağımsızlık varsayımı gerçek veride genelde sağlanmaz
    """
    return GaussianNB()


# ╔══════════════════════════════════════════════════════════════╗
# ║  2. MLP — Çok Katmanlı Algılayıcı (128-64-32)              ║
# ╚══════════════════════════════════════════════════════════════╝

def mlp_olustur(
    katmanlar: tuple = (128, 64, 32),
    maks_iterasyon: int = 500,
    rastgele_tohum: int = 42
) -> MLPClassifier:
    """
    Çok Katmanlı Algılayıcı (MLP) sınıflandırıcısı oluşturur.
    
    3 gizli katmandan (128-64-32 nöron) oluşan ileri beslemeli yapay sinir ağı.
    - Aktivasyon  : ReLU
    - Optimizasyon: Adam
    - Öğrenme oranı: Adaptive (early stopping ile)
    
    Parametreler:
        katmanlar: Gizli katman nöron sayıları (varsayılan: 128-64-32)
        maks_iterasyon: Maksimum epoch sayısı
        rastgele_tohum: Tekrarlanabilirlik için rastgele tohum
    
    Avantaj : Doğrusal olmayan ilişkileri yakalayabilir
    Dezavantaj: Çok veri gerektirir, eğitim yavaş olabilir
    """
    return MLPClassifier(
        hidden_layer_sizes=katmanlar,
        activation='relu',
        solver='adam',
        learning_rate='adaptive',
        max_iter=maks_iterasyon,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=rastgele_tohum,
        verbose=False,
    )


# ╔══════════════════════════════════════════════════════════════╗
# ║  3. Logistic Regression (Multinomial)                       ║
# ╚══════════════════════════════════════════════════════════════╝

def logistic_regression_olustur(
    C: float = 1.0,
    maks_iterasyon: int = 1000,
    rastgele_tohum: int = 42
) -> LogisticRegression:
    """
    Multinomial Logistic Regression sınıflandırıcısı oluşturur.
    
    Çok sınıflı problemler için softmax fonksiyonu kullanan doğrusal model.
    - Optimizasyon: L-BFGS
    - Düzenlileştirme: C=1.0
    
    Parametreler:
        C: Düzenlileştirme parametresi (küçük → daha fazla düzenlileştirme)
        maks_iterasyon: Maksimum iterasyon sayısı
        rastgele_tohum: Tekrarlanabilirlik için rastgele tohum
    
    Avantaj : Yorumlanabilir, kararlı
    Dezavantaj: Doğrusal karar sınırı
    """
    return LogisticRegression(
        C=C,
        solver='lbfgs',
        multi_class='multinomial',
        max_iter=maks_iterasyon,
        random_state=rastgele_tohum,
    )


# ╔══════════════════════════════════════════════════════════════╗
# ║  4. HMM — Gizli Markov Modeli                              ║
# ╚══════════════════════════════════════════════════════════════╝

class GizliMarkovModeli:
    """
    Basitleştirilmiş Gizli Markov Modeli (HMM) uygulaması.
    
    Dizilimdeki bağımlılıkları modelleyen olasılıksal bir yaklaşımdır.
    - Geçiş olasılıkları : P(etiket_i | etiket_i-1)
    - Emisyon olasılıkları: P(kelime | etiket)
    
    Çözümleme sırasında Viterbi benzeri greedy algoritma kullanılır.
    Dizilim bilgisini kullandığı için token bazlı modellere göre avantajlıdır.
    """

    def __init__(self, yumusatma: float = 1e-6):
        """
        Parametreler:
            yumusatma: Sıfır olasılığı önlemek için Laplace yumuşatma değeri
        """
        self.yumusatma = yumusatma
        self.gecis_olasikliklari = {}    # P(etiket | önceki_etiket)
        self.emisyon_olasikliklari = {}  # P(kelime | etiket)
        self.baslangic_olasikliklari = {}  # P(ilk etiket)
        self.tum_etiketler = set()

    def egit(self, cumle_tokenleri: List[List[str]], cumle_etiketleri: List[List[str]]):
        """
        Eğitim verisinden geçiş ve emisyon olasılıklarını hesaplar.
        
        Parametreler:
            cumle_tokenleri: Her cümle için token listesi
            cumle_etiketleri: Her cümle için UPOS etiket listesi
        """
        gecis_sayaci = defaultdict(lambda: defaultdict(int))
        emisyon_sayaci = defaultdict(lambda: defaultdict(int))
        baslangic_sayaci = defaultdict(int)
        etiket_sayaci = defaultdict(int)

        for tokenler, etiketler in zip(cumle_tokenleri, cumle_etiketleri):
            if not etiketler:
                continue

            # İlk etiketin başlangıç olasılığı
            baslangic_sayaci[etiketler[0]] += 1

            for i, (token, etiket) in enumerate(zip(tokenler, etiketler)):
                self.tum_etiketler.add(etiket)
                etiket_sayaci[etiket] += 1
                emisyon_sayaci[etiket][token.lower()] += 1

                # Geçiş olasılığı: mevcut etiket → sonraki etiket
                if i > 0:
                    onceki_etiket = etiketler[i - 1]
                    gecis_sayaci[onceki_etiket][etiket] += 1

        # ── Olasılıkları Hesapla ──
        toplam_baslangic = sum(baslangic_sayaci.values())
        self.baslangic_olasikliklari = {
            e: sayi / toplam_baslangic for e, sayi in baslangic_sayaci.items()
        }

        # Geçiş olasılıkları (Laplace yumuşatmalı)
        for onceki in self.tum_etiketler:
            self.gecis_olasikliklari[onceki] = {}
            toplam = sum(gecis_sayaci[onceki].values()) + self.yumusatma * len(self.tum_etiketler)
            for sonraki in self.tum_etiketler:
                sayi = gecis_sayaci[onceki].get(sonraki, 0)
                self.gecis_olasikliklari[onceki][sonraki] = (sayi + self.yumusatma) / toplam

        # Emisyon olasılıkları (Laplace yumuşatmalı)
        for etiket in self.tum_etiketler:
            self.emisyon_olasikliklari[etiket] = {}
            kelime_haznesi = set()
            for e in self.tum_etiketler:
                kelime_haznesi.update(emisyon_sayaci[e].keys())

            toplam = sum(emisyon_sayaci[etiket].values()) + self.yumusatma * (len(kelime_haznesi) + 1)
            for kelime, sayi in emisyon_sayaci[etiket].items():
                self.emisyon_olasikliklari[etiket][kelime] = (sayi + self.yumusatma) / toplam
            # Bilinmeyen kelimeler için varsayılan olasılık
            self.emisyon_olasikliklari[etiket]['<UNK>'] = self.yumusatma / toplam

    def _emisyon_al(self, etiket: str, kelime: str) -> float:
        """Bir kelimenin belirli bir etikete ait emisyon olasılığını döndürür."""
        kelime_kucuk = kelime.lower()
        if kelime_kucuk in self.emisyon_olasikliklari.get(etiket, {}):
            return self.emisyon_olasikliklari[etiket][kelime_kucuk]
        return self.emisyon_olasikliklari.get(etiket, {}).get('<UNK>', self.yumusatma)

    def tahmin_et(self, cumle_tokenleri: List[List[str]]) -> List[List[str]]:
        """
        Verilen cümleler için en olası etiket dizilimlerini tahmin eder.
        Greedy Viterbi benzeri bir algoritma kullanır.
        
        Parametreler:
            cumle_tokenleri: Her cümle için token listesi
        
        Döndürür:
            Her cümle için tahmin edilen etiket listesi
        """
        tum_tahminler = []

        for tokenler in cumle_tokenleri:
            if not tokenler:
                tum_tahminler.append([])
                continue

            tahminler = []
            etiket_listesi = sorted(self.tum_etiketler)

            # İlk token: başlangıç olasılığı × emisyon olasılığı
            en_iyi_etiket = max(
                etiket_listesi,
                key=lambda e: (
                    self.baslangic_olasikliklari.get(e, self.yumusatma)
                    * self._emisyon_al(e, tokenler[0])
                )
            )
            tahminler.append(en_iyi_etiket)

            # Sonraki tokenler: geçiş × emisyon (greedy)
            for i in range(1, len(tokenler)):
                onceki = tahminler[-1]
                en_iyi_etiket = max(
                    etiket_listesi,
                    key=lambda e: (
                        self.gecis_olasikliklari.get(onceki, {}).get(e, self.yumusatma)
                        * self._emisyon_al(e, tokenler[i])
                    )
                )
                tahminler.append(en_iyi_etiket)

            tum_tahminler.append(tahminler)

        return tum_tahminler


# ── Tüm Modelleri Topluca Oluşturan Yardımcı Fonksiyon ──

def tum_modelleri_olustur() -> Dict:
    """
    Projede kullanılan dört modelin hepsini oluşturup bir sözlükte döndürür.
    
    Döndürür:
        {"Naive Bayes": model, "MLP": model, "Logistic Regression": model, "HMM": model}
    """
    return {
        "Naive Bayes": naive_bayes_olustur(),
        "MLP (128-64-32)": mlp_olustur(),
        "Logistic Regression": logistic_regression_olustur(),
        "HMM": GizliMarkovModeli(),
    }
