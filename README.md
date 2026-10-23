# CIC-IDS2017 ve LycoS-IDS2017 karşılaştırması

Aynı Random Forest hattı (temizlik, sınıf başına en fazla 100.000 örnek, %70/%30 tabakalı bölme, 100 ağaç) iki ağ saldırı veri setinde çalıştırılır ve sonuçlar karşılaştırılır.

## Klasörler
- `notebooks/` : `# %%` hücreli Python dosyaları (VS Code Interactive penceresinde ya da `python dosya.py` ile çalışır)
  - `01_kesif.py` : CIC-IDS2017 keşfi (boyut, etiketler, NaN/inf, yinelenenler)
  - `02_temizlik_ve_model.py` : CIC-IDS2017 temizlik + Random Forest
  - `03_lycos_model.py` : LycoS-IDS2017 aynı hat + karşılaştırma
  - `04_capraz_deney.py` : veri setleri arası (CIC↔LycoS) deney
- `results/` : sınıflandırma raporları, karışıklık matrisleri, karşılaştırma tabloları
- `data/` : **repoda yok**, aşağıdaki adımlarla kendin koy

## Veriyi yerleştirme
- CIC-IDS2017 `MachineLearningCSV` dosyaları (8 CSV) → `data/cicids2017/MachineLearningCVE/`
  (kaynak: https://www.unb.ca/cic/datasets/ids-2017.html)
- LycoS-IDS2017 paketi (etiketli CSV'ler) → `data/lycos2017/lycos-ids2017/lycos-ids2017/`
  (kaynak: https://lycos-ids.univ-lemans.fr/)

## Kurulum ve çalıştırma
```
pip install -r requirements.txt
python notebooks/02_temizlik_ve_model.py
python notebooks/03_lycos_model.py
python notebooks/04_capraz_deney.py
```
Hücreleri sırayla çalıştır; eğitimler birkaç dakika sürer.
