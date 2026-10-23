# %% [markdown]
# # 01 - CIC-IDS2017 ilk keşif
# VS Code'da her `# %%` satırının üstündeki "Run Cell" ile hücre hücre çalıştırabilirsin.
# Not: bellek tasarrufu için sayılar float32'ye çevrilir (yaklaşık yarı RAM).

# %%
import glob
from pathlib import Path
import numpy as np
import pandas as pd

VERI = Path(__file__).resolve().parent.parent / "data" / "cicids2017" / "MachineLearningCVE"
dosyalar = sorted(glob.glob(str(VERI / "*.csv")))
print(len(dosyalar), "dosya bulundu")

# %%
def yukle(dosya):
    d = pd.read_csv(dosya, low_memory=False)
    d.columns = d.columns.str.strip()          # sütun adlarındaki boşluklar
    ozellikler = [c for c in d.columns if c != "Label"]
    d[ozellikler] = d[ozellikler].astype("float32")   # RAM'i yarıya indirir
    return d

df = pd.concat((yukle(f) for f in dosyalar), ignore_index=True)
df["Label"] = df["Label"].astype("category")
print("boyut:", df.shape)
print("bellek (GB):", round(df.memory_usage(deep=True).sum() / 1e9, 2))

# %%
print(df["Label"].value_counts())

# %%
# Kopya oluşturmadan, sütun sütun kontrol (MemoryError'u önler)
ozellikler = [c for c in df.columns if c != "Label"]
nan_sayisi = sum(int(df[c].isna().sum()) for c in ozellikler)
inf_sayisi = sum(int(np.isinf(df[c].to_numpy()).sum()) for c in ozellikler)
print("NaN:", nan_sayisi)
print("inf:", inf_sayisi)

# %%
# Yinelenen satırlar: satır başına özet (hash) alarak, bellek dostu
satir_ozeti = pd.util.hash_pandas_object(df, index=False)
print("yinelenen satir:", int(satir_ozeti.duplicated().sum()))
