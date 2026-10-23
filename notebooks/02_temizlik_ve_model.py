# %% [markdown]
# # 02 - Temizlik + Random Forest baz çizgisi (CIC-IDS2017)
# Hücreleri yukarıdan aşağıya sırayla çalıştır.

# %%
import glob
from pathlib import Path
import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parent.parent
VERI = KOK / "data" / "cicids2017" / "MachineLearningCVE"
SONUC = KOK / "results"
SONUC.mkdir(exist_ok=True)

def yukle(dosya):
    d = pd.read_csv(dosya, low_memory=False)
    d.columns = d.columns.str.strip()
    ozellikler = [c for c in d.columns if c != "Label"]
    d[ozellikler] = d[ozellikler].astype("float32")
    return d

df = pd.concat((yukle(f) for f in sorted(glob.glob(str(VERI / "*.csv")))), ignore_index=True)
print("ham boyut:", df.shape)

# %% [markdown]
# ## Temizlik
# inf -> NaN, NaN satırlarını at, yinelenen satırları at, bozuk etiket karakterini düzelt.

# %%
ozellikler = [c for c in df.columns if c != "Label"]
for c in ozellikler:
    v = df[c].to_numpy()
    v[np.isinf(v)] = np.nan          # inf değerlerini NaN yap (yerinde, kopyasız)

onceki = len(df)
df = df.dropna().reset_index(drop=True)
print("NaN/inf nedeniyle atılan satır:", onceki - len(df))

onceki = len(df)
df = df[~pd.util.hash_pandas_object(df, index=False).duplicated()].reset_index(drop=True)
print("yinelenen nedeniyle atılan satır:", onceki - len(df))

df["Label"] = df["Label"].str.replace("�", "-", regex=False)   # "Web Attack � XSS" düzeltmesi
print("temiz boyut:", df.shape)
print(df["Label"].value_counts())

# %% [markdown]
# ## Örnekleme
# BENIGN ve büyük sınıfları sınırlıyoruz (hız için), nadir sınıfların hepsi kalır.
# Bunu raporunda mutlaka belirt: sınıf dağılımı değişti.

# %%
SINIR = 100_000
parcalar = [g.sample(min(len(g), SINIR), random_state=42) for _, g in df.groupby("Label")]
ornek = pd.concat(parcalar).sample(frac=1, random_state=42).reset_index(drop=True)
print(ornek["Label"].value_counts())

# %% [markdown]
# ## Eğitim / test ayırma ve Random Forest

# %%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

X = ornek.drop(columns="Label")
y = ornek["Label"]

# Çok nadir sınıflarda stratify hata verebilir; en az 2 örneği olanlar için
X_egitim, X_test, y_egitim, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)
print("egitim:", X_egitim.shape, "test:", X_test.shape)

# %%
model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
model.fit(X_egitim, y_egitim)
tahmin = model.predict(X_test)

# %%
rapor = classification_report(y_test, tahmin, digits=4, zero_division=0)
print(rapor)
print("makro F1:", round(f1_score(y_test, tahmin, average="macro"), 4))
(SONUC / "cicids2017_rf_rapor.txt").write_text(rapor, encoding="utf-8")

# %%
import matplotlib.pyplot as plt
siniflar = sorted(y.unique())
cm = confusion_matrix(y_test, tahmin, labels=siniflar)
cm_oran = cm / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cm_oran, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(siniflar)), siniflar, rotation=90)
ax.set_yticks(range(len(siniflar)), siniflar)
ax.set_xlabel("Tahmin"); ax.set_ylabel("Gerçek")
ax.set_title("CIC-IDS2017 Random Forest - satır bazlı oran")
fig.colorbar(im)
fig.tight_layout()
fig.savefig(SONUC / "cicids2017_rf_karisiklik.png", dpi=150)
plt.show()
