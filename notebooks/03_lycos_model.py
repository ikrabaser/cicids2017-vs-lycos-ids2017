# %% [markdown]
# # 03 - LycoS-IDS2017: aynı hat, aynı model, sonra karşılaştırma
# 02 ile aynı adımlar: temizlik -> sınıf başına en fazla 100.000 örnek -> %70/%30 -> Random Forest.
# Hücreleri yukarıdan aşağıya sırayla çalıştır.

# %%
import glob, re
from pathlib import Path
import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parent.parent
VERI = KOK / "data" / "lycos2017" / "lycos-ids2017" / "lycos-ids2017"
SONUC = KOK / "results"
SONUC.mkdir(exist_ok=True)

dosyalar = sorted(glob.glob(str(VERI / "*.csv")))
print(len(dosyalar), "dosya bulundu")

# Modele GİRMEYECEK kimlik sütunları (IP/zaman/akış kimliği -> "ezber" riski)
ATILAN = ["flow_id", "src_addr", "dst_addr", "src_port", "timestamp"]

def yukle(dosya):
    baslik = pd.read_csv(dosya, nrows=0).columns.tolist()
    ozellikler = [c for c in baslik if c not in ATILAN + ["label"]]
    tipler = {c: "float32" for c in ozellikler}
    return pd.read_csv(dosya, usecols=ozellikler + ["label"], dtype=tipler)

df = pd.concat((yukle(f) for f in dosyalar), ignore_index=True)
print("ham boyut:", df.shape)
print("ozellik sayisi:", df.shape[1] - 1)

# %% [markdown]
# ## Temizlik (02 ile aynı)

# %%
ozellikler = [c for c in df.columns if c != "label"]
for c in ozellikler:
    v = df[c].to_numpy()
    v[np.isinf(v)] = np.nan

onceki = len(df)
df = df.dropna().reset_index(drop=True)
print("NaN/inf nedeniyle atılan satır:", onceki - len(df))

onceki = len(df)
df = df[~pd.util.hash_pandas_object(df, index=False).duplicated()].reset_index(drop=True)
print("yinelenen nedeniyle atılan satır:", onceki - len(df))

print("temiz boyut:", df.shape)
print(df["label"].value_counts())

# %% [markdown]
# ## Örnekleme (02 ile aynı: sınıf başına en fazla 100.000)

# %%
SINIR = 100_000
parcalar = [g.sample(min(len(g), SINIR), random_state=42) for _, g in df.groupby("label")]
ornek = pd.concat(parcalar).sample(frac=1, random_state=42).reset_index(drop=True)
print(ornek["label"].value_counts())

# %% [markdown]
# ## Eğitim / test ve Random Forest

# %%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

X = ornek.drop(columns="label")
y = ornek["label"]
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
makro = f1_score(y_test, tahmin, average="macro")
print("makro F1:", round(makro, 4))
(SONUC / "lycos2017_rf_rapor.txt").write_text(rapor, encoding="utf-8")

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
ax.set_title("LycoS-IDS2017 Random Forest - satır bazlı oran")
fig.colorbar(im); fig.tight_layout()
fig.savefig(SONUC / "lycos2017_rf_karisiklik.png", dpi=150)
plt.show()

# %% [markdown]
# ## Karşılaştırma: CIC-IDS2017 (02) ile LycoS-IDS2017 (bu dosya)
# Etiket adları farklı olduğu için eşleştiriyoruz. Infiltration LycoS'ta yok -> karşılaştırma dışı.

# %%
ESLEME = {
    "BENIGN": "benign", "Bot": "bot", "DDoS": "ddos",
    "DoS GoldenEye": "dos_goldeneye", "DoS Hulk": "dos_hulk",
    "DoS Slowhttptest": "dos_slowhttptest", "DoS slowloris": "dos_slowloris",
    "FTP-Patator": "ftp_patator", "Heartbleed": "heartbleed", "PortScan": "portscan",
    "SSH-Patator": "ssh_patator",
    "Web Attack - Brute Force": "webattack_bruteforce",
    "Web Attack - Sql Injection": "webattack_sql_injection",
    "Web Attack - XSS": "webattack_xss",
}

def rapor_oku(metin):
    """classification_report metninden {sınıf: (precision, recall, f1, destek)} çıkarır."""
    sonuc = {}
    for satir in metin.splitlines():
        m = re.match(r"^\s*(.+?)\s+([01]\.\d+)\s+([01]\.\d+)\s+([01]\.\d+)\s+(\d+)\s*$", satir)
        if m and m.group(1) not in ("accuracy", "macro avg", "weighted avg"):
            sonuc[m.group(1)] = tuple(map(float, m.groups()[1:4])) + (int(m.group(5)),)
    return sonuc

cic = rapor_oku((SONUC / "cicids2017_rf_rapor.txt").read_text(encoding="utf-8"))
lyc = rapor_oku(rapor)
if not set(ESLEME.values()) & set(lyc):
    raise RuntimeError(
        "LycoS sonuçları bulunamadı (sınıflar: %s). Muhtemelen 02'den kalan değişkenler var. "
        "Interactive penceresinde Restart yap ve bu dosyayı en baştan çalıştır." % list(lyc)[:3])

satirlar = []
for cic_ad, lyc_ad in ESLEME.items():
    if cic_ad in cic and lyc_ad in lyc:
        satirlar.append({"sinif": lyc_ad,
                         "CIC_F1": cic[cic_ad][2], "LycoS_F1": lyc[lyc_ad][2],
                         "CIC_destek": cic[cic_ad][3], "LycoS_destek": lyc[lyc_ad][3]})
kars = pd.DataFrame(satirlar)
kars["fark"] = (kars["LycoS_F1"] - kars["CIC_F1"]).round(4)
print(kars.to_string(index=False))
print("\nortak sınıflarda makro F1  CIC   :", round(kars["CIC_F1"].mean(), 4))
print("ortak sınıflarda makro F1  LycoS :", round(kars["LycoS_F1"].mean(), 4))
kars.to_csv(SONUC / "karsilastirma_f1.csv", index=False)

# %%
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(kars)); w = 0.4
ax.bar(x - w/2, kars["CIC_F1"], w, label="CIC-IDS2017")
ax.bar(x + w/2, kars["LycoS_F1"], w, label="LycoS-IDS2017")
ax.set_xticks(x, kars["sinif"], rotation=60, ha="right")
ax.set_ylabel("F1"); ax.set_ylim(0, 1.05)
ax.set_title("Sınıf bazlı F1: CIC-IDS2017 vs LycoS-IDS2017 (Random Forest)")
ax.legend(); fig.tight_layout()
fig.savefig(SONUC / "karsilastirma_f1.png", dpi=150)
plt.show()
