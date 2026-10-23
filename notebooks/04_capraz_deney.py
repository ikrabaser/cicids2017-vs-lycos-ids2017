# %% [markdown]
# # 04 - Çapraz veri seti deneyi (CIC-IDS2017 <-> LycoS-IDS2017)
# Soru: Bir veri setinde eğitilen model, diğer veri setinde de çalışıyor mu?
# - A: CIC'te eğit -> LycoS'ta test
# - B: LycoS'ta eğit -> CIC'te test
# - Referans: aynı ortak özelliklerle, her veri setinin kendi içinde %70/%30
# Hücreleri yukarıdan aşağıya sırayla çalıştır. Toplam süre: 15-30 dk olabilir.

# %%
import glob
from pathlib import Path
import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parent.parent
CIC_DIR = KOK / "data" / "cicids2017" / "MachineLearningCVE"
LYC_DIR = KOK / "data" / "lycos2017" / "lycos-ids2017" / "lycos-ids2017"
SONUC = KOK / "results"
SONUC.mkdir(exist_ok=True)

# CIC özellik adı -> LycoS özellik adı (anlamca aynı olduğundan emin olunanlar)
OZ = {
    "Destination Port": "dst_port", "Flow Duration": "flow_duration",
    "Total Fwd Packets": "fwd_pkt_cnt", "Total Backward Packets": "bwd_pkt_cnt",
    "Total Length of Fwd Packets": "fwd_pkt_len_tot", "Total Length of Bwd Packets": "bwd_pkt_len_tot",
    "Fwd Packet Length Max": "fwd_pkt_len_max", "Fwd Packet Length Min": "fwd_pkt_len_min",
    "Fwd Packet Length Mean": "fwd_pkt_len_mean", "Fwd Packet Length Std": "fwd_pkt_len_std",
    "Bwd Packet Length Max": "bwd_pkt_len_max", "Bwd Packet Length Min": "bwd_pkt_len_min",
    "Bwd Packet Length Mean": "bwd_pkt_len_mean", "Bwd Packet Length Std": "bwd_pkt_len_std",
    "Flow Bytes/s": "bytes_per_s", "Flow Packets/s": "pkt_per_s",
    "Flow IAT Mean": "iat_mean", "Flow IAT Std": "iat_std", "Flow IAT Max": "iat_max", "Flow IAT Min": "iat_min",
    "Fwd IAT Total": "fwd_iat_tot", "Fwd IAT Mean": "fwd_iat_mean", "Fwd IAT Std": "fwd_iat_std",
    "Fwd IAT Max": "fwd_iat_max", "Fwd IAT Min": "fwd_iat_min",
    "Bwd IAT Total": "bwd_iat_tot", "Bwd IAT Mean": "bwd_iat_mean", "Bwd IAT Std": "bwd_iat_std",
    "Bwd IAT Max": "bwd_iat_max", "Bwd IAT Min": "bwd_iat_min",
    "Fwd PSH Flags": "fwd_flag_psh", "Bwd PSH Flags": "bwd_flag_psh",
    "Fwd URG Flags": "fwd_flag_urg", "Bwd URG Flags": "bwd_flag_urg",
    "Fwd Header Length": "fwd_pkt_hdr_len_tot", "Bwd Header Length": "bwd_pkt_hdr_len_tot",
    "Fwd Packets/s": "fwd_pkt_per_s", "Bwd Packets/s": "bwd_pkt_per_s",
    "Min Packet Length": "pkt_len_min", "Max Packet Length": "pkt_len_max",
    "Packet Length Mean": "pkt_len_mean", "Packet Length Std": "pkt_len_std",
    "Packet Length Variance": "pkt_len_var",
    "FIN Flag Count": "flag_fin", "SYN Flag Count": "flag_SYN", "RST Flag Count": "flag_rst",
    "PSH Flag Count": "flag_psh", "ACK Flag Count": "flag_ack", "URG Flag Count": "flag_urg",
    "CWE Flag Count": "flag_cwr", "ECE Flag Count": "flag_ece",
    "Down/Up Ratio": "down_up_ratio",
    "Init_Win_bytes_forward": "fwd_tcp_init_win_bytes", "Init_Win_bytes_backward": "bwd_tcp_init_win_bytes",
    "Active Mean": "active_mean", "Active Std": "active_std", "Active Max": "active_max", "Active Min": "active_min",
    "Idle Mean": "idle_mean", "Idle Std": "idle_std", "Idle Max": "idle_max", "Idle Min": "idle_min",
    "Fwd Avg Bytes/Bulk": "fwd_bulk_bytes_mean", "Fwd Avg Packets/Bulk": "fwd_bulk_pkt_mean",
    "Fwd Avg Bulk Rate": "fwd_bulk_rate_mean", "Bwd Avg Bytes/Bulk": "bwd_bulk_bytes_mean",
    "Bwd Avg Packets/Bulk": "bwd_bulk_pkt_mean", "Bwd Avg Bulk Rate": "bwd_bulk_rate_mean",
}
ORTAK = list(OZ.values())
print("ortak özellik sayısı:", len(ORTAK))

# LycoS etiket adı -> CIC etiket adı (ortak 14 sınıf)
ETIKET = {
    "benign": "BENIGN", "bot": "Bot", "ddos": "DDoS",
    "dos_goldeneye": "DoS GoldenEye", "dos_hulk": "DoS Hulk",
    "dos_slowhttptest": "DoS Slowhttptest", "dos_slowloris": "DoS slowloris",
    "ftp_patator": "FTP-Patator", "heartbleed": "Heartbleed", "portscan": "PortScan",
    "ssh_patator": "SSH-Patator",
    "webattack_bruteforce": "Web Attack - Brute Force",
    "webattack_sql_injection": "Web Attack - Sql Injection",
    "webattack_xss": "Web Attack - XSS",
}

# %% [markdown]
# ## Yükleme: iki veri seti, aynı özellik adları, aynı etiket adları

# %%
def temizle(df, ad):
    for c in ORTAK:
        v = df[c].to_numpy()
        v[np.isinf(v)] = np.nan
    n0 = len(df)
    df = df.dropna().reset_index(drop=True)
    n1 = len(df)
    df = df[~pd.util.hash_pandas_object(df, index=False).duplicated()].reset_index(drop=True)
    print(f"{ad}: ham {n0}, NaN/inf sonrası {n1}, yinelenen sonrası {len(df)}")
    return df

def cic_yukle(f):
    d = pd.read_csv(f, low_memory=False)
    d.columns = d.columns.str.strip()
    d = d.loc[:, ~d.columns.duplicated()]           # CIC'te "Fwd Header Length" iki kez geçer
    y = d["Label"].str.replace("�", "-", regex=False)
    x = d[list(OZ.keys())].astype("float32")
    x.columns = ORTAK
    x["etiket"] = y.values
    return x

def lyc_yukle(f):
    d = pd.read_csv(f, usecols=ORTAK + ["label"], dtype={c: "float32" for c in ORTAK})
    d["etiket"] = d.pop("label").map(ETIKET)
    return d[ORTAK + ["etiket"]]          # sütun sırası CIC ile aynı olmalı (sklearn sırayı kontrol eder)

cic = pd.concat((cic_yukle(f) for f in sorted(glob.glob(str(CIC_DIR / "*.csv")))), ignore_index=True)
lyc = pd.concat((lyc_yukle(f) for f in sorted(glob.glob(str(LYC_DIR / "*.csv")))), ignore_index=True)

cic = cic[cic["etiket"].isin(ETIKET.values())].reset_index(drop=True)   # Infiltration atılır
lyc = lyc.dropna(subset=["etiket"]).reset_index(drop=True)

cic = temizle(cic, "CIC")
lyc = temizle(lyc, "LycoS")

# %% [markdown]
# ## Ölçek kontrolü
# Aynı özellik iki veri setinde aynı birimde mi? (Çok farklı medyanlar = birim/tanım farkı işareti.)

# %%
bakilacak = ["flow_duration", "iat_mean", "fwd_pkt_len_mean", "bytes_per_s", "fwd_tcp_init_win_bytes"]
ben = pd.DataFrame({
    "CIC_medyan": cic[cic.etiket == "BENIGN"][bakilacak].median(),
    "LycoS_medyan": lyc[lyc.etiket == "BENIGN"][bakilacak].median(),
})
print(ben)

# %% [markdown]
# ## Örnekleme (her sınıf en fazla 100.000)

# %%
SINIR = 100_000
def ornekle(df):
    p = [g.sample(min(len(g), SINIR), random_state=42) for _, g in df.groupby("etiket")]
    return pd.concat(p).sample(frac=1, random_state=42).reset_index(drop=True)

cic_o = ornekle(cic)
lyc_o = ornekle(lyc)
print("CIC örnek:", cic_o.shape, " LycoS örnek:", lyc_o.shape)

# %% [markdown]
# ## Deneyler

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score

def rf():
    return RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)

def degerlendir(ad, y_gercek, y_tahmin):
    makro = f1_score(y_gercek, y_tahmin, average="macro")
    acc = accuracy_score(y_gercek, y_tahmin)
    rapor = classification_report(y_gercek, y_tahmin, digits=4, zero_division=0)
    print(f"\n===== {ad} =====\naccuracy {acc:.4f}  makro F1 {makro:.4f}\n{rapor}")
    return {"deney": ad, "accuracy": round(acc, 4), "makro_F1": round(makro, 4)}, rapor

sonuclar, raporlar = [], {}
X_cic, y_cic = cic_o.drop(columns="etiket"), cic_o["etiket"]
X_lyc, y_lyc = lyc_o.drop(columns="etiket"), lyc_o["etiket"]

# %%
# Referans 1: CIC kendi içinde
Xa, Xb, ya, yb = train_test_split(X_cic, y_cic, test_size=0.3, random_state=42, stratify=y_cic)
m = rf().fit(Xa, ya)
r, rap = degerlendir("CIC -> CIC (ortak özellik)", yb, m.predict(Xb))
sonuclar.append(r); raporlar[r["deney"]] = rap

# %%
# Referans 2: LycoS kendi içinde
Xa, Xb, ya, yb = train_test_split(X_lyc, y_lyc, test_size=0.3, random_state=42, stratify=y_lyc)
m = rf().fit(Xa, ya)
r, rap = degerlendir("LycoS -> LycoS (ortak özellik)", yb, m.predict(Xb))
sonuclar.append(r); raporlar[r["deney"]] = rap

# %%
# Deney A: CIC'te eğit, LycoS'ta test
m = rf().fit(X_cic, y_cic)
r, rap = degerlendir("A: CIC -> LycoS", y_lyc, m.predict(X_lyc))
sonuclar.append(r); raporlar[r["deney"]] = rap

# %%
# Deney B: LycoS'ta eğit, CIC'te test
m = rf().fit(X_lyc, y_lyc)
r, rap = degerlendir("B: LycoS -> CIC", y_cic, m.predict(X_cic))
sonuclar.append(r); raporlar[r["deney"]] = rap

# %% [markdown]
# ## Özet tablo ve kayıt

# %%
ozet = pd.DataFrame(sonuclar)
print(ozet.to_string(index=False))
ozet.to_csv(SONUC / "capraz_deney_ozet.csv", index=False)
with open(SONUC / "capraz_deney_raporlar.txt", "w", encoding="utf-8") as f:
    for ad, metin in raporlar.items():
        f.write(f"===== {ad} =====\n{metin}\n")
print("kaydedildi:", SONUC)
