import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os # Dosya yollarını bulmak için gerekli
from sklearn.preprocessing import (
    MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler,
    Normalizer, PowerTransformer, QuantileTransformer
)

# 1. VERİ YÜKLEME (GARANTİLİ YÖNTEM - ONEDRIVE UYUMLU)
print("Dosya yolu aranıyor...")

# Bu komut, şu an çalışan .py dosyasının tam klasör yolunu alır.
# Yani: C:\Users\ACER\OneDrive\Desktop\sayısal çözümlemeye giriş\
script_klasoru = os.path.dirname(os.path.abspath(__file__))

# Klasör yolu ile dosya adını birleştirir
dosya_yolu = os.path.join(script_klasoru, 'housing.csv')

print(f"Aranan Tam Yol: {dosya_yolu}")

if os.path.exists(dosya_yolu):
    df = pd.read_csv(dosya_yolu)
    print(f"BAŞARILI! Dosya bulundu ve yüklendi. ({len(df)} satır)")
else:
    # Dosya bulunamazsa KIRMIZI bir hata mesajı verir ve programı durdurur.
    print("\n" + "="*50)
    print("HATA: 'housing.csv' dosyası bulunamadı!")
    print(f"Lütfen 'housing.csv' dosyasını şu klasöre attığından emin ol:")
    print(f"{script_klasoru}")
    print("="*50 + "\n")
    exit() # Hata varsa aşağıya devam etmesini engeller, 'df not defined' hatasını önler.


# 2. VERİ HAZIRLIĞI

try:
    X = df[['median_income']].dropna().values
except KeyError:
    print("HATA: 'median_income' sütunu bulunamadı. Sütun adları şunlar:")
    print(df.columns.tolist())

# 3. DÖNÜŞÜM İŞLEMLERİ
transformations = {}

# Ölçeklendirme 
transformations["Orijinal Veri"] = X
transformations["Max Scaling"] = MaxAbsScaler().fit_transform(X)
transformations["Min-Max Scaling"] = MinMaxScaler().fit_transform(X)
transformations["Mean Normalization"] = (X - X.mean()) / (X.max() - X.min())
transformations["Z-Score (Standardization)"] = StandardScaler().fit_transform(X)
transformations["Robust Scaling"] = RobustScaler().fit_transform(X)

# ---  Vektör Normlaştırma ---
transformations["L1 Norm"] = Normalizer(norm='l1').fit_transform(X.T).T 
transformations["L2 Norm"] = Normalizer(norm='l2').fit_transform(X.T).T 
transformations["L∞ (Max) Norm"] = Normalizer(norm='max').fit_transform(X.T).T

# ---  Dağılım Dönüşümleri ---
transformations["Log Dönüşümü"] = np.log1p(X)
transformations["Box-Cox (Power)"] = PowerTransformer(method='box-cox').fit_transform(X)
transformations["Yeo-Johnson (Power)"] = PowerTransformer(method='yeo-johnson').fit_transform(X)

# ---  Quantile Dönüşümleri ---
transformations["Quantile (Uniform)"] = QuantileTransformer(output_distribution='uniform', random_state=42).fit_transform(X)
transformations["Quantile (Gaussian)"] = QuantileTransformer(output_distribution='normal', random_state=42).fit_transform(X)


# 4. GÖRSELLEŞTİRME

print("Grafikler oluşturuluyor...")

fig, axes = plt.subplots(5, 3, figsize=(18, 22))
axes = axes.flatten()
fig.suptitle(f"Veri Ön İşleme Teknikleri: California Housing (Medyan Gelir)", fontsize=20, y=1.01)

keys = list(transformations.keys())
plot_color = '#e67e22' 

for i, ax in enumerate(axes):
    if i < len(keys):
        name = keys[i]
        data_to_plot = transformations[name]
        
        sns.histplot(data_to_plot, kde=True, ax=ax, color=plot_color, edgecolor='black', bins=50)
        
        ax.set_title(name, fontsize=13, fontweight='bold', color='#333')
        ax.set_xlabel("Değer Aralığı", fontsize=9)
        ax.set_ylabel("Frekans", fontsize=9)
        
        mu = np.mean(data_to_plot)
        sigma = np.std(data_to_plot)
        text_str = f'$\mu={mu:.2f}$\n$\sigma={sigma:.2f}$'
        ax.text(0.95, 0.95, text_str, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.grid(True, alpha=0.4, linestyle='--')
    else:
        ax.axis('off')

plt.tight_layout()
plt.show()

print("İşlem tamamlandı.")
