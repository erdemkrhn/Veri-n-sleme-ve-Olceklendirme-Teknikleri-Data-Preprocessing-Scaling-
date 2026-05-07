#  Veri Ön İşleme ve Ölçeklendirme Teknikleri (Data Preprocessing Techniques)

Bu proje, Makine Öğrenmesi (Machine Learning) modellerinin performansını doğrudan etkileyen **veri ön işleme, ölçeklendirme (scaling) ve dağılım dönüşümü** tekniklerinin pratik bir incelemesidir. 

Çalışmada, popüler **California Housing** veri setindeki `median_income` (medyan gelir) özelliği kullanılarak, bir verinin farklı istatistiksel işlemlerden geçtikten sonra nasıl bir forma büründüğü görselleştirilmiştir.

##  Projenin Amacı
Makine öğrenmesi algoritmaları (özellikle KNN, SVM, Lineer Regresyon ve Sinir Ağları gibi mesafe/gradyan tabanlı modeller) verinin ölçeğine karşı çok hassastır. Ayrıca, çarpık (skewed) dağılımların normal dağılıma yaklaştırılması model başarısını artırır. Bu kod bloğu, veriyi algoritmaya sunmadan önce uygulanabilecek 13 farklı dönüşüm tekniğinin etkilerini tek bir ekranda karşılaştırmayı amaçlamaktadır.

## Kullanılan Teknolojiler ve Kütüphaneler
* **Python 3.x**
* **Pandas & NumPy:** Veri manipülasyonu ve matematiksel işlemler
* **Scikit-Learn (sklearn):** Veri ölçeklendirme ve dönüşüm işlemleri
* **Matplotlib & Seaborn:** İstatistiksel veri görselleştirme

## Uygulanan Makine Öğrenmesi Dönüşüm İşlemleri

Proje kapsamında veri seti üzerinde 4 ana kategoride dönüşümler uygulanmıştır:

1. **Ölçeklendirme (Scaling):** Veriyi belirli bir aralığa sıkıştırmak veya standart sapmasına göre ayarlamak.
   * `MaxAbsScaler`, `MinMaxScaler`, `StandardScaler (Z-Score)`, `RobustScaler`, Mean Normalization

2. **Vektör Normlaştırma (Normalization):** Veri örneklerini (satırları) birim normlara göre ayarlamak.
   * `L1 Norm`, `L2 Norm`, `L∞ (Max) Norm`

3. **Dağılım Dönüşümleri (Distribution Transformations):** Çarpık verileri normal dağılıma (Gaussian) yaklaştırmak.
   * `Log Transform (np.log1p)`, `Box-Cox Transform`, `Yeo-Johnson Transform`

4. **Kuantil Dönüşümleri (Quantile Transformations):** Veriyi istatistiksel sıralamasına göre dönüştürmek.
   * `Quantile (Uniform)`, `Quantile (Gaussian)`

