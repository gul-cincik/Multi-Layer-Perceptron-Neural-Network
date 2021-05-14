# kullanılan kütüphaneler
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# scale etmek için
from sklearn.preprocessing import MinMaxScaler
# 2-fold etmek için
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
# ilk veriyi ekledik
wine = pd.read_csv("winequality-red (1).csv")

# ilk veri
print(wine)

df= wine.copy()
# hedef sütunu hariç diğer feature'ları 0 - 1 arasına scale ettik
scaler = MinMaxScaler((0, 1))
df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])

# feature'lar
X = df.iloc[:, :-1]
# hedef sütunu olmak üzere ayırdık
y = df.iloc[:, -1]

# Deep leerning API
from tensorflow import keras
# MLP için kullanılacak layers classı
from tensorflow.keras import layers

# burada 2 fold olarak ayarlıyoruz
kfold = KFold(n_splits=2)
# daha sonrasında modelin eğitim sonuçları burada yer alacak
histories = []

# split ederek feature'ların içerisindeki indexleri alıyoruz
for train_index, test_index in kfold.split(X):
    # eğitim için gerekli olaran training ve test kısımlarına bölüyoruz
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = y.iloc[train_index], y.iloc[test_index]

    # fully connected bir yapı ve art arda gelen layerler olması için keras içerisinden Sequential'ı çağırdık
    model_1 = keras.Sequential([layers.Dense(32, input_shape=[11], activation='relu'),
                                # 32 nöronlu, 11 feature kabul eden, aktivasyon fonk'u relu olan bir katman
                                layers.Dense(32, activation='relu'),
                                # 32 nöronlu,  aktivasyon fonk'u relu olan bir katman
                                layers.Dense(
                                    1)])  # regresyon problemi olduğu için burada sadece bir nörünlu dense layerı oluşturuyoruz

    model_1.compile(  # modeli eğitmeden önce derliyoruz
        optimizer="sgd",  # model parametrelei sgd(stochastic gradient descent) algoritmasıyla iyileşecek
        loss="mae",
        # yitim(model performans ölçüm) fonksiyonu olarak da mae(mean abosulute error kullanıyor çünkü regresyon problemlerinde bukullanılıyor)
    )
    history = model_1.fit(  # model eğitimi
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        batch_size=256,
        epochs=200  # 200 adımlık bir eğitim
    )
    histories.append(history)

model_1.summary()  # burada da farkli bir hücrede çalıştırarak yapısı gözlemlenebilir

# 1. fold
# iyi sayılabilir
pd.DataFrame(histories[0].history).plot()


# ikinci problem olan binary classification için gerekli veriyi yükledik
# veri su ile ilgili özellikleri barından ve hedef sütunu olarak da potability yani içilebilir(0,1) değerler barındırıyor
water = pd.read_csv("water_potability.csv")
# veri içerisindeki bazı feature'larda missing values oladuğundan dolayı olmayan feature'ları kullandık
features = ["Hardness", "Solids", "Chloramines", "Conductivity", "Organic_carbon", "Turbidity", "Potability"]

# kopyasını aldık ki işlemlerimiz ilk  veriyi etkilemesin 
df = water.loc[:, features].copy()

# feature'ları (0-1) arasında scale ettik
scaler_2 = MinMaxScaler((0, 1))
df.iloc[:, :-1] = scaler_2.fit_transform(df.iloc[:, :-1].copy())

# feature
X = df.iloc[:, :-1]
# ve target olarak ayırdık
y = df.iloc[:, -1]

histories = []
# önceki modelde uygulanan kfold işlemi bir daha uygulandı 
for train_index, test_index in kfold.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = y.iloc[train_index], y.iloc[test_index]

    # değişen yerler şunlar

    model_2 = keras.Sequential([layers.Dense(32, input_shape=[6], activation='relu'),
                                # input_shape değişti çünkü modele giren feature sayısı değişti
                                layers.Dense(32, activation='relu'),
                                layers.Dense(1,
                                             activation='sigmoid')])  # output layer değişti çünkü problem binary classification problemi
    model_2.compile(
        optimizer="sgd",
        loss='binary_crossentropy',
        # binary classification problemi olduğundan dolayı performans ölçme metirği de değişti
        metrics=['binary_accuracy'],  # classification olduğı için de accuracy değerlerini görebieceğiz böylelikle
    )
    history = model_2.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        batch_size=256,
        epochs=200
    )
    histories.append(history)
model_2.summary()

# 1. fold iyi değil
pd.DataFrame(histories[0].history).plot()

# 2. fold
# iyi değil
# çünkü val_accuracy değerleri classification problemi için düşük seviyelerde
pd.DataFrame(histories[1].history).plot()
# veriden kaynaklı olarak performansı düşük ve ek olarak
# oluşturulan bir model mimarisi farklı farklı verilere ve problemlere uygulanması uygun değildir
# her problem ve veri için farklı modeller oluşturulmalıdır