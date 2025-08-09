import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# --- Langkah 1: Memuat dan Membersihkan Data ---
try:
    # Memuat file data mobil
    df = pd.read_csv('6/car-sales-extended.csv')
    print("Berhasil memuat file 'car-sales-extended.csv'.")
except FileNotFoundError:
    print("Gagal memuat file. Pastikan 'car-sales-extended.csv' ada di direktori yang sama.")
    exit()

# Mengisi data yang kosong/hilang dengan cara sederhana
# Jika ada harga kosong, isi dengan rata-rata harga.
df['Price'].fillna(df['Price'].mean(), inplace=True)
# Jika ada data odometer kosong, isi dengan rata-rata odometer.
df['Odometer (KM)'].fillna(df['Odometer (KM)'].mean(), inplace=True)
# Mengisi data kategorikal yang kosong dengan nilai yang paling sering muncul.
for col in ['Make', 'Colour', 'Doors']:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("\nData sudah bersih dan siap dianalisis.")

# --- Langkah 2: Analisis dan Visualisasi Sederhana ---
# Kita ingin tahu: "Apakah semakin tinggi kilometer mobil, harganya semakin murah?"

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Odometer (KM)', y='Price', alpha=0.5)
plt.title('Hubungan Kilometer (Odometer) dengan Harga Mobil', fontsize=16)
plt.xlabel('Jarak Tempuh (KM)', fontsize=12)
plt.ylabel('Harga ($)', fontsize=12)
plt.grid(True)
plt.savefig('harga_vs_odometer_sederhana.png')

print("Visualisasi hubungan harga vs odometer telah disimpan sebagai 'harga_vs_odometer_sederhana.png'.")
print("Dari grafik, terlihat jelas bahwa semakin tinggi KM mobil, harganya cenderung semakin turun.")


# --- Langkah 3: Membuat Model Prediksi Sederhana ---
# Tujuan: Membuat model yang bisa menebak harga mobil berdasarkan fiturnya.
# Kita ubah data teks (Merek, Warna) menjadi angka agar bisa dihitung oleh model.
df_model = pd.get_dummies(df, columns=['Make', 'Colour'])

# Tentukan apa yang menjadi 'fitur' (X) dan apa yang menjadi 'target' (y)
# Fitur: Semua kolom kecuali harga
X = df_model.drop('Price', axis=1)
# Target: Kolom harga
y = df_model['Price']

# Bagi data menjadi data untuk melatih model dan data untuk menguji model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Membuat model prediksi (Regresi Linier)
model = LinearRegression()
# Melatih model dengan data
model.fit(X_train, y_train)

print("\nModel prediksi harga mobil berhasil dibuat!")

# Menguji seberapa bagus model kita
akurasi = model.score(X_test, y_test)
print(f"Tingkat akurasi model: {akurasi:.2%}")
print(
    "Artinya, model ini bisa menjelaskan sekitar " f"{akurasi:.0%}" " variasi harga mobil berdasarkan fiturnya.")
