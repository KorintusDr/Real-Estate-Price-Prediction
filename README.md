
# Deteksi Bahasa dengan Machine Learning

Proyek ini bertujuan untuk mendeteksi bahasa dari sebuah teks menggunakan model machine learning berbasis algoritma **Multinomial Naïve Bayes**. Model dilatih menggunakan dataset berisi 1000 kalimat dari 22 bahasa yang berbeda.

## Deskripsi Dataset

Dataset yang digunakan dapat diakses dari:  
[https://raw.githubusercontent.com/amankharwal/Website-data/master/dataset.csv](https://raw.githubusercontent.com/amankharwal/Website-data/master/dataset.csv)

Dataset ini:
- Mengandung 22 bahasa berbeda.
- Setiap bahasa memiliki 1000 kalimat.
- Tidak memiliki nilai null (kosong), sehingga siap digunakan.

## Library yang Digunakan

- `pandas`
- `numpy`
- `scikit-learn`

## Langkah-Langkah Proyek

1. **Impor Dataset**
```python
data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/dataset.csv")
```

2. **Cek Nilai Null**
```python
data.isnull().sum()
```

3. **Distribusi Bahasa**
```python
data["language"].value_counts()
```

4. **Pemrosesan Data**
```python
x = np.array(data["Text"])
y = np.array(data["language"])

cv = CountVectorizer()
X = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
```

5. **Pelatihan Model**
```python
model = MultinomialNB()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
```

6. **Prediksi Bahasa**
```python
pengguna = input("Masukan Teks")
data = cv.transform([pengguna]).toarray()
output = model.predict(data)
print(output)
```

## Catatan

Model hanya bisa mendeteksi bahasa yang tersedia pada dataset pelatihan. Untuk menambahkan deteksi bahasa lain, dataset perlu diperluas.

---

© 2025 - Proyek Deteksi Bahasa NLP dengan Scikit-Learn
