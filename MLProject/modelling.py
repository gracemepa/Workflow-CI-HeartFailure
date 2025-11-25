import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ==========================================
# KONFIGURASI MLFLOW
# ==========================================

# MLflow tracking URI: gunakan environment variable jika ada, atau default ke local tracking
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if mlflow_tracking_uri:
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    print(f"MLflow tracking URI set dari environment variable: {mlflow_tracking_uri}")
else:
    print("MLflow tracking URI tidak ditetapkan. Menggunakan default local tracking (./mlruns)")

mlflow.set_experiment("Heart_Disease_Prediction_Skilled")

def load_data(data_dir="heart_preprocessing"):
    """
    Memuat data hasil preprocessing.
    Fungsi ini mencoba beberapa lokasi / nama file fallback agar script tetap berjalan
    jika dijalankan dari folder yang berbeda. Mengembalikan X_train, X_test, y_train, y_test.
    """
    base = os.path.dirname(__file__)
    print("Mencari data preprocessing. Base dir script:", base)

    # Kandidat direktori relatif terhadap lokasi script atau working dir
    candidates = [
        data_dir,
        os.path.join(base, data_dir),
        os.path.join(base, '..', 'preprocessing', data_dir),
        os.path.join(base, '..', 'preprocessing'),
        os.path.join(base, '..')
    ]

    # Nama file fallback untuk X (beberapa notebook menyimpan dengan nama X_train_processed.csv)
    x_candidates = ['X_train.csv', 'X_train_processed.csv']
    xt_candidates = ['X_test.csv', 'X_test_processed.csv']
    y_name = 'y_train.csv'
    yt_name = 'y_test.csv'

    found = None
    for c in candidates:
        if not c:
            continue
        # cek keberadaan file untuk X_train, X_test, y_train, y_test
        xt_path = None
        for xn in x_candidates:
            p = os.path.join(c, xn)
            if os.path.exists(p):
                xt_path = p
                break

        xtest_path = None
        for xn in xt_candidates:
            p = os.path.join(c, xn)
            if os.path.exists(p):
                xtest_path = p
                break

        y_path = os.path.join(c, y_name)
        yt_path = os.path.join(c, yt_name)

        if xt_path and xtest_path and os.path.exists(y_path) and os.path.exists(yt_path):
            found = (xt_path, xtest_path, y_path, yt_path)
            print("Menemukan data di:", c)
            break

    if not found:
        # Beri pesan yang lebih informatif daripada FileNotFoundError bawaan
        msg = (
            "Tidak menemukan file preprocessing yang diperlukan.\n"
            "Dicari nama: X_train(.csv|_processed.csv), X_test(.csv|_processed.csv), y_train.csv, y_test.csv\n"
            "Direktori yang dicoba: " + ", ".join(candidates) + "\n"
            "Pastikan Anda sudah menjalankan preprocessing dan file disimpan di salah satu lokasi tersebut."
        )
        raise FileNotFoundError(msg)

    X_train = pd.read_csv(found[0])
    X_test = pd.read_csv(found[1])
    y_train = pd.read_csv(found[2]).values.ravel()
    y_test = pd.read_csv(found[3]).values.ravel()

    print(f"Data Loaded. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def main():
    # 1. Load Data
    X_train, X_test, y_train, y_test = load_data()

    # 2. Setup Hyperparameter Tuning (Syarat Skilled)
    # Kita gunakan Random Forest sebagai contoh
    rf = RandomForestClassifier(random_state=42)
    
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }

    print("Mulai Hyperparameter Tuning...")
    
    # 3. Mulai Run MLflow
    # Penting: Jangan gunakan mlflow.autolog() untuk level Skilled/Advance [cite: 77]
    with mlflow.start_run(run_name="Skilled_Manual_Tuning"):
        
        # Proses Grid Search
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Ambil model terbaik
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        print(f"Best Params: {best_params}")

        # 4. Evaluasi Model
        y_pred = best_model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='binary') # Sesuaikan average jika multiclass
        rec = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        
        print(f"Accuracy: {acc:.4f}")
        
        # 5. Manual Logging (Syarat Skilled) [cite: 77]
        # Log parameter terbaik
        for param_name, param_value in best_params.items():
            mlflow.log_param(param_name, param_value)
            
        # Log metrik evaluasi
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        
        # Log model itu sendiri
        mlflow.sklearn.log_model(best_model, "model")
        
        # 6. Log Artefak Tambahan (Confusion Matrix) [cite: 228-237]
        # Membuat plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # Simpan gambar sementara lalu log ke MLflow
        cm_filename = "training_confusion_matrix.png"
        plt.savefig(cm_filename)
        mlflow.log_artifact(cm_filename)
        
        print("Logging selesai. Cek MLflow UI.")
    
    # Pastikan run ditutup dengan baik untuk menghindari conflict di CI
    mlflow.end_run()

if __name__ == "__main__":
    main()