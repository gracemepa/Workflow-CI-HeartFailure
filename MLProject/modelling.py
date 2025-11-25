import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ==========================================
# KONFIGURASI MLFLOW (VERSI BERSIH UNTUK CI)
# ==========================================

# mlflow.set_tracking_uri("file:./mlruns")               
# mlflow.set_experiment("Heart_Disease_Prediction_Skilled") 

def load_data(data_dir="heart_preprocessing"):
    print("Memuat data dari folder:", data_dir)
    # Load X (Features)
    X_train = pd.read_csv(f"{data_dir}/X_train.csv")
    X_test = pd.read_csv(f"{data_dir}/X_test.csv")
    # Load y (Target)
    y_train = pd.read_csv(f"{data_dir}/y_train.csv").values.ravel()
    y_test = pd.read_csv(f"{data_dir}/y_test.csv").values.ravel()
    return X_train, X_test, y_train, y_test

def main():
    X_train, X_test, y_train, y_test = load_data()
    
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5]
    }

    print("Mulai Training di Cloud...")
    
    # Mulai Run (Tanpa nama run spesifik agar aman)
    with mlflow.start_run():
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        y_pred = best_model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='binary')
        rec = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        
        # Logging
        for param_name, param_value in best_params.items():
            mlflow.log_param(param_name, param_value)
            
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        
        mlflow.sklearn.log_model(best_model, "model")
        
        # Artifact Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig("training_confusion_matrix.png")
        mlflow.log_artifact("training_confusion_matrix.png")
        
        print(f"Selesai! Akurasi: {acc}")

if __name__ == "__main__":
    main()