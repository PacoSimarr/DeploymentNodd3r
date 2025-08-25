# train_model.py (VERSIÓN CORREGIDA)
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
import joblib

def train_and_save_model(use_gridsearch=False, data_path='data.csv', 
                        model_path='lr_best.pkl', scaler_path='scaler.pkl'):
    """
    Función para entrenar y guardar el modelo de predicción de fallos de máquina.
    """
    # Cargar datos
    df = pd.read_csv(data_path)
    
    # Eliminar duplicados
    df = df.drop_duplicates()
    
    # Separar características y target
    X = df.drop('fail', axis=1)
    y = df['fail']
    
    # Dividir datos PRIMERO (para evitar data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Escalar características DESPUÉS de dividir
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Inicializar best_params como None por defecto
    best_params = None
    
    if use_gridsearch:
        # 🔍 MODO GRIDSEARCH (búsqueda de hiperparámetros)
        print("Ejecutando GridSearchCV...")
        
        # Definir espacio de búsqueda de hiperparámetros
        param_grid = {
            'penalty': ['l1', 'l2'],
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'saga'],
            'class_weight': [None, 'balanced']
        }
        
        # GridSearch
        grid_search = GridSearchCV(
            LogisticRegression(random_state=42, max_iter=1000),
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        # Mejor modelo
        model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"Mejores parámetros encontrados: {best_params}")
        
    else:
        # 🚀 MODO RÁPIDO (parámetros pre-optimizados)
        print("Usando parámetros pre-optimizados...")
        model = LogisticRegression(
            C=1,
            class_weight=None, 
            penalty='l1',
            solver='liblinear',
            random_state=42,
            max_iter=1000
        )
        model.fit(X_train_scaled, y_train)
    
    # Evaluar modelo
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Guardar modelo Y scaler
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"Modelo guardado en: {model_path}")
    print(f"Scaler guardado en: {scaler_path}")
    print(f"Precisión: {accuracy:.4f}")
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'model_path': model_path,
        'scaler_path': scaler_path,
        'used_gridsearch': use_gridsearch,
        'best_params': best_params  # Puede ser None si no se usó GridSearch
    }

if __name__ == "__main__":
    # Para uso por línea de comandos (por defecto sin GridSearch)
    metrics = train_and_save_model(use_gridsearch=False)
    print(f"Accuracy: {metrics['accuracy']:.4f}")