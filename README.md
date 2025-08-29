
## 🗂️ Estructura del Proyecto

```bash
PrimerDeployment/
├── Código fuente
│   ├── app.py             # 🐍 Aplicación Streamlit principal
│   ├── train_model.py     # 🤖 Entrenamiento del modelo ML
│   └── utils.py           # 🛠️ Funciones utilitarias
├── # Datos
│   └── data.csv          # 📊 Dataset de entrenamiento
├── # Modelos guardados
│   ├── lr_best.pkl      # 🎯 Modelo Logistic Regression
│   └── scaler.pkl       # ⚖️ Scaler para normalización
├── requirements.txt      # 📋 Dependencias de Python
├── .gitignore           # 👁️ Archivos ignorados por Git
└── README.md            # 📖 Documentación del proyecto


# 🚀 Sistema de Predicción de Fallos de Máquina

Aplicación Streamlit para predecir fallos en máquinas usando Machine Learning.

## ✨ Características

- 📊 Análisis exploratorio de datos
- 🤖 Entrenamiento de modelos ML
- 🔮 Predicciones en tiempo real
- 📝 Explicaciones con SHAP
- 📈 Métricas de rendimiento



# Instalar dependencias
pip install -r requirements.txt

# Ejecutar aplicación
streamlit run app.py
