# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, RocCurveDisplay
)
import train_model  # Importamos nuestro módulo de entrenamiento

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Fallos de Máquina",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("⚙️ Sistema de Predicción de Fallos de Máquina")
st.markdown("---")

# ============================================================
# MENÚ PRINCIPAL MEJORADO EN SIDEBAR
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/machine-learning.png", width=80)
    st.title("🧭 Menú de Navegación")
    st.markdown("---")
    
    # Menú principal con iconos
    menu_opcion = st.radio(
        "**Secciones principales:**",
        [
            "🏠 Inicio", 
            "📊 Análisis de Datos", 
            "🤖 Entrenar Modelo", 
            "🔮 Predecir Fallos",
            "📈 Rendimiento del Modelo",
            "⚙️ Configuración"
        ],
        index=0
    )
    
    st.markdown("---")
    st.subheader("🚀 Acciones Rápidas")
    
    # Botones de acción rápida
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Entrenar", help="Entrenar modelo rápido"):
            menu_opcion = "🤖 Entrenar Modelo"
    with col2:
        if st.button("🔍 Predecir", help="Ir a predicción"):
            menu_opcion = "🔮 Predecir Fallos"
    
    st.markdown("---")
    st.caption("v1.0 | Sistema de Predicción Inteligente")

# ============================================================
# SECCIÓN: INICIO
# ============================================================
if menu_opcion == "🏠 Inicio":
    st.header("🏠 Bienvenido al Sistema de Predicción de Fallos")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### ¿Qué puedes hacer con esta aplicación?
        
        - **📊 Análisis de Datos**: Explora y visualiza los datos de sensores
        - **🤖 Entrenar Modelo**: Entrena modelos de machine learning
        - **🔮 Predecir Fallos**: Realiza predicciones en tiempo real
        - **📈 Rendimiento**: Evalúa el desempeño del modelo
        - **⚙️ Configuración**: Ajusta parámetros del sistema
        
        ### 📋 Dataset de Machine Failure Prediction
        Datos de sensores para predecir fallos en máquinas con:
        - 9 características de sensores
        - Variable objetivo binaria (fallo/no fallo)
        - Datos limpios y preprocesados
        """)
    
    with col2:
        st.info("""
        **📊 Estadísticas rápidas:**
        - 943 registros
        - 10 características
        - 2 clases (0: Normal, 1: Fallo)
        - 0 valores nulos
        """)
        
        # Mini visualización
        try:
            df = pd.read_csv("data.csv")
            counts = df['fail'].value_counts()
            st.metric("Registros con fallo", f"{counts.get(1, 0)}")
            st.metric("Registros normales", f"{counts.get(0, 0)}")
        except:
            st.warning("Dataset no disponible")

# ============================================================
# SECCIÓN: ANÁLISIS DE DATOS (MANTIENE TU CÓDIGO ORIGINAL)
# ============================================================
elif menu_opcion == "📊 Análisis de Datos":
    st.header("📊 Análisis Exploratorio de Datos")
    
    # Cargar datos
    @st.cache_data
    def load_data():
        return pd.read_csv("data.csv")
    
    df = load_data()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Primeras filas del dataset")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader("📈 Estadísticas descriptivas")
        st.dataframe(df.describe(), use_container_width=True)
    
    with col2:
        st.subheader("📊 Distribución de Fallos")
        counts = df['fail'].value_counts()
        st.bar_chart(counts)
        
        st.subheader("ℹ️ Información del dataset")
        st.write(f"**📏 Filas:** {df.shape[0]}")
        st.write(f"**📊 Columnas:** {df.shape[1]}")
        st.write(f"**🔍 Valores nulos:** {df.isnull().sum().sum()}")
        st.write(f"**⚖️ Balance de clases:**")
        st.write(f"  - Normal (0): {counts.get(0, 0)} registros")
        st.write(f"  - Fallo (1): {counts.get(1, 0)} registros")

# ============================================================
# SECCIÓN: ENTRENAR MODELO (MANTIENE TU CÓDIGO ORIGINAL)
# ============================================================
elif menu_opcion == "🤖 Entrenar Modelo":
    st.header("🤖 Entrenamiento del Modelo")
    
    with st.expander("⚙️ Opciones avanzadas de entrenamiento", expanded=False):
        use_gridsearch = st.checkbox(
            "Ejecutar GridSearchCV (MUY LENTO - solo para experimentación)",
            value=False,
            help="Buscará los mejores parámetros desde cero. Puede tomar varios minutos. Solo para uso experimental."
        )
    
    if use_gridsearch:
        st.warning("⚠️ **MODO EXPERIMENTAL ACTIVADO**: El entrenamiento será mucho más lento pero puede encontrar parámetros más óptimos.")
    else:
        st.info("""
        Esta acción entrenará un modelo de Logistic Regression con los mejores parámetros encontrados:
        - **Penalty**: L1
        - **C**: 1  
        - **Solver**: liblinear
        - **Class Weight**: None
        """)
    
    if st.button("🚀 Iniciar Entrenamiento", type="primary", use_container_width=True):
        with st.spinner("🧠 Entrenando modelo... Esto puede tomar unos segundos"):
            try:
                metrics = train_model.train_and_save_model(use_gridsearch=use_gridsearch)
                
                st.success("✅ ¡Modelo entrenado exitosamente!")
                
                # Mostrar métricas en columnas
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("🎯 Precisión", f"{metrics['accuracy']:.4f}")
                
                with col2:
                    st.metric("💾 Modelo", "lr_best.pkl")
                
                with col3:
                    st.metric("📐 Scaler", "scaler.pkl")
                
                # Información del modo usado
                if metrics.get('used_gridsearch', False):
                    st.info("🔍 **Modo usado**: GridSearchCV")
                    if metrics.get('best_params'):
                        st.write("**Mejores parámetros encontrados:**", metrics['best_params'])
                else:
                    st.info("🚀 **Modo usado**: Parámetros pre-optimizados")
                    st.write("**Parámetros usados:**", metrics.get('best_params', {}))
                
                # Reporte de clasificación
                st.subheader("📊 Reporte de Clasificación")
                report_df = pd.DataFrame(metrics['classification_report']).transpose()
                st.dataframe(report_df.style.highlight_max(axis=0), use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error durante el entrenamiento: {str(e)}")

# ============================================================
# SECCIÓN: PREDECIR FALLOS (MANTIENE TU CÓDIGO ORIGINAL)
# ============================================================
elif menu_opcion == "🔮 Predecir Fallos":
    st.header("🔮 Predicción de Fallos en Tiempo Real")
    
    try:
        model = joblib.load('lr_best.pkl')
        scaler = joblib.load('scaler.pkl')
        st.success("✅ Modelo y scaler cargados exitosamente!")
        
        st.subheader("🎛️ Ingresa los valores de los sensores:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            footfall = st.number_input("Footfall", min_value=0.0, value=25.0, help="Número de personas u objetos pasando")
            tempMode = st.number_input("Temp Mode", min_value=0.0, value=1.0, help="Modo de temperatura")
            AQ = st.number_input("AQ", min_value=0.0, value=50.0, help="Calidad del aire")
        
        with col2:
            USS = st.number_input("USS", min_value=0.0, value=100.0, help="Sensor ultrasónico")
            CS = st.number_input("CS", min_value=0.0, value=2.5, help="Sensor de corriente")
            VOC = st.number_input("VOC", min_value=0.0, value=2500.0, help="Compuestos orgánicos volátiles")
        
        with col3:
            RP = st.number_input("RP", min_value=0.0, value=35.0, help="Posición rotacional")
            IP = st.number_input("IP", min_value=0.0, value=50.0, help="Presión de entrada")
            Temperature = st.number_input("Temperature", min_value=0.0, value=60.0, help="Temperatura operativa")
        
        if st.button("🔍 Realizar Predicción", type="primary", use_container_width=True):
            input_data = np.array([[footfall, tempMode, AQ, USS, CS, VOC, RP, IP, Temperature]])
            input_scaled = scaler.transform(input_data)
            
            prediction = model.predict(input_scaled)
            probability = model.predict_proba(input_scaled)
            
            st.markdown("---")
            st.subheader("📋 Resultado de la Predicción")
            
            if prediction[0] == 1:
                st.error("🚨 **PREDICCIÓN: FALLO INMINENTE**")
                st.write(f"📊 Probabilidad de fallo: {probability[0][1]:.2%}")
            else:
                st.success("✅ **PREDICCIÓN: MÁQUINA EN ESTADO NORMAL**")
                st.write(f"📊 Probabilidad de fallo: {probability[0][1]:.2%}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("✅ Probabilidad de NO fallo", f"{probability[0][0]:.2%}")
            with col2:
                st.metric("❌ Probabilidad de fallo", f"{probability[0][1]:.2%}")
                
    except FileNotFoundError:
        st.warning("⚠️ Modelo no encontrado. Entrena el modelo primero.")

# ============================================================
# SECCIÓN: RENDIMIENTO (CORREGIDA Y COMPLETA)
# ============================================================
elif menu_opcion == "📈 Rendimiento del Modelo":
    st.header("📈 Rendimiento y Métricas del Modelo")
    
    try:
        # Cargar modelo, scaler y datos
        model = joblib.load('lr_best.pkl')
        scaler = joblib.load('scaler.pkl')
        df = pd.read_csv("data.csv")
        
        # Preparar datos para evaluación
        X = df.drop('fail', axis=1)
        y = df['fail']  # ✅ DEFINIMOS y AQUÍ
        X_scaled = scaler.transform(X)
        
        # Realizar predicciones ✅ DEFINIMOS y_pred AQUÍ
        y_pred = model.predict(X_scaled)
        y_pred_proba = model.predict_proba(X_scaled)[:, 1]
        
        st.success("✅ Modelo y datos cargados exitosamente!")
        
        # Pestañas para organizar las métricas
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Métricas Básicas", "📋 Reporte Completo", "🎯 Matriz de Confusión", "📈 Curva ROC"])
        
        with tab1:
            st.subheader("📊 Métricas Básicas de Rendimiento")
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            # ✅ AHORA y e y_pred ESTÁN DEFINIDAS
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred)
            recall = recall_score(y, y_pred)
            f1 = f1_score(y, y_pred)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Precisión (Accuracy)", f"{accuracy:.4f}")
            with col2:
                st.metric("Precisión (Precision)", f"{precision:.4f}")
            with col3:
                st.metric("Sensibilidad (Recall)", f"{recall:.4f}")
            with col4:
                st.metric("Puntuación F1", f"{f1:.4f}")
            
            # Gráfico de barras de métricas
            metrics_data = {
                'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                'Valor': [accuracy, precision, recall, f1]
            }
            metrics_df = pd.DataFrame(metrics_data)
            st.bar_chart(metrics_df.set_index('Métrica'))
        
        with tab2:
            st.subheader("📋 Reporte de Clasificación Completo")
            
            from sklearn.metrics import classification_report
            
            # ✅ AHORA y e y_pred ESTÁN DEFINIDAS
            report = classification_report(y, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            
            st.dataframe(
                report_df.style
                .highlight_max(axis=0, color='#90EE90')
                .format('{:.4f}'),
                use_container_width=True
            )
            
            # Descargar reporte
            csv = report_df.to_csv(index=True)
            st.download_button(
                label="📥 Descargar Reporte CSV",
                data=csv,
                file_name="reporte_clasificacion.csv",
                mime="text/csv"
            )
        
        with tab3:
            st.subheader("🎯 Matriz de Confusión")
            
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
            import matplotlib.pyplot as plt
            
            # ✅ AHORA y e y_pred ESTÁN DEFINIDAS
            cm = confusion_matrix(y, y_pred)
            
            fig, ax = plt.subplots(figsize=(6, 5))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Fallo'])
            disp.plot(cmap='Blues', ax=ax, values_format='d')
            plt.title('Matriz de Confusión')
            st.pyplot(fig)
            
            # Análisis de la matriz
            st.info("**Análisis de la matriz:**")
            st.write(f"- **Verdaderos Negativos (TN):** {cm[0, 0]}")
            st.write(f"- **Falsos Positivos (FP):** {cm[0, 1]}")
            st.write(f"- **Falsos Negativos (FN):** {cm[1, 0]}")
            st.write(f"- **Verdaderos Positivos (TP):** {cm[1, 1]}")
        
        with tab4:
            st.subheader("📈 Curva ROC y AUC")
            
            from sklearn.metrics import roc_curve, auc, RocCurveDisplay
            import matplotlib.pyplot as plt
            
            # ✅ AHORA y e y_pred_proba ESTÁN DEFINIDAS
            fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            RocCurveDisplay.from_predictions(
                y, y_pred_proba,
                name=f"ROC curve (AUC = {roc_auc:.4f})",
                ax=ax
            )
            plt.plot([0, 1], [0, 1], 'k--', label='Clasificador aleatorio')
            plt.xlabel('Tasa de Falsos Positivos')
            plt.ylabel('Tasa de Verdaderos Positivos')
            plt.title('Curva ROC')
            plt.legend(loc='lower right')
            st.pyplot(fig)
            
            st.metric("Área bajo la curva (AUC)", f"{roc_auc:.4f}")
            
            # Interpretación del AUC
            if roc_auc > 0.9:
                st.success("✅ Excelente poder de discriminación (AUC > 0.9)")
            elif roc_auc > 0.8:
                st.warning("⚠️ Buen poder de discriminación (AUC > 0.8)")
            else:
                st.error("❌ Poder de discriminación limitado (AUC < 0.8)")
        
        # Sección adicional de análisis
        st.markdown("---")
        st.subheader("📌 Análisis Adicional")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribución de probabilidades
            st.write("**Distribución de probabilidades predichas:**")
            prob_df = pd.DataFrame({
                'Probabilidad': y_pred_proba,
                'Clase Real': y.map({0: 'Normal', 1: 'Fallo'})
            })
            st.bar_chart(prob_df.groupby('Clase Real')['Probabilidad'].mean())
        
        with col2:
            # Métricas por clase
            st.write("**Resumen por clase:**")
            class_summary = pd.DataFrame({
                'Clase': ['Normal (0)', 'Fallo (1)'],
                'Registros': [len(y[y == 0]), len(y[y == 1])],
                'Precisión': [report['0']['precision'], report['1']['precision']],
                'Recall': [report['0']['recall'], report['1']['recall']]
            })
            st.dataframe(class_summary, use_container_width=True)
                
    except FileNotFoundError:
        st.error("""
        ❌ No se encontraron los archivos necesarios. Por favor:
        1. Ve a la sección **🤖 Entrenar Modelo**
        2. Entrena el modelo primero
        3. Vuelve a esta sección
        """)
    except Exception as e:
        st.error(f"❌ Error al cargar los datos: {str(e)}")

# ============================================================
# SECCIÓN: CONFIGURACIÓN (NUEVA)
# ============================================================
elif menu_opcion == "⚙️ Configuración":
    st.header("⚙️ Configuración del Sistema")
    
    st.subheader("🔧 Ajustes de la Aplicación")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.selectbox("Tema de la aplicación", ["Claro", "Oscuro"])
        st.slider("Tamaño de visualización", 1, 100, 80)
    
    with col2:
        st.number_input("Límite de registros", 100, 10000, 1000)
        st.checkbox("Modo de alto contraste")
    
    st.subheader("📁 Gestión de Archivos")
    if st.button("🗑️ Limpiar modelos antiguos"):
        st.info("Función de limpieza en desarrollo...")
    
    st.subheader("ℹ️ Información del Sistema")
    st.write(f"**Versión de Streamlit:** {st.__version__}")
    st.write("**Estado:** 🟢 Conectado")

# Footer
st.markdown("---")
st.caption("© 2024 Sistema de predicción de fallos de máquina | Desarrollado con Streamlit y Machine Learning")