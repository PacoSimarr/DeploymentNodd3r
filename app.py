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
# SECCIÓN: ANÁLISIS DE DATOS
# ============================================================
elif menu_opcion == "📊 Análisis de Datos":
    st.header("📊 Análisis Exploratorio de Datos")
    
    # Cargar datos
    @st.cache_data
    def load_data():
        return pd.read_csv("data.csv")
    
    try:
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
            fig, ax = plt.subplots()
            ax.bar(['Normal (0)', 'Fallo (1)'], counts.values, color=['green', 'red'])
            ax.set_ylabel('Cantidad de Registros')
            ax.set_title('Distribución de Clases')
            st.pyplot(fig)
            
            st.subheader("ℹ️ Información del dataset")
            st.write(f"**📏 Filas:** {df.shape[0]}")
            st.write(f"**📊 Columnas:** {df.shape[1]}")
            st.write(f"**🔍 Valores nulos:** {df.isnull().sum().sum()}")
            st.write(f"**⚖️ Balance de clases:**")
            st.write(f"  - Normal (0): {counts.get(0, 0)} registros")
            st.write(f"  - Fallo (1): {counts.get(1, 0)} registros")
    
    except FileNotFoundError:
        st.error("❌ No se encontró el archivo data.csv")
    except Exception as e:
        st.error(f"❌ Error al cargar los datos: {str(e)}")

# ============================================================
# SECCIÓN: ENTRENAR MODELO
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
# SECCIÓN: PREDECIR FALLOS - CON SHAP INTEGRADO (ACTUALIZADO)
# ============================================================
elif menu_opcion == "🔮 Predecir Fallos":
    st.header("🔮 Predicción de Fallos en Tiempo Real")
    
    # Crear pestañas separadas
    tab_pred, tab_analysis, tab_explain = st.tabs(["🎯 Realizar Predicción", "📊 Análisis Dataset", "📝 Explicación del Modelo (SHAP)"])
    
    # PESTAÑA 1: PREDICCIÓN
    with tab_pred:
        st.subheader("🎯 Ingresa los valores del sensor para predecir")
        
        try:
            model = joblib.load('lr_best.pkl')
            scaler = joblib.load('scaler.pkl')
            df = pd.read_csv("data.csv")  # Cargar datos para el explainer
            
            st.success("✅ Modelo, scaler y datos cargados exitosamente!")
            
            # Formulario de entrada MEJORADO
            st.markdown("### 🎛️ Valores de los Sensores")
            
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
            
            # Botón de predicción
            if st.button("🔍 Realizar Predicción", type="primary", use_container_width=True):
                input_data = np.array([[footfall, tempMode, AQ, USS, CS, VOC, RP, IP, Temperature]])
                input_scaled = scaler.transform(input_data)
                
                prediction = model.predict(input_scaled)
                probability = model.predict_proba(input_scaled)
                
                # RESULTADO CLARO Y VISIBLE
                st.markdown("---")
                st.success("### 📋 Resultado de la Predicción")
                
                if prediction[0] == 1:
                    st.error("""
                    🚨 **PREDICCIÓN: FALLO INMINENTE**
                    
                    **Recomendaciones:**
                    - Detener máquina inmediatamente
                    - Revisar sensores críticos
                    - Contactar con mantenimiento
                    """)
                    st.write(f"📊 **Probabilidad de fallo:** {probability[0][1]:.2%}")
                else:
                    st.success("""
                    ✅ **PREDICCIÓN: MÁQUINA EN ESTADO NORMAL**
                    
                    **Recomendaciones:**
                    - Continuar operación normal
                    - Monitoreo rutinario
                    - Mantenimiento preventivo programado
                    """)
                    st.write(f"📊 **Probabilidad de fallo:** {probability[0][1]:.2%}")
                
                # Métricas de probabilidad
                col_met1, col_met2 = st.columns(2)
                with col_met1:
                    st.metric("✅ Probabilidad de NO fallo", f"{probability[0][0]:.2%}")
                with col_met2:
                    st.metric("❌ Probabilidad de fallo", f"{probability[0][1]:.2%}")
                
                # Visualización de probabilidades
                st.markdown("---")
                st.subheader("📊 Visualización de Probabilidades")
                
                prob_data = {
                    'Estado': ['NO FALLO', 'FALLO'],
                    'Probabilidad': [probability[0][0], probability[0][1]]
                }
                prob_df = pd.DataFrame(prob_data)
                
                fig, ax = plt.subplots(figsize=(10, 4))
                bars = ax.barh(prob_df['Estado'], prob_df['Probabilidad'], 
                              color=['green', 'red'], alpha=0.7)
                
                # Añadir valores en las barras
                for i, (value, state) in enumerate(zip(prob_df['Probabilidad'], prob_df['Estado'])):
                    ax.text(value + 0.01, i, f'{value:.2%}', va='center', fontweight='bold')
                
                ax.set_xlim(0, 1)
                ax.set_xlabel('Probabilidad')
                ax.set_title('Distribución de Probabilidades de Predicción')
                st.pyplot(fig)
                
                # Guardar los datos de entrada para usar en la pestaña de explicación
                st.session_state['input_data'] = input_data
                st.session_state['input_scaled'] = input_scaled
                st.session_state['prediction'] = prediction
                st.session_state['probability'] = probability
                    
        except FileNotFoundError:
            st.warning("⚠️ Modelo no encontrado. Por favor, entrena el modelo primero en la sección '🤖 Entrenar Modelo'.")
        except Exception as e:
            st.error(f"❌ Error al realizar la predicción: {str(e)}")
    
    # PESTAÑA 2: ANÁLISIS
    with tab_analysis:
        st.subheader("📊 Análisis del Dataset de Entrenamiento")
        
        try:
            df = pd.read_csv("data.csv")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("📋 Primeras filas del dataset:")
                st.dataframe(df.head(5), use_container_width=True)
                
                st.write("📈 Distribución de clases:")
                counts = df['fail'].value_counts()
                st.write(f"- Normal (0): {counts.get(0, 0)} registros")
                st.write(f"- Fallo (1): {counts.get(1, 0)} registros")
            
            with col2:
                st.write("📊 Estadísticas descriptivas:")
                st.dataframe(df.describe(), use_container_width=True)
                
                st.write("ℹ️ Información general:")
                st.write(f"- Filas: {df.shape[0]}")
                st.write(f"- Columnas: {df.shape[1]}")
                st.write(f"- Valores nulos: {df.isnull().sum().sum()}")
        
        except FileNotFoundError:
            st.warning("📝 Dataset no disponible para análisis")
        except Exception as e:
            st.error(f"❌ Error en el análisis: {str(e)}")
            
    # PESTAÑA 3: EXPLICACIÓN CON SHAP (CÓDIGO CORREGIDO)
    with tab_explain:
        st.subheader("📝 Explicación de la Predicción con SHAP")
        
        # DEBUG: Verificación mejorada de SHAP
        try:
            import shap
            SHAP_AVAILABLE = True
        except ImportError as e:
            SHAP_AVAILABLE = False
            st.error(f"❌ Error importando SHAP: {e}")
        
        if 'input_data' not in st.session_state:
            st.info("ℹ️ Realiza una predicción primero para ver la explicación.")
        else:
            try:
                if not SHAP_AVAILABLE:
                    st.warning("🔧 SHAP no disponible temporalmente")
                    st.info("""
                    **Para habilitar explicaciones completas:**
                    - Reinicia la aplicación
                    - Verifica que SHAP esté instalado
                    """)
                    
                    # Mostrar los valores ingresados aunque no haya SHAP
                    st.subheader("📋 Valores ingresados para predicción")
                    feature_names = ['Footfall', 'Temp Mode', 'AQ', 'USS', 'CS', 'VOC', 'RP', 'IP', 'Temperature']
                    input_df = pd.DataFrame(st.session_state['input_data'], columns=feature_names)
                    st.dataframe(input_df, use_container_width=True)
                    
                else:
                    # Cargar modelo y datos
                    model = joblib.load('lr_best.pkl')
                    df = pd.read_csv("data.csv")
                    X = df.drop('fail', axis=1)
                    
                    # Preparar datos para SHAP
                    input_data = st.session_state['input_data']
                    input_scaled = st.session_state['input_scaled']
                    prediction = st.session_state['prediction']
                    probability = st.session_state['probability']
                    
                    # Crear explainer de SHAP - compatible con v0.48.0
                    with st.spinner("Calculando explicación con SHAP..."):
                        # Para modelos lineales como Logistic Regression
                        explainer = shap.LinearExplainer(model, X)
                        shap_values = explainer(input_scaled)
                        
                        # Mostrar fuerza de la predicción
                        st.subheader("📊 Contribución de cada característica")
                        
                        # Obtener nombres de características
                        feature_names = X.columns.tolist()
                        
                        # Crear un DataFrame con los valores SHAP
                        shap_df = pd.DataFrame({
                            'Característica': feature_names,
                            'Valor SHAP': shap_values.values[0],
                            'Impacto': ['Aumenta riesgo' if x > 0 else 'Reduce riesgo' for x in shap_values.values[0]]
                        }).sort_values('Valor SHAP', key=abs, ascending=False)
                        
                        # Mostrar tabla de contribuciones
                        st.dataframe(shap_df, use_container_width=True)
                        
                        # VISUALIZACIÓN CORREGIDA - usar bar plot en lugar de force plot
                        st.subheader("📈 Contribución de características (Gráfico de barras)")
                        
                        # Crear gráfico de barras horizontal
                        fig, ax = plt.subplots(figsize=(12, 8))
                        
                        # Ordenar por valor absoluto para mejor visualización
                        sorted_idx = np.argsort(np.abs(shap_values.values[0]))
                        
                        colors = ['red' if x > 0 else 'blue' for x in shap_values.values[0]]
                        
                        y_pos = np.arange(len(feature_names))
                        ax.barh(y_pos, shap_values.values[0][sorted_idx], color=np.array(colors)[sorted_idx], alpha=0.7)
                        ax.set_yticks(y_pos)
                        ax.set_yticklabels(np.array(feature_names)[sorted_idx])
                        ax.set_xlabel('Valor SHAP (Impacto en la predicción)')
                        ax.set_title('Contribución de cada característica a la predicción')
                        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
                        
                        # Añadir valores en las barras
                        for i, v in enumerate(shap_values.values[0][sorted_idx]):
                            ax.text(v + (0.01 if v >= 0 else -0.05), i, f'{v:.3f}', 
                                   va='center', fontweight='bold', 
                                   color='black' if abs(v) < 0.1 else 'white')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # VALORES ESPECÍFICOS PARA DEBUG
                        with st.expander("🔍 Valores detallados para diagnóstico"):
                            st.write("**Valor esperado (base value):**", explainer.expected_value)
                            st.write("**Valores SHAP:**", shap_values.values[0])
                            st.write("**Suma de valores SHAP:**", np.sum(shap_values.values[0]))
                            st.write("**Predicción final:**", explainer.expected_value + np.sum(shap_values.values[0]))
                        
                        # Explicación en texto
                        st.subheader("🧠 Interpretación de la explicación")
                        
                        # Encontrar las características más influyentes
                        top_positive = shap_df.nlargest(3, 'Valor SHAP')
                        top_negative = shap_df.nsmallest(3, 'Valor SHAP')
                        
                        if prediction[0] == 1:
                            st.write("""
                            **🚨 La máquina tiene alta probabilidad de fallo debido a:**
                            - Valores anómalos en los sensores con mayor impacto positivo
                            - Combinación de factors que exceden los umbrales seguros
                            """)
                            
                            st.write("**📈 Factores que más aumentan el riesgo:**")
                            for _, row in top_positive.iterrows():
                                st.write(f"  - **{row['Característica']}**: {row['Valor SHAP']:.3f}")
                                
                        else:
                            st.write("""
                            **✅ La máquina opera normalmente porque:**
                            - Los valores de los sensores están dentro de rangos normales
                            - Los factores que reducen el riesgo contrarrestan los de riesgo
                            """)
                            
                            st.write("**📉 Factores que más reducen el riesgo:**")
                            for _, row in top_negative.iterrows():
                                st.write(f"  - **{row['Característica']}**: {row['Valor SHAP']:.3f}")
                        
                        # Mostrar los valores reales ingresados
                        st.subheader("📋 Valores ingresados")
                        input_df = pd.DataFrame(input_data, columns=feature_names)
                        st.dataframe(input_df, use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ Error al generar la explicación: {str(e)}")
                import traceback
                with st.expander("🔍 Ver detalles del error"):
                    st.code(traceback.format_exc())

# ============================================================
# FIN SECCIÓN: PREDECIR FALLOS
# ============================================================

# ============================================================
# SECCIÓN: RENDIMIENTO DEL MODELO
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
        y = df['fail']
        X_scaled = scaler.transform(X)
        
        # Realizar predicciones
        y_pred = model.predict(X_scaled)
        y_pred_proba = model.predict_proba(X_scaled)[:, 1]
        
        st.success("✅ Modelo y datos cargados exitosamente!")
        
        # Pestañas para organizar las métricas
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Métricas Básicas", "📋 Reporte Completo", "🎯 Matriz de Confusión", "📈 Curva ROC"])
        
        with tab1:
            st.subheader("📊 Métricas Básicas de Rendimiento")
            
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
# SECCIÓN: CONFIGURACIÓN
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