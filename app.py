import streamlit as st
import pickle
import numpy as np

# ========== CONFIGURACIÓN ==========
st.set_page_config(
    page_title="Beer Quality Predictor",
    page_icon="🍺",
    layout="wide"
)

# ========== CARGAR MODELOS ==========
@st.cache_resource
def load_models():
    """Carga modelos scikit-learn desde disco"""
    with open('models/modelo_regresion_sklearn.pkl', 'rb') as f:
        modelo_abv = pickle.load(f)
    
    with open('models/modelo_clasificacion_sklearn.pkl', 'rb') as f:
        modelo_estilo = pickle.load(f)
    
    return modelo_abv, modelo_estilo

# Cargar modelos
try:
    modelo_abv, modelo_estilo = load_models()
    st.sidebar.success("✅ Modelos cargados correctamente")
except Exception as e:
    st.error(f"❌ Error cargando modelos: {e}")
    st.stop()

# ========== SIDEBAR INFO ==========
st.sidebar.title("ℹ️ Información del Sistema")
st.sidebar.markdown("""
**Proyecto de Big Data**  
Análisis de Calidad Cervecera 

**Autores:** Pamela Veloso - Sebastián Saravia 
**Institución:** INACAP  
**Año:** 2025

---

**Tecnologías:**
- Apache Spark (procesamiento)
- PySpark MLlib (entrenamiento)
- scikit-learn (producción)
- AWS S3 (almacenamiento)
- Streamlit (interfaz)

**Dataset:** 150 cervezas artesanales  
**Clases:** IPA, Light Lager, Premium Lager
""")

# ========== HEADER ==========
st.title("🍺 Sistema de Predicción de Calidad Cervecera")
st.markdown("### Machine Learning aplicado a la industria cervecera artesanal")
st.markdown("---")

# ========== TABS ==========
tab1, tab2, tab3 = st.tabs([
    "📊 Predicción de ABV", 
    "🔍 Clasificación de Estilo",
    "📖 Acerca del Proyecto"
])

# ========== TAB 1: PREDICCIÓN ABV ==========
with tab1:
    st.header("Predicción de Contenido Alcohólico (ABV)")
    st.markdown("**Modelo:** Random Forest Regressor | **Precisión:** R² > 0.90")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Parámetros de Entrada")
        
        with st.form("form_abv"):
            og = st.number_input(
                "Gravedad Original (OG)",
                min_value=1.030,
                max_value=1.120,
                value=1.055,
                step=0.001,
                format="%.3f",
                help="Densidad del mosto antes de fermentación"
            )
            
            ph = st.number_input(
                "pH",
                min_value=3.5,
                max_value=5.5,
                value=4.2,
                step=0.1,
                help="Nivel de acidez del mosto"
            )
            
            ibu = st.number_input(
                "IBU (Unidades de Amargor)",
                min_value=5,
                max_value=120,
                value=45,
                step=1,
                help="Nivel de amargor de la cerveza"
            )
            
            estilo = st.selectbox(
                "Estilo de Cerveza",
                ["IPA", "Light Lager", "Premium Lager"],
                help="Estilo objetivo de la cerveza"
            )
            
            submitted = st.form_submit_button("🔮 Predecir ABV", use_container_width=True)
    
    with col2:
        st.subheader("📈 Resultado de Predicción")
        
        if submitted:
            # Validar inputs
            if not (1.030 <= og <= 1.120):
                st.error("❌ OG debe estar entre 1.030 y 1.120")
            elif not (3.5 <= ph <= 5.5):
                st.error("❌ pH debe estar entre 3.5 y 5.5")
            elif not (5 <= ibu <= 120):
                st.error("❌ IBU debe estar entre 5 y 120")
            else:
                # Mapear estilo a código
                style_map = {"IPA": 0, "Light Lager": 1, "Premium Lager": 2}
                style_encoded = style_map[estilo]
                
                # Preparar features
                features = np.array([[og, ph, ibu, style_encoded]])
                
                # Predecir
                abv_pred = modelo_abv.predict(features)[0]
                
                # Mostrar resultado
                st.success(f"### 🍺 ABV Predicho: {abv_pred:.2f}%")
                
                # Rango típico
                ranges = {
                    "IPA": (5.5, 7.5),
                    "Light Lager": (4.0, 5.0),
                    "Premium Lager": (4.5, 5.5)
                }
                
                min_r, max_r = ranges[estilo]
                st.info(f"📊 Rango típico para {estilo}: {min_r}% - {max_r}%")
                
                # Validación
                if min_r <= abv_pred <= max_r:
                    st.success("✅ El ABV predicho está dentro del rango esperado")
                    progress = (abv_pred - min_r) / (max_r - min_r)
                    st.progress(progress)
                elif abv_pred < min_r:
                    st.warning(f"⚠️ ABV bajo. Diferencia: {min_r - abv_pred:.2f}%")
                    st.write("**Recomendaciones:**")
                    st.write("- Aumentar la gravedad original (OG)")
                    st.write("- Verificar salud de la levadura")
                else:
                    st.warning(f"⚠️ ABV alto. Diferencia: {abv_pred - max_r:.2f}%")
                    st.write("**Recomendaciones:**")
                    st.write("- Reducir la gravedad original (OG)")
                    st.write("- Ajustar temperatura de fermentación")

# ========== TAB 2: CLASIFICACIÓN ==========
with tab2:
    st.header("Clasificación de Estilo de Cerveza")
    st.markdown("**Modelo:** Random Forest Classifier | **Precisión:** 100%")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Parámetros de Entrada")
        
        with st.form("form_clf"):
            og2 = st.number_input(
                "Gravedad Original (OG)",
                min_value=1.030,
                max_value=1.120,
                value=1.055,
                step=0.001,
                format="%.3f",
                key="og2"
            )
            
            abv2 = st.number_input(
                "ABV (%)",
                min_value=3.0,
                max_value=12.0,
                value=5.5,
                step=0.1,
                key="abv2"
            )
            
            ph2 = st.number_input(
                "pH",
                min_value=3.5,
                max_value=5.5,
                value=4.2,
                step=0.1,
                key="ph2"
            )
            
            ibu2 = st.number_input(
                "IBU",
                min_value=5,
                max_value=120,
                value=45,
                step=1,
                key="ibu2"
            )
            
            submitted2 = st.form_submit_button("🔍 Clasificar Estilo", use_container_width=True)
    
    with col2:
        st.subheader("📈 Resultado de Clasificación")
        
        if submitted2:
            # Validar inputs
            if not (1.030 <= og2 <= 1.120):
                st.error("❌ OG debe estar entre 1.030 y 1.120")
            elif not (3.0 <= abv2 <= 12.0):
                st.error("❌ ABV debe estar entre 3.0% y 12.0%")
            elif not (3.5 <= ph2 <= 5.5):
                st.error("❌ pH debe estar entre 3.5 y 5.5")
            elif not (5 <= ibu2 <= 120):
                st.error("❌ IBU debe estar entre 5 y 120")
            else:
                # Calcular interacción
                ph_ibu_interaction = ph2 * ibu2
                
                # Preparar features
                features = np.array([[og2, abv2, ph2, ibu2, ph_ibu_interaction]])
                
                # Predecir
                clase_pred = modelo_estilo.predict(features)[0]
                probs = modelo_estilo.predict_proba(features)[0]
                
                # Mapear clase a nombre
                styles = {0: "IPA", 1: "Light Lager", 2: "Premium Lager"}
                estilo_pred = styles[int(clase_pred)]
                
                # Mostrar resultado
                st.success(f"### 🍺 Estilo Predicho: {estilo_pred}")
                st.info(f"Confianza: {probs[int(clase_pred)]*100:.1f}%")
                
                # Probabilidades
                st.markdown("**📊 Distribución de Probabilidades:**")
                
                for i in range(3):
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.progress(float(probs[i]), text=styles[i])
                    with col_b:
                        st.write(f"{probs[i]*100:.1f}%")

# ========== TAB 3: INFO ==========
with tab3:
    st.header("📖 Acerca del Proyecto")
    
    st.markdown("""
    ### Proyecto de Big Data
    
    **Título:** Análisis de Calidad Cervecera
    
    **Objetivo:** Desarrollar modelos predictivos para:
    1. Predecir el contenido alcohólico (ABV) de cervezas artesanales
    2. Clasificar automáticamente el estilo de cerveza basado en parámetros fisicoquímicos
    
    ### Tecnologías Utilizadas
    
    **Procesamiento de Datos:**
    - Amazon EMR (Elastic MapReduce)
    - Apache Spark 3.5.0
    - PySpark
    
    **Machine Learning:**
    - PySpark MLlib (entrenamiento)
    - scikit-learn (producción)
    - Random Forest (algoritmo principal)
    
    **Infraestructura:**
    - AWS S3 (almacenamiento)
    - Streamlit Cloud (deployment)
    - GitHub (versionamiento)
    
    ### Dataset
    
    - **Tamaño:** 150 registros de cervezas artesanales
    - **Distribución:** 50 IPA, 50 Light Lager, 50 Premium Lager
    - **Features:** OG, ABV, pH, IBU
    - **Split:** 70% entrenamiento (105), 30% prueba (45)
    
    ### Resultados
    
    **Modelo 1 - Regresión ABV:**
    - R² Score: > 0.90
    - RMSE: < 0.5
    - Algoritmo: Random Forest Regressor
    
    **Modelo 2 - Clasificación Estilo:**
    - Accuracy: 100%
    - F1-Score: 1.00
    - Algoritmo: Random Forest Classifier
    
    ### Autores
    
    **Nombre:** Pamela Veloso - Sebastián Saravia 
    **Institución:** INACAP  
    **Carrera:** Ingeniería en Informática  
    **Año:** 2025
    
    ---
    
    *Este proyecto demuestra la aplicabilidad de Big Data y Machine Learning 
    en la industria cervecera artesanal chilena.*
    """)

# ========== FOOTER ==========
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🍺 Beer Quality Predictor ML System | INACAP 2025</p>
</div>
""", unsafe_allow_html=True)