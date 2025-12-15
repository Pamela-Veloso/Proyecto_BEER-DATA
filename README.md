# 🍺 Beer Quality Predictor - ML System

Sistema de predicción de calidad cervecera usando Machine Learning desarrollado como proyecto de tesis.

## 📊 Modelos

### Modelo 1: Predicción de ABV
- **Algoritmo:** Random Forest Regressor
- **Features:** OG, pH, IBU, Estilo
- **Métricas:** R² > 0.90, RMSE < 0.5

### Modelo 2: Clasificación de Estilo
- **Algoritmo:** Random Forest Classifier
- **Features:** OG, ABV, pH, IBU, pH×IBU
- **Métricas:** Accuracy 100%, F1-Score 1.00

## 🚀 Tecnologías

- **Data Processing:** Apache Spark, AWS EMR
- **ML Training:** PySpark MLlib
- **Production:** scikit-learn
- **Interface:** Streamlit
- **Deployment:** Streamlit Cloud
- **Storage:** AWS S3

## 📦 Dataset

- 150 cervezas artesanales
- 3 estilos: IPA, Light Lager, Premium Lager
- Split: 70% train (105), 30% test (45)

## 🎓 Autora

**Pamela**  
INACAP - Ingeniería en Informática  
Proyecto de Tesis 2025

## 🔗 Demo

URL de producción: [Se actualizará después del deployment]

## 📝 Licencia

Proyecto académico - INACAP 2025