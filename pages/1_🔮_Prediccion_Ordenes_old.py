import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import time

# Security Check: Kick them back to the main page if not logged in
if not st.session_state.get("authenticated", False):
    st.warning("🔒 Por favor, inicie sesión en la página de Inicio para acceder a esta herramienta.")
    st.stop()

st.title("🔴 Predicción de Órdenes (Próximas 3 Semanas)")

window_size_weeks = 30 
start_date_needed = datetime.today() - timedelta(weeks=window_size_weeks)

st.info(f"""
**Instrucciones de Carga:**
Para que la Inteligencia Artificial pueda predecir el futuro, necesita contexto histórico. 
Por favor, descargue los datos del sistema empezando exactamente desde el: **{start_date_needed.strftime('%d-%m-%Y')}**.
""")

uploaded_file = st.file_uploader("Arrastre su archivo CSV de órdenes aquí", type=["csv"])

if uploaded_file is not None:
    with st.spinner('Procesando datos y ejecutando la red neuronal... 🔮'):
        time.sleep(2) # Placeholder for your PyTorch code
        
        fake_results = {
            "Producto": ["Aspirina 500mg", "Ibuprofeno 200mg", "Vitamina C"],
            "Semana 1 (Pred)": [120, 85, 300],
            "Semana 2 (Pred)": [125, 90, 310],
            "Semana 3 (Pred)": [115, 80, 290],
            "Confianza (%)": [88.5, 92.1, 75.0],
            "Rango (+/-)": ["± 12", "± 8", "± 45"]
        }
        results_df = pd.DataFrame(fake_results)

    st.success("✅ ¡Predicción completada exitosamente!")
    st.dataframe(results_df, use_container_width=True)
    
    csv_to_download = results_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Descargar Predicciones (CSV)", data=csv_to_download, file_name="predicciones.csv", mime="text/csv")
