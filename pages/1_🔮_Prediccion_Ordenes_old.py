import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import torch
import joblib

# --- Imports from your custom modules ---
from forecast.name_dict import extract_id, normalize_name, assign_id
from forecast.neural_network import GlobalLSTMRNN, extract_product_features_from_prod_df, extract_inference_tensor
from aws import fetch_s3_file_as_bytes

# ---------------------------------------------------------
# Security Check
# ---------------------------------------------------------
if not st.session_state.get("authenticated", False):
    st.warning("🔒 Por favor, inicie sesión en la página de Inicio para acceder a esta herramienta.")
    st.stop()

# ---------------------------------------------------------
# Caching the Models & Metadata
# ---------------------------------------------------------
@st.cache_resource
def load_ml_objects():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Scaler
    try:
        # UPDATED NAME: clustering_scaler.pkl
        scaler = joblib.load(fetch_s3_file_as_bytes("models/clustering_scaler.pkl"))
    except Exception as e:
        st.error("❌ AWS S3 Error: No se encontró el archivo 'models/clustering_scaler.pkl'")
        st.stop()
        
    # 2. PCA
    try:
        # UPDATED NAME: clustering_pca.pkl
        pca = joblib.load(fetch_s3_file_as_bytes("models/clustering_pca.pkl"))
    except Exception as e:
        st.error("❌ AWS S3 Error: No se encontró el archivo 'models/clustering_pca.pkl'")
        st.stop()

    # 3. KMeans
    try:
        # UPDATED NAME: clustering_kmeans.pkl
        kmeans = joblib.load(fetch_s3_file_as_bytes("models/clustering_kmeans.pkl"))
    except Exception as e:
        st.error("❌ AWS S3 Error: No se encontró el archivo 'models/clustering_kmeans.pkl'")
        st.stop()
        
    # 4. Metadata (This stays the same, assuming you named it metadata_skus.csv)
    try:
        meta_df = pd.read_csv(fetch_s3_file_as_bytes("data/metadata_skus.csv"))
        meta_df["Producto ID"] = meta_df["Producto ID"].astype(str)
    except Exception as e:
        st.warning("⚠️ No se encontró 'models/metadata_skus.csv' en AWS S3. Se mostrarán los valores de confianza como 'Desconocido'.")
        meta_df = pd.DataFrame(columns=["Producto ID", "Confianza (%)", "Rango (+/-)"])
    # 5. Metadata 2 (products info)
    try:
        meta_product = pd.read_csv(fetch_s3_file_as_bytes("data/fe_products_cost.csv"))
        meta_product = meta_product.rename( columns = {"Referencia interna" : "Producto ID"} )
        meta_product["Producto ID"] = meta_product["Producto ID"].astype(str).str.split('.').str[0]
    except Exception as e:
        st.warning("⚠️ No se encontró 'data/fe_products_cost.csv' en AWS S3. Se mostrarán los valores de confianza como 'Desconocido'.")
        st.stop()
    
    # 5. Redes Neuronales
    models = {}
    for i in range(4):
        try:
            model = GlobalLSTMRNN(
                n_features=11, 
                conv_channels=64,  # Adjust to your Optuna params
                lstm_hidden=64,    # Adjust to your Optuna params
                rnn_hidden=32,     # Adjust to your Optuna params
                rnn_act_fun="TANH" # Adjust to your Optuna params
            ).to(device)
            
            # UPDATED NAME: best_lstm_model_cluster_{i}.pth
            model_bytes = fetch_s3_file_as_bytes(f"models/best_lstm_model_cluster_{i}.pth")
            model.load_state_dict(torch.load(model_bytes, map_location=device, weights_only=True))
            model.eval()
            models[i] = model
            
        except Exception as e:
            st.error(f"❌ AWS S3 Error: No se encontró la red neuronal 'models/best_lstm_model_cluster_{i}.pth'")
            st.stop()
            
    return scaler, pca, kmeans, meta_df, meta_product, models, device

# Carga en memoria la primera vez que alguien abre la app
scaler, pca, kmeans, meta_df, meta_product, cluster_models, device = load_ml_objects()

# ---------------------------------------------------------
# UI Layout
# ---------------------------------------------------------
st.title("🔴 Predicción de Órdenes (Próximas 3 Semanas)")

window_size_weeks = 30 
start_date_needed = datetime.today() - timedelta(weeks=window_size_weeks)

st.info(f"""
**Instrucciones de Carga:**
Para que la Inteligencia Artificial pueda predecir el futuro, necesita contexto histórico. 
Por favor, descargue los datos del sistema empezando desde el: **{start_date_needed.strftime('%d-%m-%Y')}** o antes.
""")

uploaded_file = st.file_uploader("Arrastre su archivo CSV de órdenes aquí", type=["csv"])

if uploaded_file is not None:
    with st.spinner('Procesando datos y ejecutando la red neuronal... 🔮'):
        
        # --- Limpieza de Datos ---
        df = pd.read_csv(uploaded_file)
        col_filter=["Cliente", "Fecha creación", "Productos", "Estado", "Cantidad en la cesta"]
        df = df[[c for c in col_filter if c in df.columns]]
        
        if "Status Farmacia Ética" in df.columns:
            df = df[df["Estado"]=="Pedido de venta"]
            
        df["Productos"] = df["Productos"].apply(lambda x: x[:x.rfind(",")] if isinstance(x, str) and "," in x else x)
        df["product_id"] = df["Productos"].apply(extract_id)
        df["normalized_name"] = df["Productos"].apply(normalize_name)

        check = df[df["product_id"].notna()].groupby("normalized_name")["product_id"].nunique()
        valid_names = check[check == 1].index

        reference = (
            df[(df["normalized_name"].isin(valid_names)) & (df["product_id"].notna())]
            .drop_duplicates("normalized_name")
            .set_index("normalized_name")["product_id"]
            .to_dict()
        )

        df["product_id_final"] = df.apply(lambda x: assign_id(x, reference), axis=1)
        df = df.dropna(subset=["product_id_final", "Fecha creación"])

        df["Fecha creación"] = pd.to_datetime(df["Fecha creación"], errors="coerce")
        df["Fecha creación"] = df["Fecha creación"].dt.strftime('%Y-%U')

        agg_df = (
            df.groupby(["Fecha creación", "product_id_final"], as_index=False)
            .agg({
                "Cliente": lambda x: set(x),
                "Cantidad en la cesta": "sum",
                "product_id_final": "first"
            })
        )

        week_str = agg_df["Fecha creación"].astype(str)
        agg_df["Fecha creación"] = pd.to_datetime(week_str + "-0", format="%Y-%U-%w", errors="coerce")
        
        mask_nat = agg_df["Fecha creación"].isna()
        if mask_nat.any():
            def week_to_date(s):
                try:
                    y, w = s.split("-")
                    p = pd.Period(year=int(y), week=int(w), freq="W-SUN")
                    return p.start_time
                except Exception:
                    return pd.NaT
            agg_df.loc[mask_nat, "Fecha creación"] = week_str[mask_nat].apply(week_to_date)

        # --- Extracción de Features e Inferencia ---
        predictions = []
        
        for pid, prod_df in agg_df.groupby("product_id_final"):
            prod_df = prod_df.set_index("Fecha creación").sort_index()
            if prod_df.index.duplicated().any():
                prod_df = prod_df.groupby(prod_df.index).agg({
                    "Cliente": lambda x: set().union(*[s if isinstance(s, set) else set() for s in x]),
                    "Cantidad en la cesta": "sum",
                    "product_id_final": "first"
                })
                
            full_idx = pd.date_range(start=prod_df.index.min(), end=prod_df.index.max(), freq="W-SUN")
            
            # 1. Reindexamos sin forzar ceros (se llenará con valores nulos temporalmente)
            prod_df = prod_df.reindex(full_idx)
            
            # 2. Rellenamos con 0 SOLO la columna de las cantidades
            prod_df["Cantidad en la cesta"] = prod_df["Cantidad en la cesta"].fillna(0)
            
            # 3. Restauramos el ID del producto para las nuevas semanas vacías
            prod_df["product_id_final"] = pid
            
            # 4. Aseguramos que los clientes sean sets vacíos
            prod_df["Cliente"] = prod_df["Cliente"].apply(lambda x: x if isinstance(x, set) else set())

            feats = extract_product_features_from_prod_df(prod_df)
            feat_df = pd.DataFrame([feats]) 
            
            X_scaled = scaler.transform(feat_df)
            X_pca = pca.transform(X_scaled)
            cluster_id = kmeans.predict(X_pca)[0]
            
            tensor_input = extract_inference_tensor(prod_df, window_size=30).to(device)
            
            model = cluster_models[cluster_id]
            with torch.no_grad():
                # El modelo escupe el valor logarítmico (log1p)
                log_pred = model(tensor_input).item() 
                
                # Invertimos el logaritmo usando expm1 (Exponencial menos 1)
                total_3_week_pred = np.expm1(log_pred) 
                
                # Prevenir cualquier número negativo por si acaso
                total_3_week_pred = int(max(0, total_3_week_pred))
            
            predictions.append({
                "Producto ID": str(pid),
                "Cluster Asignado": cluster_id,
                "Prox. 3 semanas": total_3_week_pred,
            })

        results_df = pd.DataFrame(predictions)

        # --- Cruce con la tabla de Metadatos ---
        desired_cols = ["Producto ID", "Nombre", "Prox. 3 semanas", "Rango de Error", "Confianza (%)", "Origen", "Coste"]
        results_df = results_df.merge(meta_df, on="Producto ID", how="left")
        results_df = results_df.merge(meta_product, on="Producto ID", how="left")
        results_df["Confianza (%)"] = results_df["Confianza (%)"].fillna("Desconocido")
        results_df["Rango de Error"] = results_df["Rango de Error"].fillna("Desconocido")        
        results_df = results_df[desired_cols]
        results_df = results_df.dropna()

    # ==========================================
    # 5. SIMULADOR DE COMPRAS E INVENTARIO
    # ==========================================
    st.success("✅ ¡Predicción completada exitosamente!")
    
    st.divider()
    st.header("🛠️ Simulador de Órdenes de Compra")
    st.write("Ajusta tu nivel de riesgo y modifica los valores en la tabla para simular tu inversión por país.")
    
    # --- A. Slider de Riesgo ---
    st.markdown("#### 1. Perfil de Riesgo (Agresividad de compra)")
    risk_level = st.slider(
        "0% = Conservador (- Rango Error) | 50% = Neutral (Predicción Exacta) | 100% = Agresivo (+ Rango Error)", 
        min_value=0, max_value=100, value=50, step=1
    )
    
    # Función para extraer el número puro del "Rango de Error" (ej: "± 12" -> 12.0)
    import re
    def parse_error(val):
        if pd.isna(val) or val == "Desconocido": return 0.0
        m = re.search(r'[\d\.]+', str(val))
        return float(m.group()) if m else 0.0

    # Crear una copia de los resultados para la simulación
    sim_df = results_df.copy()
    sim_df["Error Numérico"] = sim_df["Rango de Error"].apply(parse_error)
    
    # Asegurar que el Coste sea un número puro para las matemáticas
    sim_df["Coste"] = pd.to_numeric(sim_df["Coste"], errors="coerce").fillna(0.0)
    
    # Calcular nueva cantidad según el slider (Factor va de -1.0 a +1.0)
    factor = (risk_level - 50) / 50.0 
    sim_df["Prox. 3 semanas"] = (sim_df["Prox. 3 semanas"] + (sim_df["Error Numérico"] * factor)).round().astype(int)
    
    # Evitar inventarios negativos
    sim_df["Prox. 3 semanas"] = sim_df["Prox. 3 semanas"].apply(lambda x: max(0, x))
    sim_df = sim_df.drop(columns=["Error Numérico"]) # Limpiamos la columna temporal
    
    # --- B. Tabla Editable ---
    st.markdown("#### 2. Tabla Editable (Modifica Cantidades, Coste u Origen)")
    
    edited_df = st.data_editor(
        sim_df,
        use_container_width=True,
        hide_index=True,
        # Bloqueamos las columnas que el usuario NO debería poder editar
        disabled=["Producto ID", "Nombre", "Rango de Error", "Confianza (%)"] 
    )
    
    # --- C. Finanzas: Calcular Costo Total por Origen ---
    st.divider()
    st.markdown("#### 3. Inversión Estimada por Origen")
    
    # Multiplicamos la Cantidad Editada * Coste Editado
    edited_df["Inversión"] = edited_df["Prox. 3 semanas"] * edited_df["Coste"]
    
    # Agrupamos por país de origen
    cost_by_origin = edited_df.groupby("Origen")["Inversión"].sum().reset_index()
    cost_by_origin = cost_by_origin.sort_values("Inversión", ascending=False)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Mostramos la tabla resumen con formato de dinero
        st.dataframe(
            cost_by_origin.style.format({"Inversión": "${:,.2f}"}), 
            hide_index=True, 
            use_container_width=True
        )
        
        # Métrica de inversión global
        total_inv = cost_by_origin["Inversión"].sum()
        st.metric(label="Inversión Global Simulada", value=f"${total_inv:,.2f}")
        
    with col2:
        # Gráfico dinámico de barras
        if not cost_by_origin.empty and cost_by_origin["Inversión"].sum() > 0:
            st.bar_chart(data=cost_by_origin.set_index("Origen"))
        else:
            st.info("No hay datos de inversión para graficar.")

    # --- D. Descargar Simulación ---
    st.divider()
    # Quitamos la columna matemática temporal antes de descargar
    csv_to_download = edited_df.drop(columns=["Inversión"], errors="ignore").to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Descargar Simulación Final (CSV)", 
        data=csv_to_download, 
        file_name="simulacion_compras.csv", 
        mime="text/csv"
    )
    
    csv_to_download = results_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Descargar Predicciones (CSV)", data=csv_to_download, file_name="predicciones.csv", mime="text/csv")
