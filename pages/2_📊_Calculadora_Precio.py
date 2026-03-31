import streamlit as st
import pandas as pd
import re
from io import BytesIO

from pricing.pricing import calculate_sale_price

# ---------------------
# Security Check
# ---------------------
if not st.session_state.get("authenticated", False):
    st.warning("🔒 Por favor, inicie sesión en la página de Inicio para acceder a esta herramienta.")
    st.stop()

# ---------------------
# Main App UI
# ---------------------
st.title("📊 Calculadora de Precio de Venta")

mode = st.sidebar.radio(
    "Modo de uso",
    ["📂 Cargar Excel", "✍️ Producto individual"]
)

pricing_mode_ui = st.radio(
    "Modo de fijación de precio",
    ["Basado en mercado", "Margen manual"]
)

pricing_mode = "market" if pricing_mode_ui == "Basado en mercado" else "manual"

manual_margin = None
if pricing_mode == "manual":
    manual_margin = st.slider(
        "Selecciona el margen (%)",
        min_value=20,
        max_value=200,
        value=50
    ) / 100

# -------------------------------
# Batch Mode
# -------------------------------
if mode == "📂 Cargar Excel":
    uploaded_file = st.file_uploader("Sube el archivo Excel", type=["xlsx"])

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        if "Precio de venta" in df.columns:
            df = df.drop(columns=["Precio de venta"])
        results = []

        for _, row in df.iterrows():
            raw_temp = row.get("T° Almacenamiento", 25)

            if pd.isna(raw_temp):
                cold_storage = 25
            elif isinstance(raw_temp, str):
                match = re.findall(r"\d+", raw_temp)
                cold_storage = int(match[0]) if match else 25
            else:
                cold_storage = int(raw_temp)

            result = calculate_sale_price(
                supplier_price=row["Coste"],
                market_price=row.get("Precio de mercado"),
                cold_storage=cold_storage,
                pricing_mode=pricing_mode,
                manual_margin=manual_margin
            )

            results.append(result)

        results_df = pd.DataFrame(results)

        st.subheader("🔎 Análisis de precios")
        st.dataframe(results_df, use_container_width=True)
        
        final_df = df.reset_index(drop=True)
        final_df["Precio de venta"] = [res["Precio de venta"] for res in results]
        final_df = final_df.drop(columns=[
            col for col in final_df.columns
            if col.startswith("Unnamed")
        ], errors="ignore")

        st.subheader("📋 Resultado final")
        st.dataframe(final_df, use_container_width=True)

        buffer = BytesIO()
        final_df.to_excel(buffer, index=False)
        buffer.seek(0)

        st.download_button(
            "⬇️ Descargar Excel",
            data=buffer,
            file_name="productos_con_precio.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# -------------------------------
# Single Product Mode
# -------------------------------
else:
    col1, col2 = st.columns(2)

    with col1:
        supplier_price = st.number_input("Precio proveedor", min_value=0.0, step=100.0)
        market_price = st.number_input("Precio de mercado", min_value=0.0, step=100.0)

    with col2:
        cold_storage_checked = st.checkbox("Requiere transporte refrigerado")

    cold_storage = 5 if cold_storage_checked else 25

    if st.button("Calcular precio"):
        result = calculate_sale_price(
            supplier_price=supplier_price,
            market_price=market_price if market_price > 0 else None,
            cold_storage=cold_storage,
            pricing_mode=pricing_mode,
            manual_margin=manual_margin
        )

        st.success("Precio calculado correctamente")
        st.dataframe(pd.DataFrame([result]), use_container_width=True)
