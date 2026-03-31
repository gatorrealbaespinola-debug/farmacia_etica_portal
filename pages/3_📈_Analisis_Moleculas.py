import streamlit as st
import pandas as pd
import os
import ast
import io
import boto3
import moleculas.app_da as ada  # 👈 Updated to point to the new folder

# ---------------------
# Security Check
# ---------------------
if not st.session_state.get("authenticated", False):
    st.warning("🔒 Por favor, inicie sesión en la página de Inicio para acceder a esta herramienta.")
    st.stop()

# ---------------------
# Config
# ---------------------
path_imp = "data/imports_2020_2025.csv"  # Path to daily imports
path_ana= "data/Importaciones_trim_3.csv" # Path to imports and technical analysis

# ---------------------
# Load & Save Dataset Helpers
# ---------------------
@st.cache_data
def load_data(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    return df

def save_data(df,p):
    df.to_csv(p, index=False, encoding="utf-8-sig")

# ---------------------
# S3 Data Helpers
# ---------------------
def s3_client():
    AWS_REGION = st.secrets.get("AWS_REGION") if hasattr(st, "secrets") else None
    AWS_REGION = AWS_REGION or os.environ.get("AWS_REGION")
    AWS_ACCESS_KEY = (st.secrets.get("AWS_ACCESS_KEY_ID") if hasattr(st, "secrets") else None) or os.environ.get("AWS_ACCESS_KEY_ID")
    AWS_SECRET_KEY = (st.secrets.get("AWS_SECRET_ACCESS_KEY") if hasattr(st, "secrets") else None) or os.environ.get("AWS_SECRET_ACCESS_KEY")
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION,
    )

def _get_s3_bucket():
    bucket = (st.secrets.get("AWS_BUCKET") if hasattr(st, "secrets") else None) or os.environ.get("AWS_BUCKET")
    return bucket

def read_csv_from_s3(key: str) -> pd.DataFrame:
    bucket = _get_s3_bucket()
    if not bucket:
        raise RuntimeError("AWS_BUCKET not configured in st.secrets or environment variables")
    s3 = s3_client()
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))

def upload_df_to_s3(df: pd.DataFrame, key: str) -> bool:
    bucket = _get_s3_bucket()
    if not bucket:
        raise RuntimeError("AWS_BUCKET not configured in st.secrets or environment variables")
    s3 = s3_client()
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=csv_bytes)
    return True

# ---------------------
# Main App UI
# ---------------------
st.title("Analisis Productos :red[Farmaciaetica]")

# --- Reset button ---
if st.button("🔄 Nueva búsqueda", key="reset_button"):
    st.session_state.chosen_key = None

if "chosen_key" not in st.session_state:
    st.session_state.chosen_key = None

# --- Cache df2 ---
if "df2_cache" not in st.session_state:
    try:
        st.session_state["df2_cache"] = read_csv_from_s3(path_imp)
    except Exception:
        st.session_state["df2_cache"] = pd.read_csv(path_imp)

# Prefer S3-backed analysis if configured
try:
    df = read_csv_from_s3(path_ana)
except Exception:
    df = load_data(path_ana)
    
df2 = st.session_state["df2_cache"]

# --- Search logic ---
if st.session_state.chosen_key is None:
    all_keys = df["key"].unique().tolist()

    # Text input with stable key
    search_key = st.text_input(
        "Buscar principio activo + dosis:",
        value="",
        key="search_input"
    )

    if len(search_key) > 1:
        matches = [k for k in all_keys if search_key.lower() in k.lower()]
    else:
        matches = []

    if matches:
        chosen_key = st.selectbox(
            "Coincidencias:",
            matches,
            index=None,
            key=f"selectbox_{search_key}",
        )
        if chosen_key:
            st.session_state.chosen_key = chosen_key

else:
    chosen_key = st.session_state.chosen_key
    # Filter dataset to only include rows where 'key' matches the chosen_key
    sub = df[df["key"] == chosen_key]
    
    # reading reg as a list and creating a row for each registro
    regs = sub["reg"].apply(lambda x: ast.literal_eval(x)).explode("reg")
    df3 = pd.DataFrame()

    for r in regs:
        add_in = df2[df2["Registro"] == r]
        add_in = add_in[["NomProd", "Registro", "Titular", "mult"]].head(1)
        df3 = pd.concat([df3, add_in], ignore_index=True)

    st.write("### Registros encontrados:")
    st.write(f"## {chosen_key}")
    df_pres = df3.rename(columns={"mult": "Presentación (por unidad)"})
    df_pres.set_index('NomProd', inplace=True)
    st.dataframe(df_pres)

    refine = st.radio("¿Desea refinar por presentación?", ["No", "Sí"])

    if 'show_new_button' not in st.session_state:
        st.session_state.show_new_button = False

    if refine == "Sí":
        u = True 
        updated = False
        for prod, reg, tit, m in df3.values:
            pres = f"{reg} – {prod} – {tit}"
            mult = st.number_input(f"Ingrese presentación (factor) para:\n {pres}", min_value=1, step=1, value=int(m))

            if (mult != m) and (mult > 0) and isinstance(mult, int):
                mask = (df2["Registro"] == reg) & (
                    df2["UMedida"].str.contains("caja") | df2["UMedida"].str.contains("estuche")
                )
                df2.loc[mask, "Cantidad"] = df2.loc[mask, "Cantidad_base"] * mult
                df2.loc[df2["Registro"] == reg, "mult"] = mult
                st.success(f"{len(df2.loc[mask, ['Cantidad_base','mult','Cantidad']])} importaciones actualizadas✅")
                updated = True
            elif not isinstance(mult, int):
                st.error("Elija un numero positivo para la presentacion del producto!")

        if updated:
            upload_df_to_s3(df2, path_imp)
            del st.session_state["df2_cache"]
            st.success("Datos actualizados y guardados permanentemente ✅")
            st.session_state.show_new_button = True
    else:
        u = False
        st.session_state.show_new_button = True

    # Show new button only after 'Sí' process is completed
    if st.session_state.get('show_new_button', False):
        if 'analysis_active' not in st.session_state:
            st.session_state.analysis_active = False

        start_analysis = st.button("🔄 Comenzar Analisis")
        if start_analysis:
            st.session_state.analysis_active = True

        if st.session_state.get('analysis_active', False):
            st.info(f"Procesando Molecula: {chosen_key}")

            imp_trim = ada.single_mol_analisis(df2, regs)
            df_mol = pd.DataFrame()
            if u:
                df_mol = ada.single_mol_analisis(df2, regs, plot=False)

            tab_report, tab_regs = st.tabs(["Reporte por molecula", "Importaciones por registro"])

            with tab_report:
                st.write("## Tendencia importaciones:")
                ana_mol = ada.molecule_trend(df, df2, chosen_key, df_mol, update=u)

                if ana_mol is not None:
                    ana_mol = ana_mol.reset_index().loc[0].reindex(df.columns)
                    mask = (df["key"] == chosen_key)
                    existing_row = df[mask].iloc[0]
                    ana_mol = ana_mol.fillna(existing_row)
                    
                    df.set_index('key', inplace=True)
                    df.loc[chosen_key] = ana_mol
                    df.reset_index(inplace=True)
                    
                    try:
                        upload_df_to_s3(df, path_ana)
                    except Exception:
                        save_data(df, path_ana)
                        
                    try:
                        load_data.clear()
                    except Exception:
                        pass
                        
                    try:
                        df = read_csv_from_s3(path_ana)
                    except Exception:
                        df = load_data(path_ana)
                        
                    st.session_state["df_ana_cache"] = df
                    st.success("Análisis guardado✅")

            with tab_regs:
                st.write("## Importaciones por registro ultimos años:")
                ada.plot_imports_by_trim(imp_trim)

            if st.button("⏪ Terminar análisis y volver"):
                st.session_state.analysis_active = False
                st.session_state.show_new_button = False
