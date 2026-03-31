import streamlit as st
import pandas as pd
import os
import ast
import io
import boto3
import moleculas.app_da as ada  

# ---------------------
# Security Check
# ---------------------
if not st.session_state.get("authenticated", False):
    st.warning("🔒 Por favor, inicie sesión en la página de Inicio para acceder a esta herramienta.")
    st.stop()

# ---------------------
# Config
# ---------------------
path_imp = "data/imports_2020_2025.csv"  
path_ana = "data/Importaciones_trim_3.csv" 

# ---------------------
# AWS S3 Helpers
# ---------------------
def s3_client():
    AWS_REGION = st.secrets.get("AWS_REGION") or os.environ.get("AWS_REGION")
    AWS_ACCESS_KEY = st.secrets.get("AWS_ACCESS_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID")
    AWS_SECRET_KEY = st.secrets.get("AWS_SECRET_ACCESS_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY")
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION,
    )

def _get_s3_bucket():
    return st.secrets.get("AWS_BUCKET") or os.environ.get("AWS_BUCKET")

def read_csv_from_s3(key: str) -> pd.DataFrame:
    bucket = _get_s3_bucket()
    if not bucket:
        raise RuntimeError("AWS_BUCKET no configurado.")
    s3 = s3_client()
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))

def upload_df_to_s3(df: pd.DataFrame, key: str) -> bool:
    bucket = _get_s3_bucket()
    if not bucket:
        raise RuntimeError("AWS_BUCKET no configurado.")
    s3 = s3_client()
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=csv_bytes)
    return True

# ---------------------
# Main App UI
# ---------------------
st.title("Analisis Productos :red[Farmaciaetica]")

# --- Reset button ---
if st.button("🔄 Nueva búsqueda"):
    st.session_state.chosen_key = None

if "chosen_key" not in st.session_state:
    st.session_state.chosen_key = None

# --- Cache & Load Data from AWS ---
if "df2_cache" not in st.session_state:
    try:
        st.session_state["df2_cache"] = read_csv_from_s3(path_imp)
    except Exception as e:
        st.error(f"❌ Error descargando imports de AWS: {e}")
        st.stop()

try:
    if "df_ana_cache" not in st.session_state:
        st.session_state["df_ana_cache"] = read_csv_from_s3(path_ana)
    df = st.session_state["df_ana_cache"]
except Exception as e:
    st.error(f"❌ Error descargando análisis de AWS: {e}")
    st.stop()

df2 = st.session_state["df2_cache"]

# --- Search logic ---
if st.session_state.chosen_key is None:
    all_keys = df["key"].unique().tolist()
    search_key = st.text_input("Buscar principio activo + dosis:", value="")

    matches = [k for k in all_keys if search_key.lower() in k.lower()] if len(search_key) > 1 else []

    if matches:
        chosen_key = st.selectbox("Coincidencias:", matches, index=None)
        if chosen_key:
            st.session_state.chosen_key = chosen_key
            st.rerun()

else:
    chosen_key = st.session_state.chosen_key
    sub = df[df["key"] == chosen_key]
    
    # Extracting registros
    regs = sub["reg"].apply(lambda x: ast.literal_eval(x)).explode("reg")
    df3 = pd.DataFrame()

    for r in regs:
        add_in = df2[df2["Registro"] == r][["NomProd", "Registro", "Titular", "mult"]].head(1)
        df3 = pd.concat([df3, add_in], ignore_index=True)

    st.write("### Registros encontrados:")
    st.write(f"## {chosen_key}")
    df_pres = df3.rename(columns={"mult": "Presentación (por unidad)"}).set_index('NomProd')
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

            if (mult != m) and (mult > 0):
                mask = (df2["Registro"] == reg) & (
                    df2["UMedida"].str.contains("caja") | df2["UMedida"].str.contains("estuche")
                )
                df2.loc[mask, "Cantidad"] = df2.loc[mask, "Cantidad_base"] * mult
                df2.loc[df2["Registro"] == reg, "mult"] = mult
                st.success(f"✅ {len(df2.loc[mask])} importaciones actualizadas.")
                updated = True

        if updated:
            try:
                # OVERWRITE AWS S3 WITH NEW PRESENTATIONS
                upload_df_to_s3(df2, path_imp)
                del st.session_state["df2_cache"] # Clear cache to force fresh download next time
                st.success("☁️ Datos actualizados y guardados permanentemente en AWS ✅")
            except Exception as e:
                st.error(f"Error subiendo a AWS: {e}")
            st.session_state.show_new_button = True
    else:
        u = False
        st.session_state.show_new_button = True

    if st.session_state.get('show_new_button', False):
        if 'analysis_active' not in st.session_state:
            st.session_state.analysis_active = False

        if st.button("🔄 Comenzar Analisis"):
            st.session_state.analysis_active = True

        if st.session_state.get('analysis_active', False):
            st.info(f"Procesando Molecula: {chosen_key}")

            imp_trim = ada.single_mol_analisis(df2, regs)
            df_mol = ada.single_mol_analisis(df2, regs, plot=False) if u else pd.DataFrame()

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
                        # OVERWRITE AWS S3 WITH NEW ANALYSIS
                        upload_df_to_s3(df, path_ana)
                        st.session_state["df_ana_cache"] = df # Update local session state
                        st.success("☁️ Análisis guardado permanentemente en AWS ✅")
                    except Exception as e:
                        st.error(f"Error subiendo análisis a AWS: {e}")

            with tab_regs:
                st.write("## Importaciones por registro ultimos años:")
                
                # 👈 Quick fix: Rename 'level_1' back to 'trims' if Pandas messed it up
                if "level_1" in imp_trim.columns:
                    imp_trim = imp_trim.rename(columns={"level_1": "trims"})
                    
                ada.plot_imports_by_trim(imp_trim)
            if st.button("⏪ Terminar análisis y volver"):
                st.session_state.analysis_active = False
                st.session_state.show_new_button = False
                st.session_state.chosen_key = None
                st.rerun()
