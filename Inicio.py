import streamlit as st
import pandas as pd
import hashlib
from datetime import datetime
import os
import io
import smtplib
from email.message import EmailMessage
import boto3
from pathlib import Path

# MUST BE THE FIRST COMMAND
st.set_page_config(page_title="Farmac-IA Etica Portal", page_icon="🔴", layout="wide")
# --- Custom CSS to make the sidebar text bigger ---
st.markdown(
    """
    <style>
    /* Targets standard text in the sidebar */
    [data-testid="stSidebar"] .stMarkdown p {
        font-size: 18px !important; 
    }
    
    /* Targets radio buttons and checkboxes in the sidebar */
    [data-testid="stSidebar"] .stRadio label, 
    [data-testid="stSidebar"] .stCheckbox label {
        font-size: 18px !important; 
    }
    </style>
    """,
    unsafe_allow_html=True
)
# --- All your amazing AWS, S3, and Email helper functions go here ---
# (I am keeping them exactly as you wrote them in easy_app.py)
admin_email = "gate472001@gmail.com"
users_file = Path("data/users.csv")

def s3_client():
    AWS_REGION = st.secrets.get("AWS_REGION") if hasattr(st, "secrets") else os.environ.get("AWS_REGION")
    AWS_ACCESS_KEY = st.secrets.get("AWS_ACCESS_KEY_ID") if hasattr(st, "secrets") else os.environ.get("AWS_ACCESS_KEY_ID")
    AWS_SECRET_KEY = st.secrets.get("AWS_SECRET_ACCESS_KEY") if hasattr(st, "secrets") else os.environ.get("AWS_SECRET_ACCESS_KEY")
    return boto3.client("s3", aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY, region_name=AWS_REGION)

def _get_s3_bucket():
    return st.secrets.get("AWS_BUCKET") if hasattr(st, "secrets") else os.environ.get("AWS_BUCKET")

def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()

def load_users():
    cols = ["email", "password_hash", "approved", "requested_at", "approver", "approved_at"]
    key = "data/users.csv"
    try:
        dfu = None
        if _get_s3_bucket():
            try:
                obj = s3_client().get_object(Bucket=_get_s3_bucket(), Key=key)
                dfu = pd.read_csv(io.BytesIO(obj["Body"].read()))
            except Exception:
                dfu = None
        if dfu is None and users_file.exists():
            dfu = pd.read_csv(users_file)
        if dfu is None:
            return pd.DataFrame(columns=cols)
        for c in cols:
            if c not in dfu.columns:
                dfu[c] = pd.NA
        return dfu[cols]
    except Exception:
        return pd.read_csv(users_file) if users_file.exists() else pd.DataFrame(columns=cols)

def save_users(df_users: pd.DataFrame):
    key = "data/users.csv"
    try:
        if _get_s3_bucket():
            csv_bytes = df_users.to_csv(index=False).encode("utf-8")
            s3_client().put_object(Bucket=_get_s3_bucket(), Key=key, Body=csv_bytes)
            return
    except Exception:
        pass
    users_file.parent.mkdir(parents=True, exist_ok=True)
    df_users.to_csv(users_file, index=False)

def send_email(to_email: str, subject: str, body: str) -> bool:
    conf = {}
    try:
        conf["host"] = st.secrets["SMTP_HOST"]
        conf["port"] = int(st.secrets.get("SMTP_PORT", 587))
        conf["user"] = st.secrets.get("SMTP_USER")
        conf["pw"] = st.secrets.get("SMTP_PW")
        conf["from"] = st.secrets.get("SMTP_FROM", conf["user"])
    except Exception:
        return False
    if not conf.get("host"): return False
    try:
        msg = EmailMessage()
        msg["From"] = conf.get("from")
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.set_content(body)
        server = smtplib.SMTP(conf["host"], conf.get("port", 587))
        server.starttls()
        if conf.get("user") and conf.get("pw"):
            server.login(conf["user"], conf["pw"])
        server.send_message(msg)
        server.quit()
        return True
    except Exception:
        return False

# --- UI LOGIC ---
st.title("🔴 Portal Farmac-IA etica")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user_email = None

users = load_users()

if not st.session_state.authenticated:
    st.info("👋 Bienvenido. Por favor, inicie sesión o regístrese para acceder a las herramientas.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Iniciar sesión")
        login_email = st.text_input("Email", key="login_email")
        login_pw = st.text_input("Contraseña", type="password", key="login_pw")
        if st.button("Entrar"):
            if login_email in users.get("email", []).astype(str).tolist():
                row = users.loc[users["email"] == login_email].iloc[0]
                if row["password_hash"] == hash_password(login_pw):
                    if not bool(row.get("approved", False)):
                        st.warning("Su registro está pendiente de aprobación.")
                    else:
                        st.session_state.authenticated = True
                        st.session_state.user_email = login_email
                        st.rerun() # Refresh page to show success
                else:
                    st.error("Contraseña incorrecta")
            else:
                st.error("Email no registrado.")

    with col2:
        st.subheader("Registro de usuario")
        reg_email = st.text_input("Email", key="reg_email")
        reg_pw1 = st.text_input("Contraseña", type="password", key="reg_pw1")
        reg_pw2 = st.text_input("Confirmar contraseña", type="password", key="reg_pw2")
        if st.button("Registrarme"):
            if not reg_email or "@" not in reg_email:
                st.error("Introduce un email válido")
            elif reg_pw1 != reg_pw2:
                st.error("Las contraseñas no coinciden")
            elif len(reg_pw1) < 6:
                st.error("La contraseña debe tener al menos 6 caracteres")
            elif reg_email in users.get("email", []).astype(str).tolist():
                st.warning("Email ya registrado.")
            else:
                now = datetime.utcnow().isoformat()
                new = pd.DataFrame([{"email": reg_email, "password_hash": hash_password(reg_pw1), "approved": False, "requested_at": now, "approver": pd.NA, "approved_at": pd.NA}])
                users = pd.concat([users, new], ignore_index=True)
                save_users(users)
                st.success("Registro recibido. Pendiente de aprobación.")
                send_email(admin_email, "Nuevo registro", f"Aprobar email: {reg_email}")

else:
    st.success(f"✅ Sesión iniciada como: {st.session_state.user_email}")
    st.write("### 👈 Seleccione una herramienta en el menú de la izquierda.")
    
    if st.button("Cerrar sesión"):
        st.session_state.authenticated = False
        st.session_state.user_email = None
        st.rerun()
        
    # --- ADMIN PANEL ---
    if st.session_state.user_email == admin_email:
        st.markdown("---")
        st.subheader("🛠️ Panel de Administrador")
        pending = users[users["approved"].fillna(False) == False]
        if not pending.empty:
            for i, row in pending.iterrows():
                st.write(f"**{row['email']}** — {row.get('requested_at')}")
                c1, c2 = st.columns([1, 10])
                if c1.button(f"Aprobar", key=f"ap_{i}"):
                    users.at[i, "approved"] = True
                    save_users(users)
                    st.success("Aprobado!")
                    st.rerun()
                if c2.button(f"Denegar", key=f"den_{i}"):
                    users = users.drop(index=i).reset_index(drop=True)
                    save_users(users)
                    st.rerun()
        else:
            st.write("No hay solicitudes pendientes.")
