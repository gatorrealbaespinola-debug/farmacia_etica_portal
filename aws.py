import io
import boto3
import os
import streamlit as st  
import pandas as pd

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

def fetch_s3_file_as_bytes(key: str) -> io.BytesIO:
    """Descarga un archivo de S3 directamente a la memoria RAM."""
    bucket = _get_s3_bucket()
    if not bucket:
        raise RuntimeError("AWS_BUCKET no configurado.")
    s3 = s3_client()
    obj = s3.get_object(Bucket=bucket, Key=key)
    return io.BytesIO(obj["Body"].read())
