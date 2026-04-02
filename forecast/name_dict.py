import pandas as pd
import re
import unicodedata

def normalize_name(text):
    if pd.isna(text):
        return None

    text = text.lower()

    # remove accents
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )

    # remove ID
    text = re.sub(r"\[\d+\]", "", text)

    # normalize decimal separator
    text = text.replace(",", ".")

    # normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_id(text):
    if pd.isna(text):
        return None
    m = re.search(r"\[(\d+)\]", text)
    return m.group(1) if m else None

def assign_id(row,reference):
    if row["product_id"] is not None:
        return row["product_id"]
    return reference.get(row["normalized_name"], None)
