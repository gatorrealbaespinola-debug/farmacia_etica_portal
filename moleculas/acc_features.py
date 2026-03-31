import re
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import mannwhitneyu
import streamlit as st  



def parse_quarters(cols):
    qcols = [c for c in cols if re.match(r'^\d{4}Q[1-4]$', str(c))]
    periods = pd.PeriodIndex(qcols, freq="Q")
    return [str(p) for p in sorted(periods)]

def trailing_zeros(arr):
    arr = np.asarray(arr, dtype=float)
    return len(arr) - np.argmax(arr[::-1] != 0) if np.any(arr) else len(arr)

def gini(array):
    x = np.array(array, dtype=float).flatten()
    if x.size == 0 or np.all(x == 0): 
        return 0.0
    x = np.sort(x)
    n = x.size
    index = np.arange(1, n + 1)
    g=(np.sum((2 * index - n - 1) * x)) / (n * x.sum()) 
    return g / ((n - 1) / n) if n > 1 else 1.0

def build_features(df, last_c=-1, n_agg_cols=0):
    # Identify quarter columns
    df = df.copy()
    df.columns = df.columns.map(str) 
    qcols = list(parse_quarters(df.columns))
    qty = df[qcols].fillna(0).astype(float)
    
    if n_agg_cols>0:
        n_agg_cols+=1
        cutoff = len(qcols) + last_c + 1
        last_col = qcols[cutoff - n_agg_cols:cutoff]  # last N before cutoff
        hist_cols = qcols[:cutoff - n_agg_cols]  
    else:
        hist_cols,last_col = qcols[:last_c], qcols[last_c]
    reg_cols=qcols[:last_c]

    feats = []
    for key, row in qty.iterrows():
        if n_agg_cols>0:
            hist, last, reg = row[hist_cols].values, row[last_col].values, row[reg_cols]
        else: 
            hist, last, reg = row[hist_cols].values, row[last_col], row[reg_cols]

        mean_hist, var_hist = np.nanmean(hist), np.nanvar(hist, ddof=1)
        # T-test last vs history
        m_pval = 1.0
        if hist.size > 1:
            _, m_pval = mannwhitneyu(hist, last, alternative="two-sided")

        # Linear regression slope
        slope, slope_p = 0.0, 1.0
        if reg.size > 1:
            x = np.arange(len(reg))
            res = stats.linregress(x, reg)
            slope, slope_p, r_value, intercept = res.slope, res.pvalue, res.rvalue, res.intercept

        feats.append({
            "key": key,
            "mean_hist": mean_hist,
            "var_hist": var_hist,
            "last": last if isinstance(last,np.float64) else np.nanmean(last),
            "m_pval": m_pval,
            "slope": slope,
            "intercept": intercept,
            "mean_slope": slope/ np.nanmean(reg.values) if np.nanmean(reg.values)!=0 else 0,
            "R2": r_value**2,
            "slope_pval": slope_p,
            "trailing_zeros": trailing_zeros(row.values),
            "zero_frac_hist": (hist == 0).mean() if hist.size > 0 else 1.0,
        })

    feats = pd.DataFrame(feats).set_index("key")

    # Gini indices directly from titular_weighted_qty
    if "titular_weighted_qty" in df.columns:
        def _gini_from_dict(x):
            if isinstance(x,list):
                vals = [v for _, v in x]
                return gini(vals)

        feats["gini_qty"] = df["titular_weighted_qty"].apply(_gini_from_dict)
    else:
        feats["gini_qty"] = 0.0


    return qty, feats.fillna(0)
