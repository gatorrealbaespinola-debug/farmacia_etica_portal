import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
import plotly.express as px
import ast
from .acc_features import build_features

# give real significance to p_value maybe with means_square error?
# add R2 to the plot
# add Si implementation also for the easy_app.py in the case of the last analysis
# put data in drive and release the beta of the app ask chatgpt
# Añadir eleccion de porcentaje de mercado y un bar plot con el porcentaje de importaciones de cada titular
# este ultimo deberia tener la informacion de cuanto es recomendable como porcentaje de mercado.

def all_t(df):
    all_trims = pd.period_range(
    df["trims"].min(),
    df["trims"].max(),
    freq="Q"
    ).astype(str)

    all_trims.name = "trims"  # 👈 ADD THIS LINE HERE!
    
    return all_trims

def single_mol_analisis(df,regs,plot=True):
    regs_set=set(regs)
    df=df[df["Registro"].isin(regs_set)]
    df["FchResolucion"]=pd.to_datetime(df["FchResolucion"])
    df["trims"]=df["FchResolucion"].dt.to_period("Q")
    df["trims"] = df["trims"].astype(str)
    all_trims = all_t(df)
    if plot:
        imp_trim = (
    df.groupby(["Registro", "trims"])["Cantidad"]
      .sum()
      .unstack(fill_value=0)
      .reindex(columns=all_trims, fill_value=0)
      .stack()
      .reset_index(name="Cantidad")
    )
    elif not plot:
        imp_trim=df.groupby(["Registro","trims"]).agg({"Cantidad":sum}).unstack(fill_value=0)
    return imp_trim
def titular_mol_df(df_imp,regs,key):
    df_imp=df_imp[df_imp["Registro"].isin(regs)]
    df_imp["FchResolucion"]=pd.to_datetime(df_imp["FchResolucion"])
    df_imp["trims"]=df_imp["FchResolucion"].dt.to_period("Q")
    all_trims=all_t(df_imp)
    imp_trim = (
    df_imp.groupby(["Registro", "trims"])
          .agg({"Cantidad": "sum", "Titular": set})
          .unstack(fill_value=0)
    )
    # Reindex trimesters
    imp_trim = imp_trim.reindex(columns=all_trims, fill_value=0)
    # back to long format
    imp_trim = imp_trim.stack().reset_index()
    trims_sorted = sorted(imp_trim["trims"].unique(),reverse=True)  # ensure chronological order
    n_trims = len(trims_sorted)
    weights = np.linspace(1,0, n_trims)
    weight_dict = dict(zip(trims_sorted, weights))
    imp_trim["Titular"] = imp_trim["Titular"].apply(
    lambda x: x if isinstance(x, set) else set()
    )
    df_exp = imp_trim.explode("Titular").dropna(subset=["Titular"])
    # Apply weight based on trimester
    df_exp["weighted_qty"] = df_exp.apply(lambda row: int(row["Cantidad"] * weight_dict[row["trims"]]), axis=1)



    # Group by molecule and titular to get cumulative weighted quantity
    titular_qty = (
        df_exp.groupby("Titular")["weighted_qty"]
        .sum()
        .reset_index()
    )
    titular_qty["key"]=key
    titular_dict = (
        titular_qty.groupby("key").apply(lambda x: dict(zip(x["Titular"], x["weighted_qty"])))
    )

    titular_dict=titular_dict.apply(lambda x: [(k,round(((v/sum(x.values()))*100),2)) for k,v in x.items() if sum(x.values())>0])
    return titular_dict

def molecule_trend(df_ana,df_imp,key,df_mol,update=False):
    def plot_imports_and_trend(imp, anal,table=True):
        sns.set_theme(style="whitegrid")

        # --- Prepare imports data ---
        imp_t = imp.T.reset_index()
        imp_t.columns = ["Trimestre", "Cantidad"]
        imp_t["Cantidad"] = pd.to_numeric(imp_t["Cantidad"], errors="coerce")

        # --- Extract regression info ---
        slope = anal["slope"].values[0]
        intercept = anal["intercept"].values[0]
        pval = anal["slope_pval"].values[0]

        # --- Prepare numeric X for regression line ---
        x = np.arange(len(imp_t))
        y_pred = intercept + slope * x

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=imp_t, x="Trimestre", y="Cantidad", marker="o", linewidth=2.5, ax=ax, color="#1f77b4", label="Importaciones reales")

        # Add regression (trend) line
        ax.plot(imp_t["Trimestre"], y_pred, "--", color="#ff7f0e", label="Tendencia (modelo lineal)")

        # Title & axes
        ax.set_title("Evolución trimestral de importaciones", fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Trimestre", fontsize=12)
        ax.set_ylabel("Cantidad importada", fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.legend()

        # Annotate slope info
        color = "green" if pval < 0.05 else "gray"
        ax.text(
            0.02, 0.95,
            f"Slope = {slope:.2f}\nP-valor = {pval:.3f}",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor=color)
        )

        # Streamlit display
        st.pyplot(fig)

        # --- Explanation ---
        st.markdown(
            f"""
            #### ℹ️ Interpretación del valor *p* de la pendiente
            El **valor p ({pval:.3f})** indica la probabilidad de que la tendencia observada sea aleatoria.
            - Si *p* < 0.05 → la tendencia es **estadísticamente significativa**.
            - Si *p* ≥ 0.05 → no hay evidencia suficiente para afirmar una tendencia real.
            """
        )
        # --- Bar chart for titular weighted quantities (only if present) ---
        try:
            tw_raw = None
            # anal may be a DataFrame slice, or a dict-like object when update path used
            if isinstance(anal, dict):
                tw_raw = anal.get("titular_weighted_qty")
            else:
                if hasattr(anal, "columns") and "titular_weighted_qty" in anal.columns:
                    tw_raw = anal["titular_weighted_qty"].iloc[0]

            if tw_raw is None:
                # nothing to plot
                tw = None
            else:
                # parse stringified dict/list if necessary
                if isinstance(tw_raw, str):
                    try:
                        tw = ast.literal_eval(tw_raw)
                    except Exception:
                        tw = None
                else:
                    tw = tw_raw

            items = []
            if isinstance(tw, dict):
                items = list(tw.items())
            elif isinstance(tw, (list, tuple)):
                # list of tuples or list of [k,v]
                for el in tw:
                    if isinstance(el, (list, tuple)) and len(el) >= 2:
                        items.append((el[0], el[1]))

            if items:
                df_tw = pd.DataFrame(items, columns=["Titular", "WeightedQty"])
                df_tw = df_tw.sort_values("WeightedQty", ascending=False)

                # Try get gini (if anal is DataFrame-like)
                gini = None
                try:
                    if hasattr(anal, "columns") and "gini_qty" in anal.columns:
                        gini = float(anal["gini_qty"].iloc[0])
                except Exception:
                    gini = None

                title = "Distribución ponderada por Titular"
                if gini is not None:
                    title = f"{title} — Gini: {gini:.3f}"
                df_tw["Titular"] = df_tw["Titular"].apply(lambda x: str(x).lstrip("{").rstrip("}").strip("'"))
                fig_bar = px.bar(df_tw, x="Titular", y="WeightedQty", title=title,
                                 labels={"WeightedQty": "Weighted (pct)", "Titular": "Titular"},
                                 template="plotly_white")
                fig_bar.update_layout(xaxis_tickangle=-45, margin=dict(t=60, b=160))
                st.plotly_chart(fig_bar, use_container_width=True)
        except Exception as e:
            # only warn, don't break the main analysis
            st.warning(f"No se pudo generar gráfico de titulares: {e}")
        if table:
            # After: y_pred = intercept + slope * x

            # 1. Find the last trimester index and year
            first_trim = imp_t["Trimestre"].iloc[1]
            first_year = int(first_trim[:4])
            known_years = len(imp_t)//4
            last_year = first_year + known_years - 1
            x_last = (known_years*4) + 1  # Next x to use

            # 2. Predict for next 20 trimesters
            future_x = np.arange(x_last, x_last + 20)
            future_y_pred = intercept + slope * future_x

            # 3. Aggregate every 4 trimesters to get yearly predictions
            years = [last_year + i for i in range(1, 6)]
            yearly_preds = [int(future_y_pred[i*4:(i+1)*4].sum()) for i in range(5)]

            # 4. Create DataFrame
            df_pred = pd.DataFrame({
                "Year": years,
                "Predicted Quantity": yearly_preds
            })

            st.write("### Predicción anual para los próximos 5 años")

            # Allow the user to specify an adjustment percentage (0-100)
            pct = st.slider("Ajustar predicción anual por porcentaje (%)", min_value=0, max_value=100, value=100, step=1)
            # Compute adjusted predicted quantity
            df_pred["Adj Predicted Quantity"] = (df_pred["Predicted Quantity"] * (pct / 100.0)).round().astype(int)
            df_pred.set_index("Year", inplace=True)
            st.write(f"Usando un ajuste de {pct}% sobre las predicciones anuales")
            st.dataframe(df_pred)
    if update:
        df_mol["key"] = key
        # Group by 'key' and sum all numeric columns, keep first for non-numeric
        
        df_mol_key = df_mol.groupby("key").agg(lambda x: x.sum() if np.issubdtype(x.dtype, np.number) 
                                           else x.iloc[0])
        df_mol_key.columns = df_mol_key.columns.get_level_values(1)
        regs=set(df_mol.index)
        titular=titular_mol_df(df_imp,regs,key)
        df_mol_key=df_mol_key.merge(titular.rename("titular_weighted_qty"),on="key",how="left")
        qty, feat = build_features(df_mol_key, last_c=-2, n_agg_cols=3)
        feat["titular_weighted_qty"]=df_mol_key["titular_weighted_qty"]
        plot_imports_and_trend(qty, feat)
        anal_combined = pd.concat([qty, feat], axis=1)
        return anal_combined
    else:
        c=df_ana.columns
        c_imp=c.get_loc("2020Q1")
        c_s=c.get_loc("Promedio_imp")
        statss=df_ana[df_ana["key"]==key]
        anal=statss.iloc[:,c_s:]
        imp=statss.iloc[:,c_imp:c_s]
        plot_imports_and_trend(imp, anal)
        return None


def plot_imports_by_trim(df):
    # Ensure 'trims' is categorical with correct order
    df["trims"] = pd.Categorical(
    df["trims"],
    categories=sorted(df["trims"].unique(), key=lambda x: (int(x[:4]), int(x[-1]))),
    ordered=True)
    sns.set_theme(style="whitegrid", palette="tab10")
    # Unique registros
    unique_regs = df["Registro"].unique()
    n_regs = len(unique_regs)
    # 🔹 Compute y-axis range for all plots
    y_min, y_max = df["Cantidad"].min(), df["Cantidad"].max()
    y_margin = (y_max - y_min) * 0.05  # small margin for nicer look
    y_limits = (max(0, y_min - y_margin), y_max + y_margin)
   
    def _plot_subset(subset_regs, title):
        fig, ax = plt.subplots(figsize=(10, 5))
        sub = df[df["Registro"].isin(subset_regs)]
        sns.lineplot(
            data=sub,
            x="trims",
            y="Cantidad",
            hue="Registro",
            marker="o",
            linewidth=2,
            ax=ax,
        )
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Trimestre")
        ax.set_ylabel("Cantidad")
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(y_limits)  # ✅ same scale for all plots
        ax.legend(title="Registro", bbox_to_anchor=(1.05, 1), loc="upper left")
        sns.despine()
        st.pyplot(fig)
    # --- Plot based on number of regs ---
    if n_regs <= 5:
        _plot_subset(unique_regs, "Importaciones por Trimestre y Registro")
    else:
        half = n_regs // 2
        _plot_subset(unique_regs[:half], "Importaciones (Grupo 1)")

        _plot_subset(unique_regs[half:], "Importaciones (Grupo 2)")
