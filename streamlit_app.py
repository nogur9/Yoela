"""Streamlit EDA dashboard for welfare-office workload survey (encoded wide format)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

COL_Y = "ערך משתנה"
COL_PARAM = "משתנה"
LABEL_IMPORTANCE = "חשיבות המשתנה"
PREFIX_POPULATION = "מאפייני אוכלוסייה_"
PREFIX_LOAD = "תחומי העומס_"
OFFICE_NUMERIC_CONTEXT = [
    "סוג הרשות",
    "אשכול חברתי-כלכלי",
    "דיוק מדידה",
    "פער מול תקינה",
]
OFFICE_BINARY_CONTEXT = ['מאפייני אוכלוסייה_אוכלוסייה כללית',
       'מאפייני אוכלוסייה_אוכלוסייה מעורבת', 'מאפייני אוכלוסייה_חברה חרדית',
       'מאפייני אוכלוסייה_חברה ערבית', 'תחומי העומס_מוגבלויות',
       'תחומי העומס_אזרחים ותיקים', 'תחומי העומס_חוק סדרי דין',
       'תחומי העומס_חוק נוער', 'תחומי העומס_משפחה',
       'תחומי העומס_תחום נוער וצעירים']
Y_NUM_ALIAS = "__y_num__"
SPECIAL_PARAM_FROM = "עבודה חוקית / רגולטורית ( האם יש מעורבות עו״ס לחוק)"
SPECIAL_PARAM_TO = "מעורבות חוק"

AUTHORITY_ORDER: dict[str, int | None] = {
    "עיר גדולה (מעל 200 אלף תושבים)": 5,
    'עיר גדולה (100- 200 אלף תושבים)': 4,
    "עיר בינונית (50-100 אלף תושבים": 3,
    "עיר קטנה (20-50 אלף תושבים)": 2,
    "מועצה מקומית (עד 20 אלף תושבים)": 1,
    "מועצה אזורית": None,
}

CLUSTER_ORDER: dict[str, int] = {
    "7–10": 9,
    "1–3": 2,
    "4–6": 5,
}

GAP_ORDER: dict[str, int] = {
    "במידה מועטה": 1,
    "במידה בינונית": 2,
    "במידה גבוהה": 3,
    "במידה חריגה מאוד": 4,
}


def _default_data_path() -> Path:
    return Path("survey_encoded.xlsx")


@st.cache_data(show_spinner=False)
def load_survey(path_str: str) -> pd.DataFrame:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_excel(path, engine="openpyxl")
    return df


def prepare_display(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "index" in out.columns:
        out = out.drop(columns=["index"])
    return out


def ohe_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    return [c for c in df.columns if isinstance(c, str) and c.startswith(prefix)]


def to_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.map(
        lambda v: v is True
        or v == 1
        or (isinstance(v, str) and v.strip().lower() in ("true", "1", "yes"))
    ).fillna(False)


def unique_office_frame(df: pd.DataFrame) -> pd.DataFrame:
    pop_cols = ohe_columns(df, PREFIX_POPULATION)
    load_cols = ohe_columns(df, PREFIX_LOAD)
    key_cols = [c for c in OFFICE_NUMERIC_CONTEXT if c in df.columns] + pop_cols + load_cols
    if not key_cols:
        return df.drop_duplicates()
    return df.drop_duplicates(subset=key_cols, ignore_index=True)


def legend_table_from_mapping(mapping: dict[Any, Any], value_label: str) -> pd.DataFrame:
    rows: list[tuple[Any, Any]] = []
    for raw_label, code in mapping.items():
        rows.append((code, raw_label))
    t = pd.DataFrame(rows, columns=[value_label, "מה זה אומר (בשאלון)"])
    return t.sort_values(value_label, na_position="last")


def download_csv_button(data: pd.DataFrame, name: str, key: str) -> None:
    csv_bytes = data.to_csv(index=False).encode("utf-8-sig")
    st.download_button("CSV", csv_bytes, file_name=f"{name}.csv", mime="text/csv", key=key)


def apply_axis_style(fig: go.Figure, x_label: str, y_label: str) -> None:
    fig.update_xaxes(title_text=f"<b>{x_label}</b>", title_font={"size": 18})
    fig.update_yaxes(title_text=f"<b>{y_label}</b>", title_font={"size": 18})


def plot_config(download_name: str) -> dict[str, Any]:
    return {
        "displaylogo": False,
        "toImageButtonOptions": {"format": "png", "filename": download_name},
    }


def show_plot(fig: go.Figure, download_name: str) -> None:
    st.plotly_chart(fig, use_container_width=True, config=plot_config(download_name))


def inverse_mapping(mapping: dict[str, int | None]) -> dict[int | float, str]:
    out: dict[int | float, str] = {}
    for label, val in mapping.items():
        if val is not None:
            out[val] = label
            out[float(val)] = label
    return out


AUTHORITY_LABELS = inverse_mapping(AUTHORITY_ORDER)
CLUSTER_LABELS = inverse_mapping(CLUSTER_ORDER)
GAP_LABELS = inverse_mapping(GAP_ORDER)


def display_value(col: str, value: Any) -> str:
    if pd.isna(value):
        return "חסר"
    if col == "סוג הרשות":
        return AUTHORITY_LABELS.get(value, str(value))
    if col == "אשכול חברתי-כלכלי":
        return CLUSTER_LABELS.get(value, str(value))
    if col == "פער מול תקינה":
        return GAP_LABELS.get(value, str(value))
    return str(value)


def normalize_param_name(value: Any) -> Any:
    if isinstance(value, str) and value.strip() == SPECIAL_PARAM_FROM:
        return SPECIAL_PARAM_TO
    return value


def make_x_categories(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        s_num = pd.to_numeric(series, errors="coerce")
        valid = s_num.dropna()
        if valid.nunique() > 8:
            try:
                bins = pd.qcut(valid, q=6, duplicates="drop").astype(str)
            except ValueError:
                bins = pd.cut(valid, bins=min(6, valid.nunique())).astype(str)
            out = pd.Series("חסר", index=series.index, dtype="object")
            out.loc[valid.index] = bins
            return out
    return series.fillna("חסר").astype(str)


def y_is_numeric_or_binary(col: str, series: pd.Series, pop_cols: list[str], load_cols: list[str]) -> bool:
    if col == COL_Y or col in pop_cols or col in load_cols:
        return True
    if pd.api.types.is_bool_dtype(series):
        return True
    s_num = pd.to_numeric(series, errors="coerce")
    if s_num.notna().sum() > 0:
        return True
    return False


def bivariate_aggregate(
    frame: pd.DataFrame,
    col_x: str,
    col_y: str,
    pop_cols: list[str],
    load_cols: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = frame[[col_x, col_y]].copy()
    work["_x"] = make_x_categories(work[col_x]).map(lambda v: display_value(col_x, v))
    work[col_y] = work[col_y].map(normalize_param_name)
    use_mean = y_is_numeric_or_binary(col_y, work[col_y], pop_cols, load_cols)

    if use_mean:
        if col_y in pop_cols or col_y in load_cols:
            y_num = to_bool_series(work[col_y]).astype(float)
        else:
            y_num = pd.to_numeric(work[col_y], errors="coerce")
        g = pd.DataFrame({"_x": work["_x"], "_y": y_num}).dropna(subset=["_y"])
        g = g.groupby("_x", as_index=False)["_y"].mean()
        meta = {
            "title": f"{col_x} \\ {col_y}",
            "x": "_x",
            "y": "_y",
            "color": None,
            "y_label": "ממוצע",
            "x_label": col_x,
        }
        return g, meta

    g2 = pd.DataFrame({"_x": work["_x"], "_hue": work[col_y].fillna("חסר").astype(str)})
    g2["_hue"] = g2["_hue"].map(lambda v: display_value(col_y, v))
    g2 = g2.groupby(["_x", "_hue"], as_index=False).size().rename(columns={"size": "_count"})
    meta2 = {
        "title": f"{col_x} \\ {col_y}",
        "x": "_x",
        "y": "_count",
        "color": "_hue",
        "y_label": "כמות",
        "x_label": col_x,
        "legend": col_y,
    }
    return g2, meta2


def main() -> None:
    st.set_page_config(
        page_title="Survey EDA — עומס רווחה",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("סקר עומס")

    with st.sidebar:
        st.subheader("מקרא")
        st.markdown("**סוג הרשות**")
        st.dataframe(
            prepare_display(legend_table_from_mapping(AUTHORITY_ORDER, "מספר בקובץ")),
            hide_index=True,
            use_container_width=True,
        )
        st.markdown("**אשכול חברתי-כלכלי**")
        st.dataframe(
            prepare_display(legend_table_from_mapping(CLUSTER_ORDER, "מספר בקובץ")),
            hide_index=True,
            use_container_width=True,
        )
        st.markdown("**פער מול תקינה**")
        st.dataframe(
            prepare_display(legend_table_from_mapping(GAP_ORDER, "מספר בקובץ")),
            hide_index=True,
            use_container_width=True,
        )

        st.divider()
        st.subheader("קובץ")
        default_p = _default_data_path()
        path_input = st.text_input(
            "מיקום קובץ הנתונים (Excel)",
            value=str(default_p),
            help="ברירת מחדל",
        )
        try:
            df = load_survey(path_input)
        except FileNotFoundError as e:
            st.error(str(e))
            st.info("הרץ את `encoding.ipynb` ליצירת survey_encoded.xlsx, או בחר נתיב אחר.")
            st.stop()
        except Exception as e:
            st.exception(e)
            st.stop()

    if COL_PARAM in df.columns:
        df[COL_PARAM] = df[COL_PARAM].map(normalize_param_name)

    pop_cols = ohe_columns(df, PREFIX_POPULATION)
    load_cols = ohe_columns(df, PREFIX_LOAD)
    y_numeric = pd.to_numeric(df[COL_Y], errors="coerce") if COL_Y in df.columns else pd.Series(dtype=float)

    tab_overview, tab_y, tab_context, tab_pair, tab_ohe, tab_corr = st.tabs(
        ["סקירה", "חשיבות", "הקשר משרד", "השוואת שני שדות", "מקודדות", "מתאמים"]
    )

    with tab_overview:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("שורות", f"{len(df):,}")
        c2.metric("עמודות", len(df.columns))
        n_params = df[COL_PARAM].nunique() if COL_PARAM in df.columns else 0
        c3.metric("משתנים", n_params)
        c4.metric("עמודות OHE", len(pop_cols) + len(load_cols))

        st.subheader("דוגמה")
        st.dataframe(prepare_display(df.head(50)), use_container_width=True, height=400, hide_index=True)

        st.subheader("חוסרים")
        miss_df = (
            df.isna()
            .mean()
            .sort_values(ascending=False)
            .rename_axis("עמודה")
            .reset_index(name="שיעור חוסר")
        )
        fig_miss = px.bar(
            miss_df.head(40),
            x="שיעור חוסר",
            y="עמודה",
            orientation="h",
            title="חוסרים",
        )
        apply_axis_style(fig_miss, "שיעור חוסר", "עמודה")
        show_plot(fig_miss, "missing_values")
        download_csv_button(miss_df, "missing_values", "dl_missing")

        with st.expander("dtypes"):
            st.dataframe(
                prepare_display(pd.DataFrame({"עמודה": df.columns, "סוג": [str(d) for d in df.dtypes]})),
                use_container_width=True,
                hide_index=True,
            )

    with tab_y:
        if COL_Y not in df.columns or COL_PARAM not in df.columns:
            st.warning("חסרות עמודות")
        else:
            bad_y = y_numeric.isna() & df[COL_Y].notna()
            if bad_y.any():
                st.warning(f"שורות לא מספריות: {bad_y.sum()}")

            y_df = df.assign(_y=y_numeric).dropna(subset=["_y"])
            nu = int(y_df["_y"].nunique())
            nb = min(30, max(10, nu)) if nu > 0 else 10
            fig_hist = px.histogram(
                y_df,
                x="_y",
                nbins=nb,
                title="התפלגות חשיבות",
            )
            apply_axis_style(fig_hist, "ציון", "כמות")
            show_plot(fig_hist, "importance_hist")
            download_csv_button(y_df[["_y"]].rename(columns={"_y": "חשיבות"}), "importance_hist_data", "dl_hist")

            agg = (
                df.assign(_y=y_numeric)
                .dropna(subset=["_y"])
                .groupby(COL_PARAM, as_index=False)
                .agg(
                    ממוצע=("_y", "mean"),
                    סטיית_תקן=("_y", "std"),
                    n=("_y", "count"),
                )
                .sort_values("ממוצע", ascending=False)
            )
            st.subheader("ממוצע לפי משתנה")
            st.dataframe(prepare_display(agg), use_container_width=True, height=min(400, 35 * len(agg)), hide_index=True)

            short_labels = agg[COL_PARAM].astype(str).str.slice(0, 60)
            fig_bar = px.bar(
                agg.assign(_lab=short_labels),
                x="_lab",
                y="ממוצע",
                error_y="סטיית_תקן",
                title="משתנה \\ חשיבות",
            )
            fig_bar.update_xaxes(tickangle=-45)
            apply_axis_style(fig_bar, "משתנה", "ממוצע")
            show_plot(fig_bar, "importance_by_variable")
            download_csv_button(agg, "importance_by_variable", "dl_imp_var")

    with tab_context:
        ctx = [c for c in OFFICE_NUMERIC_CONTEXT + OFFICE_BINARY_CONTEXT if c in df.columns]
        if not ctx:
            st.info("אין עמודות משרד")
        elif COL_Y not in df.columns or COL_PARAM not in df.columns:
            st.warning("חסרות עמודות")
        else:

            plot_df = df.assign(_y=y_numeric).dropna(subset=["_y"])
            x_choice = st.selectbox("X", options=ctx, index=0)

            bar_ctx = (
                plot_df.groupby([x_choice, COL_PARAM], dropna=False)["_y"]
                .mean()
                .reset_index()
                .sort_values(x_choice)
            )
            bar_ctx[x_choice] = bar_ctx[x_choice].map(lambda v: display_value(x_choice, v))
            fig_bar_ctx = px.bar(
                bar_ctx,
                x=x_choice,
                y="_y",
                color=COL_PARAM,
                barmode="group",
                title=f"{x_choice} \\ {COL_PARAM}",
            )
            fig_bar_ctx.update_layout(legend_title_text=COL_PARAM)
            fig_bar_ctx.update_xaxes(tickangle=-25)
            apply_axis_style(fig_bar_ctx, x_choice, "ממוצע")
            show_plot(fig_bar_ctx, "office_context")
            download_csv_button(bar_ctx, "office_context", "dl_office")

    with tab_pair:
        all_cols = [c for c in df.columns if isinstance(c, str)]
        c1 = st.selectbox("X", options=all_cols, index=0, key="pair_a")
        second_options = [c for c in all_cols if c != c1] or all_cols
        c2_default = 1 if len(second_options) > 1 else 0
        c2 = st.selectbox(
            "Y",
            options=second_options,
            index=min(c2_default, len(second_options) - 1),
            key="pair_b",
        )
        if c1 == c2:
            st.warning("נא לבחור שני שדות שונים.")
        else:
            g, kw = bivariate_aggregate(df, c1, c2, pop_cols, load_cols)
            if g.empty:
                st.info("אין נתונים מספיקים לגרף.")
            else:
                if kw["color"] is not None:
                    fig_p = px.bar(g, x=kw["x"], y=kw["y"], color=kw["color"], barmode="group", title=kw["title"])
                    fig_p.update_layout(legend_title_text=str(kw.get("legend", "")))
                else:
                    fig_p = px.bar(g, x=kw["x"], y=kw["y"], title=kw["title"])
                fig_p.update_xaxes(tickangle=-35)
                apply_axis_style(fig_p, kw["x_label"], kw["y_label"])
                show_plot(fig_p, "pair_plot")
                download_csv_button(g, "pair_plot_data", "dl_pair")

    with tab_ohe:
        st.markdown("עמודות כן/לא")
        u = unique_office_frame(df)
        st.metric("פרופילים", len(u))

        def counts_from_dummies(frame: pd.DataFrame, cols: list[str], label: str) -> pd.DataFrame:
            if not cols:
                return pd.DataFrame()
            sums = []
            for c in cols:
                s = to_bool_series(frame[c])
                sums.append((c, int(s.sum())))
            out = pd.DataFrame(sums, columns=["עמודה", "ספירה"])
            pfx = label if label.endswith("_") else f"{label}_"
            out["תווית"] = out["עמודה"].str.replace(pfx, "", regex=False)
            return out.sort_values("ספירה", ascending=False)

        if pop_cols:
            st.subheader("אוכלוסייה")
            cp = counts_from_dummies(u, pop_cols, PREFIX_POPULATION.rstrip("_"))
            fig_p = px.bar(
                cp,
                x="תווית",
                y="ספירה",
                title="אוכלוסייה",
            )
            fig_p.update_xaxes(tickangle=-40)
            apply_axis_style(fig_p, "קטגוריה", "משרדים")
            show_plot(fig_p, "population_ohe")
            download_csv_button(cp, "population_ohe", "dl_pop_ohe")

        if load_cols:
            st.subheader("תחומי עומס")
            cl = counts_from_dummies(u, load_cols, PREFIX_LOAD.rstrip("_"))
            fig_l = px.bar(
                cl,
                x="תווית",
                y="ספירה",
                title="תחומי עומס",
            )
            fig_l.update_xaxes(tickangle=-40)
            apply_axis_style(fig_l, "קטגוריה", "משרדים")
            show_plot(fig_l, "load_ohe")
            download_csv_button(cl, "load_ohe", "dl_load_ohe")

        with st.expander("טבלת פרופילים", expanded=False):
            show_cols = [c for c in OFFICE_NUMERIC_CONTEXT if c in u.columns] + pop_cols + load_cols
            show_cols = [c for c in show_cols if c in u.columns]
            profiles = prepare_display(u[show_cols])
            st.dataframe(profiles, use_container_width=True, height=360, hide_index=True)
            download_csv_button(profiles, "office_profiles", "dl_profiles")

    with tab_corr:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for c in pop_cols + load_cols:
            if c in df.columns and c not in num_cols:
                num_cols.append(c)
        num_cols = list(dict.fromkeys(num_cols))
        focus = st.checkbox(
            "מיקוד",
            value=True,
        )
        if focus and COL_Y in df.columns:
            extra = [c for c in OFFICE_NUMERIC_CONTEXT if c in df.columns] + pop_cols + load_cols + [COL_Y]
            picked = [c for c in extra if c in df.columns]
            corr_df_full = df[picked].copy()
            corr_df_full[Y_NUM_ALIAS] = pd.to_numeric(corr_df_full[COL_Y], errors="coerce")
            for c in pop_cols + load_cols:
                if c in corr_df_full.columns:
                    corr_df_full[c] = to_bool_series(corr_df_full[c]).astype(float)
            for c in OFFICE_NUMERIC_CONTEXT:
                if c in corr_df_full.columns:
                    corr_df_full[c] = pd.to_numeric(corr_df_full[c], errors="coerce")
            use_cols = [c for c in corr_df_full.columns if c != COL_Y] + [Y_NUM_ALIAS]
            corr_df = corr_df_full[use_cols].copy()
        elif len(num_cols) >= 2:
            corr_df = df[num_cols].copy()
            for c in pop_cols + load_cols:
                if c in corr_df.columns:
                    corr_df[c] = to_bool_series(corr_df[c]).astype(float)
            corr_df = corr_df.apply(pd.to_numeric, errors="coerce")
        else:
            corr_df = pd.DataFrame()

        if corr_df.shape[1] < 2:
            st.info("אין מספיק עמודות")
        else:
            cmat = corr_df.corr(numeric_only=True)
            if Y_NUM_ALIAS in cmat.columns:
                nice_y = "חשיבות"
                cmat = cmat.rename(index={Y_NUM_ALIAS: nice_y}, columns={Y_NUM_ALIAS: nice_y})
            fig_c = go.Figure(
                data=go.Heatmap(
                    z=cmat.values,
                    x=cmat.columns,
                    y=cmat.columns,
                    zmin=-1,
                    zmax=1,
                    colorscale="RdBu_r",
                )
            )
            fig_c.update_layout(
                title="מפת מתאם",
                height=max(500, len(cmat.columns) * 14),
                xaxis_title="",
                yaxis_title="",
            )
            show_plot(fig_c, "correlation")
            download_csv_button(cmat.reset_index().rename(columns={"index": "row"}), "correlation_matrix", "dl_corr")

    st.divider()
    st.caption("survey_encoded.xlsx")


if __name__ == "__main__":
    main()
