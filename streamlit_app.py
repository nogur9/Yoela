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
Y_NUM_ALIAS = "__y_num__"

AUTHORITY_ORDER: dict[str, int | None] = {
    "עיר גדולה (מעל 200 אלף תושבים)": 4,
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
    return Path(__file__).resolve().parent / "survey_encoded.xlsx"


@st.cache_data(show_spinner=False)
def load_survey(path_str: str) -> pd.DataFrame:
    path = Path(path_str).expanduser()
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
    t = pd.DataFrame(rows, columns=[value_label, "תיאור מקורי"])
    return t.sort_values(value_label, na_position="last")


def column_is_categorical(col: str, series: pd.Series, pop_cols: list[str], load_cols: list[str]) -> bool:
    if col == COL_Y:
        return False
    if col in pop_cols or col in load_cols:
        return True
    if col in OFFICE_NUMERIC_CONTEXT or col == COL_PARAM:
        return True
    if pd.api.types.is_bool_dtype(series):
        return True
    if pd.api.types.is_object_dtype(series):
        return True
    if isinstance(series.dtype, pd.StringDtype):
        return True
    if isinstance(series.dtype, pd.CategoricalDtype):
        return True
    return False


def column_is_numeric_for_agg(col: str, series: pd.Series, pop_cols: list[str], load_cols: list[str]) -> bool:
    if col in pop_cols or col in load_cols:
        return False
    if col in OFFICE_NUMERIC_CONTEXT or col == COL_PARAM:
        return False
    if pd.api.types.is_bool_dtype(series):
        return False
    if col == COL_Y:
        return True
    return pd.api.types.is_numeric_dtype(series)


def bivariate_aggregate(
    frame: pd.DataFrame,
    col_a: str,
    col_b: str,
    pop_cols: list[str],
    load_cols: list[str],
) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    a = frame[col_a]
    b = frame[col_b]
    cat_a = column_is_categorical(col_a, a, pop_cols, load_cols)
    cat_b = column_is_categorical(col_b, b, pop_cols, load_cols)
    num_a = column_is_numeric_for_agg(col_a, a, pop_cols, load_cols)
    num_b = column_is_numeric_for_agg(col_b, b, pop_cols, load_cols)

    work = frame[[col_a, col_b]].copy()
    work = work.dropna(how="all")
    if col_a in work.columns and col_b in work.columns:
        if col_a == COL_Y:
            work[col_a] = pd.to_numeric(work[col_a], errors="coerce")
        if col_b == COL_Y:
            work[col_b] = pd.to_numeric(work[col_b], errors="coerce")

    kwargs: dict[str, Any] = {}

    if cat_a and cat_b:
        g = work.groupby([col_a, col_b], dropna=False).size().reset_index(name="ספירה")
        title = f"ספירת שורות לפי «{col_a}» ו-«{col_b}»"
        kwargs = dict(x=col_a, y="ספירה", color=col_b, barmode="group")
        return g, title, kwargs

    if cat_a and num_b:
        g = work.dropna(subset=[col_b]).groupby(col_a, dropna=False)[col_b].mean().reset_index()
        title = f"ממוצע «{col_b}» לפי «{col_a}»"
        kwargs = dict(x=col_a, y=col_b, color=None, barmode="relative")
        return g, title, kwargs

    if num_a and cat_b:
        g = work.dropna(subset=[col_a]).groupby(col_b, dropna=False)[col_a].mean().reset_index()
        title = f"ממוצע «{col_a}» לפי «{col_b}»"
        kwargs = dict(x=col_b, y=col_a, color=None, barmode="relative")
        return g, title, kwargs

    work_num = work.dropna()
    if work_num.empty:
        return work_num, "אין נתונים", {}
    try:
        work_num = work_num.assign(
            _bin=pd.qcut(work_num[col_a], q=5, duplicates="drop"),
        )
    except ValueError:
        work_num = work_num.assign(_bin=pd.cut(work_num[col_a], bins=min(5, work_num[col_a].nunique())))
    g = work_num.groupby("_bin", observed=True)[col_b].mean().reset_index()
    g["_bin"] = g["_bin"].astype(str)
    title = f"ממוצע «{col_b}» לפי חלוקת «{col_a}» לקבוצות (שני משתנים מספריים)"
    kwargs = dict(x="_bin", y=col_b, color=None, barmode="relative")
    return g, title, kwargs


def main() -> None:
    st.set_page_config(
        page_title="Survey EDA — עומס רווחה",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("ניתוח נתוני סקר — עומס בתחום הרווחה")
    st.caption(
        "כל שורה = ציון שמנהל/ת משרד רווחה נתן/ה למשתנה אפשרי אחד. "
        "עמודות הקשר (סוג רשות, אשכול, וכו') ועמודות ה-one-hot חוזרות על עצמן לכל משתנה באותו משרד."
    )

    with st.sidebar:
        st.subheader("מקרא — קידוד אורדינלי לפני הניתוח")
        st.caption(
            "הערכים בקובץ המקודד כבר ממופים; המקרא מציג את התאמת הקוד לתוויות המקוריות."
        )
        st.markdown("**סוג הרשות**")
        st.dataframe(
            prepare_display(legend_table_from_mapping(AUTHORITY_ORDER, "קוד")),
            hide_index=True,
            use_container_width=True,
        )
        st.markdown("**אשכול חברתי-כלכלי**")
        st.dataframe(
            prepare_display(legend_table_from_mapping(CLUSTER_ORDER, "קוד")),
            hide_index=True,
            use_container_width=True,
        )
        st.markdown("**פער מול תקינה**")
        st.dataframe(
            prepare_display(legend_table_from_mapping(GAP_ORDER, "קוד")),
            hide_index=True,
            use_container_width=True,
        )

        st.divider()
        st.subheader("קובץ נתונים")
        default_p = _default_data_path()
        path_input = st.text_input(
            "נתיב לקובץ Excel",
            value=str(default_p),
            help="ברירת מחדל: survey_encoded.xlsx בתיקיית האפליקציה",
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

    pop_cols = ohe_columns(df, PREFIX_POPULATION)
    load_cols = ohe_columns(df, PREFIX_LOAD)
    y_numeric = pd.to_numeric(df[COL_Y], errors="coerce") if COL_Y in df.columns else pd.Series(dtype=float)

    tab_overview, tab_y, tab_context, tab_pair, tab_ohe, tab_corr = st.tabs(
        ["סקירה", f"משתנה תלוי ({LABEL_IMPORTANCE})", "הקשר משרד", "שני פרמטרים", "קטגוריות מקודדות", "מתאמים"]
    )

    with tab_overview:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("שורות", f"{len(df):,}")
        c2.metric("עמודות", len(df.columns))
        n_params = df[COL_PARAM].nunique() if COL_PARAM in df.columns else 0
        c3.metric("משתנים (פרמטרים) שונים", n_params)
        c4.metric("עמודות one-hot (אוכלוסייה / עומס)", len(pop_cols) + len(load_cols))

        st.subheader("תצוגה מקדימה")
        st.dataframe(prepare_display(df.head(50)), use_container_width=True, height=400, hide_index=True)

        st.subheader("חוסרים וסוגי נתונים")
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
            title="Top 40 עמודות לפי שיעור ערכים חסרים",
        )
        st.plotly_chart(fig_miss, use_container_width=True)

        with st.expander("סוגי עמודות (dtype)"):
            st.dataframe(
                prepare_display(pd.DataFrame({"עמודה": df.columns, "dtype": [str(d) for d in df.dtypes]})),
                use_container_width=True,
                hide_index=True,
            )

    with tab_y:
        if COL_Y not in df.columns or COL_PARAM not in df.columns:
            st.warning("חסרות עמודות 'ערך משתנה' או 'משתנה'.")
        else:
            bad_y = y_numeric.isna() & df[COL_Y].notna()
            if bad_y.any():
                st.warning(
                    f"{bad_y.sum()} שורות עם ערך לא מספרי בעמודת «{COL_Y}» — לא יוצגו בחלק מהגרפים."
                )

            y_df = df.assign(_y=y_numeric).dropna(subset=["_y"])
            nu = int(y_df["_y"].nunique())
            nb = min(30, max(10, nu)) if nu > 0 else 10
            fig_hist = px.histogram(
                y_df,
                x="_y",
                nbins=nb,
                title=f"התפלגות {LABEL_IMPORTANCE} ({COL_Y})",
            )
            fig_hist.update_xaxes(title_text=f"{LABEL_IMPORTANCE} ({COL_Y})")
            st.plotly_chart(fig_hist, use_container_width=True)

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
            st.subheader(f"ממוצע {LABEL_IMPORTANCE} לפי שם משתנה")
            st.dataframe(prepare_display(agg), use_container_width=True, height=min(400, 35 * len(agg)), hide_index=True)

            short_labels = agg[COL_PARAM].astype(str).str.slice(0, 60)
            fig_bar = px.bar(
                agg.assign(_lab=short_labels),
                x="_lab",
                y="ממוצע",
                error_y="סטיית_תקן",
                title=f"ממוצע {LABEL_IMPORTANCE} לפי פרמטר (± סטיית תקן)",
            )
            fig_bar.update_xaxes(tickangle=-45)
            fig_bar.update_yaxes(title_text=LABEL_IMPORTANCE)
            st.plotly_chart(fig_bar, use_container_width=True)

            pick = st.multiselect(
                "בחר משתנים להשוואת התפלגות",
                options=sorted(df[COL_PARAM].dropna().unique().astype(str)),
                default=list(sorted(df[COL_PARAM].dropna().unique().astype(str))[: min(5, df[COL_PARAM].nunique())]),
            )
            if pick:
                sub = df[df[COL_PARAM].astype(str).isin(pick)].assign(_y=y_numeric).dropna(subset=["_y"])
                fig_v = px.box(
                    sub,
                    x=COL_PARAM,
                    y="_y",
                    title=f"{LABEL_IMPORTANCE} לפי פרמטר (נבחרים)",
                )
                fig_v.update_yaxes(title_text=LABEL_IMPORTANCE)
                fig_v.update_xaxes(tickangle=-35)
                st.plotly_chart(fig_v, use_container_width=True)

    with tab_context:
        ctx = [c for c in OFFICE_NUMERIC_CONTEXT if c in df.columns]
        if not ctx:
            st.info("לא נמצאו עמודות הקשר מספריות מוכרות.")
        elif COL_Y not in df.columns or COL_PARAM not in df.columns:
            st.warning("חסרות עמודות נדרשות לוויזואליזציה.")
        else:
            plot_df = df.assign(_y=y_numeric).dropna(subset=["_y"])
            x_choice = st.selectbox("ציר X (מאפיין משרד)", options=ctx, index=0)
            st.caption(f"צבע לפי «{COL_PARAM}» (קבוע) — ממוצע {LABEL_IMPORTANCE} לכל צירוף.")

            bar_ctx = (
                plot_df.groupby([x_choice, COL_PARAM], dropna=False)["_y"]
                .mean()
                .reset_index()
                .sort_values(x_choice)
            )
            fig_bar_ctx = px.bar(
                bar_ctx,
                x=x_choice,
                y="_y",
                color=COL_PARAM,
                barmode="group",
                title=f"ממוצע {LABEL_IMPORTANCE} לפי {x_choice} ו-{COL_PARAM}",
            )
            fig_bar_ctx.update_layout(legend_title_text=COL_PARAM)
            fig_bar_ctx.update_yaxes(title_text=f"ממוצע {LABEL_IMPORTANCE}")
            fig_bar_ctx.update_xaxes(title_text=x_choice)
            st.plotly_chart(fig_bar_ctx, use_container_width=True)

            st.subheader(f"ממוצע {LABEL_IMPORTANCE} לפי ערך במאפיין משרד (ללא פיצול לפי משתנה)")
            g_choice = st.selectbox("קבץ לפי", options=ctx, key="grp_ctx")
            gmean = (
                plot_df.groupby(g_choice, as_index=False)["_y"]
                .mean()
                .sort_values("_y", ascending=False)
            )
            st.dataframe(prepare_display(gmean), use_container_width=True, hide_index=True)
            fig_g = px.bar(
                gmean,
                x=g_choice,
                y="_y",
                title=f"ממוצע {LABEL_IMPORTANCE} לפי {g_choice}",
            )
            fig_g.update_yaxes(title_text=f"ממוצע {LABEL_IMPORTANCE}")
            st.plotly_chart(fig_g, use_container_width=True)

    with tab_pair:
        st.markdown(
            "בחרו שני שדות מהטבלה. **שניהם קטגוריאליים** (כולל קידוד אורדינלי ועמודות one-hot): "
            "הגרף מציג **ספירת שורות** לכל צירוף. "
            "אם הפרמטר השני **מספרי** (למשל ערך משתנה): מוצג **ממוצע** שלו לפי קבוצות הפרמטר הראשון. "
            "שני משתנים מספריים: ממוצע השני לפי חלוקת הראשון לרבעונים."
        )
        all_cols = [c for c in df.columns if isinstance(c, str)]
        c1 = st.selectbox("פרמטר ראשון", options=all_cols, index=0, key="pair_a")
        second_options = [c for c in all_cols if c != c1] or all_cols
        c2_default = 1 if len(second_options) > 1 else 0
        c2 = st.selectbox(
            "פרמטר שני",
            options=second_options,
            index=min(c2_default, len(second_options) - 1),
            key="pair_b",
        )
        if c1 == c2:
            st.warning("יש לבחור שני שדות שונים.")
        else:
            g, title, kw = bivariate_aggregate(df, c1, c2, pop_cols, load_cols)
            if g.empty:
                st.info("אין נתונים להצגה.")
            elif not kw:
                st.info(title)
            else:
                color_kw: dict[str, Any] = {}
                if kw.get("color"):
                    color_kw["color"] = kw["color"]
                fig_p = px.bar(
                    g,
                    x=kw["x"],
                    y=kw["y"],
                    barmode=kw.get("barmode", "group"),
                    title=title,
                    **color_kw,
                )
                fig_p.update_xaxes(tickangle=-35)
                st.plotly_chart(fig_p, use_container_width=True)
                with st.expander("טבלת צבירה"):
                    st.dataframe(prepare_display(g), use_container_width=True, hide_index=True)

    with tab_ohe:
        st.markdown(
            f"עמודות **{PREFIX_POPULATION.rstrip('_')}** ו-**{PREFIX_LOAD.rstrip('_')}** "
            "פוצלו ל-one-hot. להלן פרופילי משרדים ייחודיים (לפי שילוב עמודות אלה והמאפיינים המספריים)."
        )
        u = unique_office_frame(df)
        st.metric("משרדים / פרופילים ייחודיים (משוער)", len(u))

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
            st.subheader("מאפייני אוכלוסייה (מספר פרופילי משרד עם כל קטגוריה)")
            cp = counts_from_dummies(u, pop_cols, PREFIX_POPULATION.rstrip("_"))
            fig_p = px.bar(cp, x="תווית", y="ספירה", title="תדירות קטגוריה (ברמת משרד ייחודי)")
            fig_p.update_xaxes(tickangle=-40)
            st.plotly_chart(fig_p, use_container_width=True)

        if load_cols:
            st.subheader("תחומי העומס")
            cl = counts_from_dummies(u, load_cols, PREFIX_LOAD.rstrip("_"))
            fig_l = px.bar(cl, x="תווית", y="ספירה", title="תדירות שילוב תחומי עומס (ברמת משרד ייחודי)")
            fig_l.update_xaxes(tickangle=-40)
            st.plotly_chart(fig_l, use_container_width=True)

        with st.expander("טבלת פרופילי משרד (שורה לכל שילוב ייחודי)"):
            show_cols = [c for c in OFFICE_NUMERIC_CONTEXT if c in u.columns] + pop_cols + load_cols
            show_cols = [c for c in show_cols if c in u.columns]
            st.dataframe(prepare_display(u[show_cols]), use_container_width=True, height=360, hide_index=True)

    with tab_corr:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for c in pop_cols + load_cols:
            if c in df.columns and c not in num_cols:
                num_cols.append(c)
        num_cols = list(dict.fromkeys(num_cols))
        focus = st.checkbox(
            f"התמקד ב-{LABEL_IMPORTANCE} + הקשר משרד + one-hot (מומלץ לסקרים צפופים)",
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
            st.info("אין מספיק עמודות למפת חום.")
        else:
            cmat = corr_df.corr(numeric_only=True)
            if Y_NUM_ALIAS in cmat.columns:
                nice_y = f"{LABEL_IMPORTANCE} ({COL_Y}, מספרי)"
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
                title="מטריצת מתאמים (מספרי + דמה-בינארי ל-one-hot)",
                height=max(500, len(cmat.columns) * 14),
            )
            st.plotly_chart(fig_c, use_container_width=True)

    st.divider()
    st.caption("Streamlit EDA · survey_encoded.xlsx")


if __name__ == "__main__":
    main()
