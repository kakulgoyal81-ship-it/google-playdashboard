"""Google Play Store hexbin dashboard (Dash + Plotly).

Run:  python app.py   ->  http://127.0.0.1:8050

The dashboard is visible only between 5:00 PM and 7:00 PM IST
(ZoneInfo("Asia/Kolkata")). The clock and the time gate refresh
automatically every second, and the page auto-reloads every 5 minutes
while the dashboard is open.
"""

import os
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import html
from plotly.subplots import make_subplots

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_A = os.path.join(BASE_DIR, "data_apps.csv")
FILE_B = os.path.join(BASE_DIR, "data_reviews.csv")

ALLOWED_CATEGORIES = ["Game", "Beauty", "Business", "Comics", "Communication",
                      "Dating", "Entertainment", "Social", "Events"]

DISPLAY_MAP = {
    "beauty": "सुंदरता",
    "business": "வணிகம்",
    "dating": "Dating",
}

RATING_MIN = 3.5
INSTALLS_MIN = 50_000
REVIEWS_MIN = 500
SIZE_MIN, SIZE_MAX = 10.0, 100.0
SUBJECTIVITY_MIN = 0.5
RATING_MAX = 5.0

IST = ZoneInfo("Asia/Kolkata")
OPEN_HOUR, CLOSE_HOUR = 17, 19
RELOAD_MS = 300_000
GAME_COLOR = "#FF69B4"


def norm_col(c):
    return re.sub(r"[^a-z0-9]", "", str(c).strip().lower())


def read_csv(path):
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path, encoding="utf-8", engine="python")


def detect_inputs():
    store, store_path, reviews, reviews_path = None, None, None, None
    for path in (FILE_A, FILE_B):
        if not os.path.exists(path):
            continue
        df = read_csv(path)
        cols = {norm_col(c) for c in df.columns}
        if {"category", "size", "installs"} <= cols and store is None:
            store, store_path = df, path
        elif "sentimentsubjectivity" in cols and reviews is None:
            reviews, reviews_path = df, path
    return store, store_path, reviews, reviews_path


def size_to_mb(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower().replace(",", "")
    if s in ("varies with device", "varies", "nan", "none", ""):
        return np.nan
    try:
        if s.endswith("m"):
            return float(s[:-1])
        if s.endswith("k"):
            return float(s[:-1]) / 1024.0
        if s.endswith("g"):
            return float(s[:-1]) * 1024.0
        return float(s)
    except Exception:
        return np.nan


def numeric_series(s):
    return pd.to_numeric(
        s.astype(str).str.replace(",", "", regex=False)
        .str.replace("+", "", regex=False).str.strip(), errors="coerce")


def display_category(cat):
    key = str(cat).strip().lower()
    if key in DISPLAY_MAP:
        return DISPLAY_MAP[key]
    return str(cat).strip().title()


def build_dataset():
    store, store_path, reviews, reviews_path = detect_inputs()
    if store is None:
        raise FileNotFoundError(
            "Could not find the store data (columns: Category, Size, "
            "Installs) in data_apps.csv / data_reviews.csv")

    df = store.copy()
    df["_app_str"] = df["App"].astype(str).str.strip()
    df["_cat_str"] = df["Category"].astype(str).str.strip()
    df["_rating_num"] = pd.to_numeric(df["Rating"], errors="coerce")
    df["_reviews_num"] = numeric_series(df["Reviews"])
    df["_size_mb"] = df["Size"].apply(size_to_mb)
    df["_installs_num"] = numeric_series(df["Installs"])

    if reviews is not None:
        rv = reviews.copy()
        key_col = "App" if "App" in rv.columns else rv.columns[0]
        subj_col = None
        for c in rv.columns:
            if "subjectiv" in norm_col(c):
                subj_col = c
                break
        if subj_col is not None:
            rv["_key"] = rv[key_col].astype(str).str.strip().str.lower()
            rv[subj_col] = pd.to_numeric(rv[subj_col], errors="coerce")
            agg = (rv.groupby("_key")[subj_col].mean()
                   .reset_index()
                   .rename(columns={subj_col: "_subj_num"}))
            df["_key"] = df["_app_str"].str.lower()
            df = df.merge(agg, on="_key", how="left")
            df.drop(columns=["_key"], inplace=True)
        else:
            df["_subj_num"] = np.nan
    else:
        df["_subj_num"] = np.nan

    n_raw = len(df)
    allowed_lower = {c.lower() for c in ALLOWED_CATEGORIES}
    df = df[df["_cat_str"].str.lower().isin(allowed_lower)].copy()
    n_cat = len(df)
    mask = ((df["_rating_num"] > RATING_MIN)
            & (df["_installs_num"] > INSTALLS_MIN)
            & (df["_reviews_num"] > REVIEWS_MIN)
            & (df["_size_mb"] >= SIZE_MIN)
            & (df["_size_mb"] <= SIZE_MAX)
            & (df["_subj_num"] > SUBJECTIVITY_MIN))
    df = df[mask].copy()
    n_num = len(df)
    df = df[~df["_app_str"].str.contains("s", case=False, na=False)].copy()
    df["_category_display"] = df["_cat_str"].apply(display_category)

    counts = {"raw": n_raw, "after_category": n_cat,
              "after_numeric": n_num, "final": len(df)}

    iqr_stats, outlier_frames = [], []
    for cat, g in df.groupby("_cat_str"):
        vals = g["_installs_num"]
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        flags = (vals < lower) | (vals > upper)
        iqr_stats.append({
            "Category": display_category(cat), "Apps": len(g),
            "Q1": q1, "Q3": q3, "IQR": iqr,
            "Lower": lower, "Upper": upper,
            "Outliers": int(flags.sum()),
        })
        out = g[flags].copy()
        if not out.empty:
            out["_out_reason"] = np.where(
                out["_installs_num"] > upper,
                f"above upper bound ({upper:,.0f})",
                f"below lower bound ({lower:,.0f})")
            outlier_frames.append(out)
    iqr_stats_df = pd.DataFrame(iqr_stats).sort_values("Category")
    outliers_df = (pd.concat(outlier_frames, ignore_index=True)
                   if outlier_frames else df.iloc[0:0].copy())

    game_df = df[df["_cat_str"].str.lower() == "game"].copy()

    return {
        "df": df, "counts": counts, "iqr_stats": iqr_stats_df,
        "outliers": outliers_df, "game": game_df,
        "store_file": os.path.basename(os.path.abspath(store_path)),
        "reviews_file": (os.path.basename(os.path.abspath(reviews_path))
                         if reviews_path else "—"),
    }


def hexbin_aggregate(df, nx=18, ny=14,
                     x_range=(SIZE_MIN, SIZE_MAX),
                     y_range=(RATING_MIN, RATING_MAX)):
    dx = (x_range[1] - x_range[0]) / nx
    dy = (y_range[1] - y_range[0]) / ny
    sub = df[["_size_mb", "_rating_num", "_installs_num"]].dropna().copy()
    sub = sub[(sub["_size_mb"] >= x_range[0]) & (sub["_size_mb"] <= x_range[1])
              & (sub["_rating_num"] >= y_range[0]) & (sub["_rating_num"] <= y_range[1])]
    if sub.empty:
        return sub
    sub["_row"] = np.floor((sub["_rating_num"] - y_range[0]) / dy).astype(int).clip(0, ny - 1)
    x_shifted = sub["_size_mb"] - (sub["_row"] % 2) * (dx / 2.0)
    sub["_col"] = np.floor((x_shifted - x_range[0]) / dx).astype(int).clip(0, nx - 1)
    rows = []
    for (c, r), g in sub.groupby(["_col", "_row"]):
        cx = x_range[0] + c * dx + dx / 2.0 + (dx / 2.0 if r % 2 == 1 else 0.0)
        cy = y_range[0] + r * dy + dy / 2.0
        rows.append({
            "col": c, "row": r, "center_x": cx, "center_y": cy,
            "x0": max(cx - dx / 2, x_range[0]), "x1": min(cx + dx / 2, x_range[1]),
            "y0": max(cy - dy / 2, y_range[0]), "y1": min(cy + dy / 2, y_range[1]),
            "n_apps": len(g),
            "avg_installs": g["_installs_num"].mean(),
        })
    return pd.DataFrame(rows).sort_values(["row", "col"]).reset_index(drop=True)


def hex_pixel_size(nx, fig_width=1080):
    plot_px = fig_width * 0.784 - 95
    return float(np.clip((plot_px / nx) * 1.12, 14, 60))


def ist_now():
    return datetime.now(IST)


def is_open(now=None):
    now = now or ist_now()
    return True  # TEMPORARY TEST OVERRIDE - restore: return OPEN_HOUR <= now.hour < CLOSE_HOUR


def next_opening(now=None):
    now = now or ist_now()
    if now.hour < OPEN_HOUR:
        return now.replace(hour=OPEN_HOUR, minute=0, second=0, microsecond=0)
    return (now + timedelta(days=1)).replace(
        hour=OPEN_HOUR, minute=0, second=0, microsecond=0)


def gate_state(now=None):
    now = now or ist_now()
    clock = f"{now.strftime('%I:%M:%S %p').lstrip('0')} IST"
    if is_open(now):
        return {"open": True, "clock": clock}
    nxt = next_opening(now)
    hours, rem = divmod(int((nxt - now).total_seconds()), 3600)
    when = "Today" if nxt.date() == now.date() else "Tomorrow"
    return {
        "open": False,
        "clock": clock,
        "countdown": (f"Opens {when} at {nxt.strftime('%I:%M %p').lstrip('0')} IST "
                      f"(in {hours} h {rem // 60} min)"),
    }


def build_hexbin_figure(bins, filtered_df, game_df, outliers_df,
                        colorscale="YlOrRd", show_game=True, show_outliers=True,
                        nx=18, ny=14):
    fig = make_subplots(
        rows=2, cols=2,
        row_heights=[0.20, 0.80],
        column_widths=[0.80, 0.20],
        horizontal_spacing=0.015,
        vertical_spacing=0.04,
    )

    if bins is not None and not bins.empty:
        hover = [
            f"Size: {r.x0:.1f}–{r.x1:.1f} MB<br>"
            f"Rating: {r.y0:.2f}–{r.y1:.2f}<br>"
            f"Apps in hex: {int(r.n_apps)}<br>"
            f"Average Installs: {r.avg_installs:,.0f}"
            for r in bins.itertuples()
        ]
        fig.add_trace(go.Scatter(
            x=bins["center_x"], y=bins["center_y"],
            mode="markers", name="Hexbin (colour = avg installs)",
            text=hover, hoverinfo="text",
            marker=dict(
                symbol="hexagon", size=hex_pixel_size(nx),
                color=bins["avg_installs"], colorscale=colorscale,
                showscale=True,
                colorbar=dict(title="Average Installs", tickformat=",d",
                              orientation="h",
                              x=0.32, y=-0.155, len=0.42, xpad=0, ypad=0),
                line=dict(width=1, color="rgba(60,60,60,0.55)"), opacity=0.95),
        ), row=2, col=1)
    else:
        fig.add_annotation(
            text="No data in the hexbin range after filters.",
            xref="x3 domain", yref="y3 domain", x=0.5, y=0.5, showarrow=False,
            row=2, col=1)

    if show_outliers and outliers_df is not None and not outliers_df.empty:
        fig.add_trace(go.Scatter(
            x=outliers_df["_size_mb"], y=outliers_df["_rating_num"],
            mode="markers+text", name="IQR outlier (labeled)",
            text=outliers_df["_app_str"], textposition="top center",
            textfont=dict(size=8, color="#1a1a1a"),
            hovertemplate="<b>%{text}</b><br>Size: %{x:.2f} MB<br>Rating: %{y:.2f}<br>"
                          "Installs: %{customdata[0]:,.0f}<br>Category: %{customdata[1]}<extra></extra>",
            customdata=np.stack([outliers_df["_installs_num"].to_numpy(),
                                 outliers_df["_category_display"].to_numpy()], axis=-1),
            marker=dict(symbol="diamond", size=12, color="black",
                        line=dict(width=1.5, color="white")),
        ), row=2, col=1)

    if show_game and game_df is not None and not game_df.empty:
        fig.add_trace(go.Scatter(
            x=game_df["_size_mb"], y=game_df["_rating_num"],
            mode="markers", name="Game (pink)",
            hovertemplate="<b>%{text}</b><br>Size: %{x:.2f} MB<br>Rating: %{y:.2f}<br>"
                          "Installs: %{customdata[0]:,.0f}<br>Reviews: %{customdata[1]:,.0f}<extra></extra>",
            text=game_df["_app_str"],
            customdata=np.stack([game_df["_installs_num"].to_numpy(),
                                 game_df["_reviews_num"].to_numpy()], axis=-1),
            marker=dict(color=GAME_COLOR, size=9, opacity=0.9,
                        line=dict(width=1, color="white")),
        ), row=2, col=1)

    size_counts = np.array([0.0])
    rating_counts = np.array([0.0])
    if filtered_df is not None and not filtered_df.empty:
        size_counts, _ = np.histogram(filtered_df["_size_mb"].dropna(),
                                      bins=24, range=(SIZE_MIN, SIZE_MAX))
        rating_counts, _ = np.histogram(filtered_df["_rating_num"].dropna(),
                                        bins=15, range=(RATING_MIN, RATING_MAX))
        fig.add_trace(go.Histogram(
            x=filtered_df["_size_mb"],
            xbins=dict(start=SIZE_MIN, end=SIZE_MAX, size=(SIZE_MAX - SIZE_MIN) / 24),
            marker=dict(color="#8a9bb0"), name="Size distribution", showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Histogram(
            y=filtered_df["_rating_num"],
            ybins=dict(start=RATING_MIN, end=RATING_MAX, size=(RATING_MAX - RATING_MIN) / 15),
            marker=dict(color="#8a9bb0"), name="Rating distribution", showlegend=False,
        ), row=2, col=2)

    fig.update_xaxes(matches="x3", overlaying="x3", showticklabels=False,
                     showgrid=False, row=1, col=1)
    fig.update_yaxes(range=[0, float(size_counts.max()) * 1.25],
                     showticklabels=False, showgrid=False, row=1, col=1)
    fig.update_yaxes(matches="y3", overlaying="y3", showticklabels=False,
                     showgrid=False, row=2, col=2)
    fig.update_xaxes(range=[0, float(rating_counts.max()) * 1.25],
                     showticklabels=False, showgrid=False, row=2, col=2)
    fig.update_xaxes(visible=False, row=1, col=2)
    fig.update_yaxes(visible=False, row=1, col=2)
    fig.update_xaxes(title_text="App Size (MB)", range=[SIZE_MIN, SIZE_MAX],
                     row=2, col=1)
    fig.update_yaxes(title_text="Rating", range=[RATING_MIN, RATING_MAX],
                     row=2, col=1)

    fig.update_layout(
        title="App Size vs Rating — True Hexbin (hex colour = average Installs)",
        template="plotly_white", height=640, width=1080,
        margin=dict(l=70, r=30, t=70, b=150),
        legend=dict(orientation="h", y=-0.34, x=0.22),
        font=dict(family="Arial"),
    )
    return fig


def build_box_figure(filtered_df):
    fig = go.Figure()
    if filtered_df is None or filtered_df.empty:
        fig.add_annotation(text="No data for box plot.", xref="paper",
                           yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    for cat, g in sorted(filtered_df.groupby("_cat_str"),
                         key=lambda kv: display_category(kv[0]).lower()):
        fig.add_trace(go.Box(
            y=g["_installs_num"], name=display_category(cat),
            boxpoints="outliers", jitter=0.3,
            hovertemplate=f"{display_category(cat)}<br>Installs: %{{y:,.0f}}<extra></extra>",
        ))
    fig.update_layout(
        title="Installs distribution per category (log scale)",
        yaxis=dict(title="Installs", type="log", tickformat=",d"),
        template="plotly_white", height=430,
        margin=dict(l=70, r=20, t=60, b=90), font=dict(family="Arial"),
    )
    return fig


def fmt(v):
    if v is None:
        return "—"
    if isinstance(v, (bool, np.bool_)):
        return str(v)
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        if pd.isna(v):
            return "—"
        return f"{v:,.0f}"
    return str(v)


def data_table(rows, columns):
    th = [html.Th(c, style=TH_STYLE) for c in columns]
    body = [html.Tr([html.Td(fmt(r.get(c)), style=TD_STYLE) for c in columns])
            for r in rows]
    return html.Table([html.Tr(th, style=TH_ROW_STYLE)] + body,
                      style=TABLE_STYLE)


TH_STYLE = {"border": "1px solid #e3e6ea", "padding": "7px 10px",
            "fontSize": "12px", "textAlign": "left", "fontWeight": "bold",
            "background": "#f0f2f5", "whiteSpace": "nowrap"}
TD_STYLE = {"border": "1px solid #e3e6ea", "padding": "6px 10px",
            "fontSize": "12px", "whiteSpace": "nowrap"}
TH_ROW_STYLE = {"border": "none"}
TABLE_STYLE = {"borderCollapse": "collapse", "width": "100%",
               "overflowX": "auto"}


def create_dash_app(data):
    from dash import Dash, dcc, html, Input, Output, no_update

    app = Dash(__name__)
    app.title = "Play Store Hexbin Dashboard"

    df = data["df"]
    counts = data["counts"]
    iqr_stats = data["iqr_stats"]
    outliers_df = data["outliers"]
    game_df = data["game"]
    bins = hexbin_aggregate(df)

    kpi_n = len(df)
    kpi_rating = df["_rating_num"].mean() if kpi_n else float("nan")
    kpi_inst = df["_installs_num"].mean() if kpi_n else float("nan")
    kpi_cats = df["_category_display"].nunique() if kpi_n else 0

    def kpi_card(title, value):
        return html.Div([
            html.Div(title, style={"fontSize": "12px", "color": "#667",
                                   "textTransform": "uppercase",
                                   "letterSpacing": "0.5px"}),
            html.Div(value, style={"fontSize": "24px", "fontWeight": "bold",
                                   "color": "#1f2a37"}),
        ], style={"background": "white", "border": "1px solid #e5e7eb",
                  "borderRadius": "10px", "padding": "14px 18px",
                  "minWidth": "170px", "flex": "1",
                  "boxShadow": "0 1px 3px rgba(0,0,0,0.05)"})

    iqr_rows = iqr_stats.to_dict("records") if not iqr_stats.empty else []
    outlier_rows = []
    if not outliers_df.empty:
        for rec in outliers_df.to_dict("records"):
            outlier_rows.append({
                "App": rec["_app_str"],
                "Category": rec["_category_display"],
                "Size (MB)": rec["_size_mb"],
                "Rating": rec["_rating_num"],
                "Installs": rec["_installs_num"],
                "Reviews": rec["_reviews_num"],
                "Avg Subjectivity": rec["_subj_num"],
                "IQR reason": rec["_out_reason"],
            })

    open_now = is_open()
    locked_screen = html.Div([
        html.Div([
            html.H1("Google Play Store Dashboard",
                    style={"margin": "0 0 6px 0", "fontSize": "28px"}),
            html.Div("Live IST clock", style={"color": "#667", "fontSize": "13px"}),
            html.Div(id="locked-clock",
                     style={"fontSize": "52px", "fontWeight": "bold",
                            "color": "#1f2a37", "margin": "10px 0"}),
            html.Div("This dashboard is only available between 5:00 PM and "
                     "7:00 PM IST (Asia/Kolkata).",
                     style={"fontSize": "16px", "color": "#374151"}),
            html.Div(id="locked-countdown",
                     style={"fontSize": "15px", "color": "#6b7280",
                            "marginTop": "10px"}),
            html.Div("The page refreshes the time gate automatically — no "
                     "manual reload needed.",
                     style={"fontSize": "13px", "color": "#9ca3af",
                            "marginTop": "18px"}),
        ], style={"background": "white", "border": "1px solid #e5e7eb",
                  "borderRadius": "12px", "padding": "40px",
                  "textAlign": "center", "maxWidth": "640px",
                  "boxShadow": "0 2px 8px rgba(0,0,0,0.05)"}),
    ], id="locked-container",
       style={"display": "none" if open_now else "flex",
              "justifyContent": "center",
              "alignItems": "center", "minHeight": "80vh",
              "padding": "24px", "background": "#f7f8fa",
              "fontFamily": "Arial, sans-serif"})

    dashboard = html.Div([
        html.Div([
            html.H1("Google Play Store — Hexbin Density Dashboard",
                    style={"margin": "0 0 4px 0", "fontSize": "24px"}),
            html.Div([
                html.Span("X = App Size (MB)   •   Y = Rating   •   "
                          "Hex colour = Average Installs   •   "),
                html.Span("Game highlighted in pink",
                          style={"color": GAME_COLOR, "fontWeight": "bold"}),
                html.Span("   •   Live IST time: "),
                html.Span(id="dash-clock", style={"fontWeight": "bold"}),
            ], style={"color": "#4b5563", "fontSize": "13px"}),
            html.Div([
                html.Span(f"Store data: {data['store_file']}   |   "),
                html.Span(f"Reviews/sentiment data: {data['reviews_file']}   |   "),
                html.Span(f"Rows: raw {counts['raw']:,} → category "
                          f"{counts['after_category']:,} → numeric "
                          f"{counts['after_numeric']:,} → final "
                          f"{counts['final']:,}   |   "),
                html.Span(f"Game apps: {len(game_df)}   |   "
                          f"IQR outliers: {len(outliers_df)}"),
            ], style={"marginTop": "8px", "fontSize": "12px", "color": "#6b7280"}),
            html.Details([
                html.Summary("Filters & category display rules",
                             style={"cursor": "pointer", "fontSize": "12px",
                                    "color": "#4b5563"}),
                html.Div(
                    "Categories: Game, Beauty, Business, Comics, Communication, "
                    "Dating, Entertainment, Social, Events.  Filters (all applied "
                    "together): Rating > 3.5, Installs > 50,000, Reviews > 500, "
                    "Size 10–100 MB, average Sentiment Subjectivity > 0.5.  App "
                    "names containing the letter \u201cs\u201d (case-insensitive) are "
                    "excluded.  Average Sentiment Subjectivity is merged from the "
                    "reviews file by App.  Category display names: "
                    "Beauty → सुंदरता (Hindi), Business → வணிகம் (Tamil), "
                    "Dating → Dating.",
                    style={"fontSize": "12px", "color": "#4b5563",
                           "marginTop": "6px", "lineHeight": "1.5"}),
            ], style={"marginTop": "10px"}),
        ], style={"background": "white", "padding": "20px 24px",
                  "borderBottom": "1px solid #e5e7eb"}),

        html.Div([
            html.Div([
                kpi_card("Filtered apps", f"{kpi_n:,}"),
                kpi_card("Avg rating", f"{kpi_rating:.2f}" if pd.notna(kpi_rating) else "—"),
                kpi_card("Avg installs", f"{kpi_inst:,.0f}" if pd.notna(kpi_inst) else "—"),
                kpi_card("Categories", f"{kpi_cats}"),
                kpi_card("Game apps", f"{len(game_df):,}"),
                kpi_card("IQR outliers", f"{len(outliers_df):,}"),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),

            html.Div([
                html.Div([
                    html.Label("Hex grid — size bins (X)",
                               style={"fontSize": "12px", "color": "#4b5563"}),
                    dcc.Slider(8, 30, 1, value=18, id="nx-slider"),
                ], style={"flex": "1", "minWidth": "220px"}),
                html.Div([
                    html.Label("Hex grid — rating bins (Y)",
                               style={"fontSize": "12px", "color": "#4b5563"}),
                    dcc.Slider(6, 24, 1, value=14, id="ny-slider"),
                ], style={"flex": "1", "minWidth": "220px"}),
                html.Div([
                    html.Label("Colour scale",
                               style={"fontSize": "12px", "color": "#4b5563"}),
                    dcc.Dropdown(
                        options=[{"label": c, "value": c} for c in
                                 ["YlOrRd", "Viridis", "Plasma", "Inferno",
                                  "Turbo", "Blues"]],
                        value="YlOrRd", clearable=False, id="colorscale-dd",
                    ),
                ], style={"flex": "1", "minWidth": "170px"}),
                html.Div([
                    dcc.Checklist(
                        options=[{"label": " Show Game (pink)", "value": "game"},
                                 {"label": " Show IQR outliers", "value": "out"}],
                        value=["game", "out"], id="overlay-chk",
                    ),
                ], style={"flex": "1", "minWidth": "210px", "paddingTop": "22px"}),
            ], style={"display": "flex", "gap": "20px", "flexWrap": "wrap",
                      "background": "white", "border": "1px solid #e5e7eb",
                      "borderRadius": "10px", "padding": "14px 18px",
                      "marginTop": "14px"}),

            dcc.Graph(id="hexbin-graph",
                      figure=build_hexbin_figure(bins, df, game_df, outliers_df)),

            html.Div([
                html.Div([
                    html.H3("Installs distribution per category",
                            style={"marginTop": "0"}),
                    dcc.Graph(figure=build_box_figure(df)),
                ], style={"flex": "1", "minWidth": "320px", "background": "white",
                          "border": "1px solid #e5e7eb", "borderRadius": "10px",
                          "padding": "12px"}),
                html.Div([
                    html.H3(f"Category-level IQR outliers ({len(outlier_rows)})",
                            style={"marginTop": "0"}),
                    html.P("IQR is computed separately per category on Installs: "
                           "Q1 = 25th percentile, Q3 = 75th, IQR = Q3 − Q1, "
                           "bounds = Q1 − 1.5·IQR and Q3 + 1.5·IQR. Outliers are "
                           "labeled on the hexbin plot (black diamonds).",
                           style={"fontSize": "12px", "color": "#6b7280",
                                  "lineHeight": "1.5"}),
                    data_table(outlier_rows,
                               ["App", "Category", "Size (MB)", "Rating",
                                "Installs", "Reviews", "Avg Subjectivity",
                                "IQR reason"]) if outlier_rows
                    else html.P("No IQR outliers in the filtered data.",
                                style={"fontSize": "13px", "color": "#6b7280"}),
                    html.H4("IQR bounds per category",
                            style={"marginBottom": "6px"}),
                    data_table(iqr_rows,
                               ["Category", "Apps", "Q1", "Q3", "IQR",
                                "Lower", "Upper", "Outliers"]),
                ], style={"flex": "1", "minWidth": "380px", "background": "white",
                          "border": "1px solid #e5e7eb", "borderRadius": "10px",
                          "padding": "12px"}),
            ], style={"display": "flex", "gap": "14px", "flexWrap": "wrap",
                      "marginTop": "14px"}),

            html.Div("Marginal distributions: App Size histogram (top) and "
                     "Rating histogram (right) of the filtered dataset.  "
                     "Category display names: Beauty → सुंदरता, "
                     "Business → வணிகம், Dating → Dating.  "
                     f"Auto time-refresh every 1 s; page auto-reload every "
                     f"{RELOAD_MS // 60000} min while open.",
                     style={"fontSize": "12px", "color": "#9ca3af",
                            "marginTop": "16px", "textAlign": "center"}),
        ], style={"padding": "18px 24px", "background": "#f7f8fa",
                  "fontFamily": "Arial, sans-serif"}),
    ], id="dashboard-container",
       style={"display": "block" if open_now else "none"})

    app.layout = html.Div([
        dcc.Interval(id="time-interval", interval=1000, n_intervals=0),
        dcc.Interval(id="reload-interval", interval=RELOAD_MS, n_intervals=0),
        html.Div(id="reload-sentinel", style={"display": "none"}),
        locked_screen,
        dashboard,
    ], style={"margin": "0", "fontFamily": "Arial, sans-serif"})

    app.clientside_callback(
        """function(n){
            var el = document.getElementById('dashboard-container');
            if (n && el && el.style.display !== 'none') {
                window.location.reload();
            }
            return window.dash_clientside.no_update;
        }""",
        Output("reload-sentinel", "children"),
        Input("reload-interval", "n_intervals"),
        prevent_initial_call=True,
    )

    @    app.callback(
        Output("dashboard-container", "style"),
        Output("locked-container", "style"),
        Output("locked-clock", "children"),
        Output("locked-countdown", "children"),
        Output("dash-clock", "children"),
        Input("time-interval", "n_intervals"),
    )
    def _time_gate(n):
        state = gate_state()
        if state["open"]:
            return ({"display": "block"}, {"display": "none"},
                    no_update, no_update, state["clock"])
        return ({"display": "none"}, {"display": "flex"},
                state["clock"], state["countdown"], no_update)

    @app.callback(
        Output("hexbin-graph", "figure"),
        Input("nx-slider", "value"),
        Input("ny-slider", "value"),
        Input("colorscale-dd", "value"),
        Input("overlay-chk", "value"),
    )
    def _update_hexbin(nx, ny, cs, overlays):
        b = hexbin_aggregate(df, nx=nx, ny=ny)
        return build_hexbin_figure(b, df, game_df, outliers_df, colorscale=cs,
                                   show_game=("game" in (overlays or [])),
                                   show_outliers=("out" in (overlays or [])),
                                   nx=nx, ny=ny)

    return app


if __name__ == "__main__":
    try:
        DATA = build_dataset()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)

    print("=" * 70)
    print("Filter counts:", DATA["counts"])
    print("Hexbins:", len(hexbin_aggregate(DATA["df"])),
          "| Game:", len(DATA["game"]),
          "| IQR outliers:", len(DATA["outliers"]))
    print("Current IST:", ist_now().isoformat(),
          "| Dashboard open now:", is_open())
    print("=" * 70)

    dash_app = create_dash_app(DATA)
    dash_app.run(debug=False, 
                 host="0.0.0.0", 
                 port=int(os.environ.get("PORT", 8050))
