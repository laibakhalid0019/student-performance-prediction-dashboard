# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="CS-4048 Student Performance Dashboard",
    layout="wide",
    page_icon="📊",
)

# =========================
# LOAD DATA
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

cleaned_path = os.path.join(OUTPUT_DIR, "cleaned_marks_dataset.csv")
results_path = os.path.join(OUTPUT_DIR, "model_results.csv")

df = pd.read_csv(cleaned_path)
results_df = pd.read_csv(results_path)

# =========================
# DETECT IMPORTANT COLUMNS (same logic as notebook)
# =========================
cols = df.columns

assign_cols = [c for c in cols if c.lower().startswith("as:")]
quiz_cols = [c for c in cols if c.lower().startswith("qz:")]
project_cols = [c for c in cols if c.lower().startswith("proj")]
mid1_cols = [c for c in cols if c.lower().startswith("s-i")]
mid2_cols = [c for c in cols if c.lower().startswith("s-ii")]
final_cols = [c for c in cols if c.lower().startswith("final")]

MID1_COL = mid1_cols[0] if mid1_cols else None
MID2_COL = mid2_cols[0] if mid2_cols else None
FINAL_COL = final_cols[0] if final_cols else None

# =========================
# GLOBAL STYLING (BLUE PROFESSIONAL THEME)
# =========================
st.markdown(
    """
    <style>
    .main {
        background-color: #0f172a;
        color: #e5e7eb;
    }
    .card {
        background: #111827;
        padding: 18px 18px 12px 18px;
        border-radius: 14px;
        border: 1px solid #1d4ed8;
        cursor: pointer;
        transition: all 0.2s ease-in-out;
    }
    .card:hover {
        background: #1e293b;
        border-color: #60a5fa;
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(15,23,42,0.7);
    }
    .card-title {
        font-size: 18px;
        font-weight: 700;
        color: #bfdbfe;
    }
    .card-subtitle {
        font-size: 13px;
        color: #9ca3af;
    }
    .section-title {
        font-size: 26px;
        font-weight: 700;
        margin-top: 8px;
        margin-bottom: 4px;
        color: #eff6ff;
    }
    .section-subtitle {
        font-size: 14px;
        color: #9ca3af;
        margin-bottom: 16px;
    }
    .metric-card {
        background: #111827;
        padding: 14px;
        border-radius: 12px;
        border: 1px solid #1d4ed8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("📊 CS-4048 Dashboard")
st.sidebar.markdown("**Student Performance Prediction System**")

page_options = [
    "🏠 Home",
    "📁 Data Overview",
    "📈 Visualizations",
    "🤖 Models",
    "🔮 Predict Final",
    "📜 Workflow & About",
]

if "active_page" not in st.session_state:
    st.session_state["active_page"] = "🏠 Home"

selected_sidebar = st.sidebar.radio("Navigate", page_options, index=0)

# If user clicks in sidebar, update active page
st.session_state["active_page"] = selected_sidebar


# =========================
# HELPER: CARD CLICK (using form buttons)
# =========================
def nav_card(title, subtitle, key, target_page):
    with st.form(key):
        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">{title}</div>
                <div class="card-subtitle">{subtitle}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        clicked = st.form_submit_button("")
    if clicked:
        st.session_state["active_page"] = target_page


# =========================
# PAGE: HOME (CARDS MENU)
# =========================
def page_home():
    # Title & subtitle
    st.markdown("<div class='section-title'>📊 CS-4048 Student Performance Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subtitle'>Analytics for quizzes, assignments, mid exams & final exam performance.</div>", unsafe_allow_html=True)

    # ===== KPIs / Small Summary =====
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("🎒 Total Records", df.shape[0])
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("📚 Total Features", df.shape[1])
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        avg_mid = f"{df[MID1_COL].mean():.2f}" if MID1_COL else "N/A"
        st.metric("✏ Avg Midterm I", avg_mid)
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        avg_final = f"{df[FINAL_COL].mean():.2f}" if FINAL_COL else "N/A"
        st.metric("🏁 Avg Final", avg_final)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ===== Mini Bar Chart (Top Feature vs Final) =====
    if FINAL_COL and len(df.columns) > 4:
        top_feature = list(df.select_dtypes(include=[np.number]).columns)[0]
        st.markdown("### 📈 Trend Overview (Sample Feature vs Final)")
        fig = px.scatter(
            df,
            x=top_feature,
            y=FINAL_COL,
            trendline="ols",
            color_discrete_sequence=["#3b82f6"],
        )
        fig.update_layout(
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font_color="#e5e7eb",
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Final marks not detected — chart skipped.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ===== Navigation Cards (Small & Modern) =====
    st.markdown("### 🚀 Quick Actions")

    colA, colB, colC = st.columns(3)

    with colA:
        nav_card(
            "📁 Data Overview",
            "View dataset & summary stats",
            "card_data2",
            "📁 Data Overview",
        )
    with colB:
        nav_card(
            "📈 Visualizations",
            "Explore graphs & distributions",
            "card_visuals2",
            "📈 Visualizations",
        )
    with colC:
        nav_card(
            "🤖 Models",
            "Compare regression models",
            "card_models2",
            "🤖 Models",
        )

    colD, colE, colF = st.columns(3)

    with colD:
        nav_card(
            "🔮 Predict Final",
            "Estimate final score",
            "card_predict2",
            "🔮 Predict Final",
        )
    with colE:
        nav_card(
            "📜 Workflow & About",
            "ML Pipeline & Viva Explain",
            "card_workflow2",
            "📜 Workflow & About",
        )
    with colF:
        st.write("")  # empty for spacing


# =========================
# PAGE: DATA OVERVIEW
# =========================
def page_data_overview():
    st.markdown("<div class='section-title'>📁 Data Overview</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-subtitle'>Merged dataset from all 6 Excel sheets, after cleaning and preprocessing.</div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Total Rows (Students x Attempts)", df.shape[0])
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Total Columns (Features)", df.shape[1])
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        if MID1_COL:
            st.metric("Avg Midterm I", f"{df[MID1_COL].mean():.2f}")
        else:
            st.metric("Avg Midterm I", "N/A")
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        if FINAL_COL:
            st.metric("Avg Final", f"{df[FINAL_COL].mean():.2f}")
        else:
            st.metric("Avg Final", "N/A")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### 🔍 Preview (first 15 rows)")
    st.dataframe(df.head(15), use_container_width=True)

    st.markdown("### 📊 Summary Statistics")
    st.dataframe(df.describe(include="all").T, use_container_width=True)

    st.markdown("### 📥 Download Cleaned Dataset")
    st.download_button(
        label="⬇ Download cleaned_marks_dataset.csv",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_marks_dataset.csv",
        mime="text/csv",
    )


# =========================
# PAGE: VISUALIZATIONS
# =========================
def page_visuals():
    st.markdown("<div class='section-title'>📈 Visualizations</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-subtitle'>Understand the distribution of marks and relationships between assessments.</div>",
        unsafe_allow_html=True,
    )

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.warning("No numeric columns found in the dataset.")
        return

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Histogram")
        num_col = st.selectbox("Select column for distribution", numeric_cols, key="hist_col")
        fig = px.histogram(
            df,
            x=num_col,
            nbins=20,
            color_discrete_sequence=["#2563eb"],
            opacity=0.9,
        )
        fig.update_layout(
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font_color="#e5e7eb",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Box Plot (Spread & Outliers)")
        num_col2 = st.selectbox("Select column for box plot", numeric_cols, key="box_col")
        fig2 = px.box(
            df,
            y=num_col2,
            color_discrete_sequence=["#38bdf8"],
            points="all",
        )
        fig2.update_layout(
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font_color="#e5e7eb",
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 📉 Relationship with Final Exam")
    if FINAL_COL and len(numeric_cols) > 1:
        feature_for_scatter = st.selectbox(
            "Select feature to compare vs Final",
            [c for c in numeric_cols if c != FINAL_COL],
            key="scatter_col",
        )
        fig3 = px.scatter(
            df,
            x=feature_for_scatter,
            y=FINAL_COL,
            color_discrete_sequence=["#22c55e"],
        )
        fig3.update_layout(
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font_color="#e5e7eb",
            xaxis_title=feature_for_scatter,
            yaxis_title=FINAL_COL,
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Final exam column not detected; cannot draw scatter vs Final.")


# =========================
# PAGE: MODELS
# =========================
def page_models():
    st.markdown("<div class='section-title'>🤖 Regression Models & Metrics</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-subtitle'>Compare Dummy, Simple Linear and Polynomial Regression for each research question.</div>",
        unsafe_allow_html=True,
    )

    rq_options = sorted(results_df["RQ"].unique())
    selected_rq = st.selectbox("Select Research Question", rq_options)

    rq_df = results_df[results_df["RQ"] == selected_rq].copy()

    st.markdown("### 📋 Model Comparison Table")
    st.dataframe(
        rq_df[
            [
                "Model",
                "Features",
                "Train_MAE",
                "Train_RMSE",
                "Train_R2",
                "Test_MAE",
                "Test_RMSE",
                "Test_R2",
                "Bootstrap_MAE_CI_Lower",
                "Bootstrap_MAE_CI_Upper",
            ]
        ],
        use_container_width=True,
    )

    # Best model by Test MAE
    best_idx = rq_df["Test_MAE"].idxmin()
    best_row = rq_df.loc[best_idx]

    st.markdown("### 🏆 Best Model Summary")
    st.success(
        f"""
        **Research Question:** {selected_rq}  
        **Best Model:** {best_row['Model']}  
        **Features:** {best_row['Features']}  
        **Test MAE:** {best_row['Test_MAE']:.3f}  
        **Test RMSE:** {best_row['Test_RMSE']:.3f}  
        **Test R²:** {best_row['Test_R2']:.3f}
        """
    )

    st.markdown("### 📊 Test MAE across Models")
    fig_mae = px.bar(
        rq_df,
        x="Model",
        y="Test_MAE",
        color="Model",
        color_discrete_sequence=px.colors.sequential.Blues,
        text="Test_MAE",
    )
    fig_mae.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_mae.update_layout(
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font_color="#e5e7eb",
        xaxis_title="Model",
        yaxis_title="Test MAE",
    )
    st.plotly_chart(fig_mae, use_container_width=True)

    # CI bar if available
    ci_l = best_row["Bootstrap_MAE_CI_Lower"]
    ci_u = best_row["Bootstrap_MAE_CI_Upper"]

    st.markdown("### 🎯 95% Confidence Interval for MAE (Best Model)")
    if not np.isnan(ci_l) and not np.isnan(ci_u):
        ci_df = pd.DataFrame(
            {
                "Model": [best_row["Model"]],
                "MAE": [best_row["Test_MAE"]],
                "Lower": [ci_l],
                "Upper": [ci_u],
            }
        )
        fig_ci = px.bar(
            ci_df,
            x="Model",
            y="MAE",
            error_y=ci_df["Upper"] - ci_df["MAE"],
            error_y_minus=ci_df["MAE"] - ci_df["Lower"],
            color_discrete_sequence=["#1d4ed8"],
        )
        fig_ci.update_layout(
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font_color="#e5e7eb",
            yaxis_title="MAE",
        )
        st.plotly_chart(fig_ci, use_container_width=True)
    else:
        st.info("Bootstrap CI is only computed for the polynomial model; if it's not the best by MAE, CI may be NaN.")


# =========================
# PAGE: PREDICT FINAL (DEMO)
# =========================
def page_predict():
    st.markdown("<div class='section-title'>🔮 Demo: Estimate Final Exam Marks</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-subtitle'>This is a simple weighted demo formula: 0.2 × Quiz + 0.2 × Assignment + 0.3 × Mid I + 0.3 × Mid II.</div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        quiz_total = st.number_input("Total Quiz Marks (0–100)", 0.0, 100.0, 20.0)
        assign_total = st.number_input("Total Assignment Marks (0–100)", 0.0, 100.0, 20.0)
    with col2:
        mid1 = st.number_input("Midterm I Marks (0–100)", 0.0, 100.0, 25.0)
        mid2 = st.number_input("Midterm II Marks (0–100)", 0.0, 100.0, 25.0)

    if st.button("🔮 Predict Final (Demo)"):
        predicted = 0.2 * quiz_total + 0.2 * assign_total + 0.3 * mid1 + 0.3 * mid2
        st.success(f"🎯 Estimated Final Marks: **{predicted:.2f} / 100**")
        st.caption("Note: This front-end formula is only for demo; real regression models are trained & evaluated in the Jupyter notebook.")


# =========================
# PAGE: WORKFLOW & ABOUT
# =========================
def page_workflow():
    st.markdown("<div class='section-title'>📜 Workflow & ML Pipeline</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-subtitle'>End-to-end project pipeline you can explain in viva and report.</div>",
        unsafe_allow_html=True,
    )

    steps = [
        "1️⃣ Load 6 Excel sheets (each sheet = different section/semester).",
        "2️⃣ Merge all sheets row-wise into a single dataframe (added `sheet_source` column).",
        "3️⃣ Basic cleaning: dropped junk columns, fixed types, handled missing values for formative assessments.",
        "4️⃣ Domain-based feature selection for each RQ (no future exam marks used to predict earlier exams).",
        "5️⃣ Train/Test split (80/20, random_state=42) to prevent data leakage.",
        "6️⃣ Scikit-learn Pipelines: Imputation + Scaling + Simple/Polynomial Regression.",
        "7️⃣ DummyRegressor as baseline; Linear & Polynomial models compared with MAE, RMSE, R².",
        "8️⃣ Bootstrapping (500 samples) to build 95% CI around MAE of best model.",
        "9️⃣ Exported cleaned dataset & model_results CSV and built this Streamlit dashboard.",
    ]
   

# =========================
# ROUTER: RENDER THE ACTIVE PAGE
# =========================
if st.session_state["active_page"] == "🏠 Home":
    page_home()
elif st.session_state["active_page"] == "📁 Data Overview":
    page_data_overview()
elif st.session_state["active_page"] == "📈 Visualizations":
    page_visuals()
elif st.session_state["active_page"] == "🤖 Models":
    page_models()
elif st.session_state["active_page"] == "🔮 Predict Final":
    page_predict()
elif st.session_state["active_page"] == "📜 Workflow & About":
    page_workflow()
