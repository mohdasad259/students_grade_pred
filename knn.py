import streamlit as st
import joblib
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="EduPredict Pro",
    page_icon="🎓",
    layout="wide"
)

# --- GOLDEN THEME CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600;700&family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

    * { box-sizing: border-box; }

    /* ── BACKGROUND ── */
    .stApp {
        background: #0E0B05;
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        padding: 2rem 2.5rem 4rem !important;
        max-width: 1280px !important;
    }

    /* ── ALL TEXT ── */
    .stMarkdown p, .stText,
    .element-container p {
        color: #C9A84C !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* ── HEADINGS ── */
    h1 {
        font-family: 'Cormorant Garamond', serif !important;
        font-size: 3rem !important;
        font-weight: 700 !important;
        color: #D4AF37 !important;
        letter-spacing: -0.5px !important;
        line-height: 1.1 !important;
        margin-bottom: 0.2rem !important;
    }

    h2, h3 {
        font-family: 'Cormorant Garamond', serif !important;
        font-weight: 600 !important;
        color: #E8D07A !important;
        letter-spacing: -0.2px !important;
    }

    /* ── LABELS ── */
    label, .stSlider label, .stNumberInput label,
    .stSelectbox label, .stRadio label {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 1.8px !important;
        color: #8A6E1E !important;
    }

    /* ── EXPANDER ── */
    [data-testid="stExpander"] {
        background: rgba(212,175,55,0.04) !important;
        border: 1px solid rgba(212,175,55,0.15) !important;
        border-radius: 12px !important;
        margin-bottom: 10px !important;
    }

    [data-testid="stExpander"] summary {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
        color: #B8922A !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.5px !important;
        padding: 0.85rem 1.1rem !important;
        list-style: none !important;
    }

    /* FIX: Hide "keyboard_arrow_down" raw icon text from Streamlit Material icons */
    [data-testid="stExpander"] summary span[class*="material"] {
        font-size: 0 !important;
        line-height: 0 !important;
        color: transparent !important;
        width: 20px !important;
        height: 20px !important;
        display: inline-block !important;
        overflow: hidden !important;
    }

    [data-testid="stExpander"] summary svg {
        fill: #B8922A !important;
        color: #B8922A !important;
    }

    [data-testid="stExpander"] details summary::-webkit-details-marker {
        display: none !important;
    }

    /* ── INPUTS ── */
    input[type="number"], .stTextInput input {
        background: rgba(212,175,55,0.05) !important;
        border: 1px solid rgba(212,175,55,0.22) !important;
        border-radius: 8px !important;
        color: #E8D07A !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
    }

    input[type="number"]:focus {
        border-color: #D4AF37 !important;
        box-shadow: 0 0 0 3px rgba(212,175,55,0.12) !important;
    }

    /* ── SELECTBOX ── */
    .stSelectbox > div > div {
        background: rgba(212,175,55,0.05) !important;
        border: 1px solid rgba(212,175,55,0.22) !important;
        border-radius: 8px !important;
        color: #E8D07A !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* ── SLIDER ── */
    .stSlider [data-testid="stThumbValue"] {
        color: #D4AF37 !important;
        font-weight: 700 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 0.82rem !important;
    }

    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background: #D4AF37 !important;
        border: 2px solid #0E0B05 !important;
        box-shadow: 0 0 8px rgba(212,175,55,0.4) !important;
    }

    [data-testid="stSlider"] > div > div > div > div {
        background: linear-gradient(90deg, #5A3800, #D4AF37) !important;
    }

    /* ── RADIO ── */
    .stRadio [data-testid="stMarkdownContainer"] p {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 500 !important;
        color: #C9A84C !important;
        font-size: 0.9rem !important;
    }

    .stRadio [data-baseweb="radio"] div:first-child {
        border-color: #8A6E1E !important;
    }

    .stRadio [aria-checked="true"] div:first-child {
        background: #D4AF37 !important;
        border-color: #D4AF37 !important;
    }

    /* ── CHECKBOX ── */
    .stCheckbox [data-baseweb="checkbox"] div {
        border-color: rgba(212,175,55,0.4) !important;
        border-radius: 4px !important;
        background: rgba(212,175,55,0.04) !important;
    }

    .stCheckbox [aria-checked="true"] div {
        background: #8A6E1E !important;
        border-color: #8A6E1E !important;
    }

    .stCheckbox span {
        font-family: 'Inter', sans-serif !important;
        color: #C9A84C !important;
        font-size: 0.9rem !important;
    }

    /* ── SELECT SLIDER ── */
    .stSelectSlider [data-baseweb="slider"] div[role="slider"] {
        background: #D4AF37 !important;
        border: 2px solid #0E0B05 !important;
    }

    /* ── BUTTON ── */
    .stButton > button {
        width: 100% !important;
        border-radius: 8px !important;
        height: 3.5em !important;
        background: #100D03 !important;
        color: #D4AF37 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 0.82rem !important;
        font-weight: 700 !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        border: 1px solid rgba(212,175,55,0.35) !important;
        transition: all 0.2s ease !important;
        margin-top: 0.8rem !important;
    }

    .stButton > button:hover {
        background: rgba(212,175,55,0.08) !important;
        border-color: #D4AF37 !important;
        color: #F5E09A !important;
    }

    /* ── ALERT ── */
    [data-testid="stAlert"] {
        background: rgba(212,175,55,0.08) !important;
        border: 1px solid rgba(212,175,55,0.3) !important;
        border-radius: 10px !important;
    }

    .stAlert p {
        color: #D4AF37 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }

    /* ── PROGRESS BAR ── */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #5A3800, #D4AF37, #F5E09A) !important;
        border-radius: 999px !important;
    }

    .stProgress > div > div > div {
        background: rgba(212,175,55,0.1) !important;
        border-radius: 999px !important;
        height: 7px !important;
    }

    /* ── DIVIDER ── */
    hr {
        border: none !important;
        border-top: 1px solid rgba(212,175,55,0.1) !important;
        margin: 1.2rem 0 !important;
    }

    /* ── CAPTION ── */
    .stCaption {
        color: rgba(180,150,50,0.35) !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.8px !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }

    /* ── NUMBER INPUT ── */
    [data-testid="stNumberInput"] input {
        background: rgba(212,175,55,0.05) !important;
        border: 1px solid rgba(212,175,55,0.22) !important;
        border-radius: 8px !important;
        color: #E8D07A !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
    }

    /* ── SCROLLBAR ── */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: #0E0B05; }
    ::-webkit-scrollbar-thumb { background: rgba(212,175,55,0.3); border-radius: 3px; }

    /* ── CUSTOM COMPONENTS ── */
    .gold-eyebrow {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #6B5C2E;
        margin-bottom: 4px;
    }

    .gold-divider {
        height: 2px;
        background: linear-gradient(90deg, #5A3800, #D4AF37, #F5E09A, #D4AF37, #5A3800);
        border-radius: 999px;
        margin: 0.6rem 0 1.4rem;
        border: none !important;
    }

    .section-label {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #6B5C2E;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(212,175,55,0.1);
        margin-bottom: 14px;
    }

    .pred-card {
        background: #0A0802;
        border: 1px solid rgba(212,175,55,0.2);
        border-radius: 14px;
        padding: 1.5rem 1.6rem;
        position: relative;
        overflow: hidden;
        margin-top: 0.8rem;
    }

    .pred-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #5A3800, #D4AF37, #F5E09A, #D4AF37, #5A3800);
        border-radius: 14px 14px 0 0;
    }

    .model-tag {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        color: #8A6E1E;
        margin-bottom: 4px;
    }

    .gpa-display {
        text-align: center;
        padding: 1.2rem 0 0.6rem;
    }

    .gpa-lbl {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.68rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #6B5C2E;
        margin-bottom: 4px;
    }

    .gpa-num {
        font-family: 'Cormorant Garamond', serif;
        font-size: 4.5rem;
        font-weight: 700;
        color: #D4AF37;
        line-height: 1;
        letter-spacing: -1px;
    }

    .gpa-denom {
        font-size: 1.3rem;
        color: #5A4A1E;
        font-weight: 400;
    }

    .gpa-sub {
        font-size: 0.72rem;
        color: #5A4A1E;
        margin-top: 4px;
        font-family: 'Inter', sans-serif;
    }

    .status-row {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 7px;
        margin-top: 0.5rem;
    }

    .status-dot {
        width: 6px; height: 6px;
        border-radius: 50%;
        background: #D4AF37;
        display: inline-block;
    }

    .status-text {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #B8922A;
    }

    .stats-grid {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 8px;
        margin-top: 14px;
    }

    .stat-box {
        background: rgba(212,175,55,0.04);
        border: 1px solid rgba(212,175,55,0.1);
        border-radius: 8px;
        padding: 10px 8px 8px;
        text-align: center;
    }

    .stat-val {
        font-family: 'Cormorant Garamond', serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: #D4AF37;
        line-height: 1;
    }

    .stat-lbl {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.6rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #5A4A1E;
        margin-top: 3px;
    }

    .grade-box {
        text-align: center;
        padding: 1.4rem 1rem;
        border-radius: 10px;
        margin-top: 1rem;
    }

    .grade-lbl {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.68rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #6B5C2E;
        margin-bottom: 6px;
    }

    .grade-letter {
        font-family: 'Cormorant Garamond', serif;
        font-size: 6.5rem;
        font-weight: 700;
        line-height: 1;
    }

    .grade-desc {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 6px;
        opacity: 0.85;
    }
    </style>
""", unsafe_allow_html=True)

# ── LOAD MODELS ──
@st.cache_resource
def load_assets():
    sc  = joblib.load('sc.pkl')
    nb  = joblib.load('nb.pkl')
    knn = joblib.load('model.pkl')
    return sc, nb, knn

try:
    sc, nb, knn = load_assets()
except Exception as e:
    st.error(f"Model files not found — {e}")
    st.stop()

# ── HEADER ──
st.markdown('<div class="gold-eyebrow">Academic Intelligence Suite</div>', unsafe_allow_html=True)
st.title("EduPredict Pro")
st.markdown(
    "<p style='color:#6B5C2E; font-size:0.92rem; margin-top:-0.3rem; "
    "font-family:Inter,sans-serif;'>"
    "Student performance prediction powered by KNN & Naive Bayes"
    "</p>",
    unsafe_allow_html=True
)
st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

# ── COLUMNS ──
left_col, right_col = st.columns([1, 1.2], gap="large")

with left_col:
    st.markdown('<div class="section-label">Student Profile</div>', unsafe_allow_html=True)

    with st.expander("Demographics", expanded=True):
        a1, a2 = st.columns(2)
        age    = a1.number_input("Age", 15, 20, 18)
        gender = a2.selectbox("Gender", [0, 1],
                              format_func=lambda x: "Male" if x == 1 else "Female")
        ethnicity = st.selectbox(
            "Ethnicity", [0, 1, 2, 3],
            format_func=lambda x: ["Caucasian","African American","Asian","Other"][x]
        )

    with st.expander("Academic Background", expanded=True):
        study_time = st.slider("Weekly Study Time (hrs)", 0.0, 25.0, 10.0, step=0.5)
        absences   = st.number_input("Total Absences", 0, 30, 2)
        b1, b2 = st.columns(2)
        parent_edu  = b1.selectbox(
            "Parental Education", [0, 1, 2, 3, 4],
            format_func=lambda x: ["None","High School","Some College","Bachelor's","Higher"][x]
        )
        parent_supp = b2.select_slider(
            "Parental Support",
            options=[0, 1, 2, 3, 4],
            format_func=lambda x: ["None","Low","Moderate","High","Very High"][x]
        )

    with st.expander("Activities and Extras"):
        c1, c2 = st.columns(2)
        tutoring     = c1.checkbox("Tutoring")
        extra        = c2.checkbox("Extracurricular")
        sports       = c1.checkbox("Sports")
        music        = c2.checkbox("Music")
        volunteering = st.checkbox("Volunteering")
        tut, ext, spt, mus, vol = map(int, [tutoring, extra, sports, music, volunteering])

with right_col:
    st.markdown('<div class="section-label">Prediction Dashboard</div>', unsafe_allow_html=True)

    choice = st.radio(
        "Select model",
        ["Predict GPA  (KNN)", "Predict Grade  (Naive Bayes)"],
        horizontal=True
    )

    st.markdown('<div class="pred-card">', unsafe_allow_html=True)

    # ── GPA MODE ──
    if choice == "Predict GPA  (KNN)":
        st.markdown('<div class="model-tag">K-Nearest Neighbours Regressor</div>', unsafe_allow_html=True)
        st.markdown(
            "<h3 style='margin:0 0 1rem; font-size:1.3rem;'>GPA Estimator</h3>",
            unsafe_allow_html=True
        )

        gc = st.selectbox(
            "Current Grade Class (0 = Top, 4 = Bottom)",
            [0, 1, 2, 3, 4]
        )

        if st.button("Calculate GPA"):
            features = np.array([[age, gender, ethnicity, parent_edu, study_time,
                                  absences, tut, parent_supp, ext, spt, mus, vol, gc]])
            scaled_f = sc.transform(features)
            gpa_val  = float(knn.predict(scaled_f)[0])

            st.markdown(
                f'<div class="gpa-display">'
                f'<div class="gpa-lbl">Estimated GPA</div>'
                f'<div class="gpa-num">{gpa_val:.2f}'
                f'<span class="gpa-denom"> / 4.00</span></div>'
                f'<div class="gpa-sub">Academic Performance Score</div>'
                f'</div>',
                unsafe_allow_html=True
            )

            st.progress(min(gpa_val / 4.0, 1.0))

            pct = (gpa_val / 4.0) * 100
            if pct >= 87:
                label, grade_letter = "Excellent Standing", "A"
            elif pct >= 72:
                label, grade_letter = "Good Standing", "B"
            elif pct >= 57:
                label, grade_letter = "Average Standing", "C"
            else:
                label, grade_letter = "Needs Improvement", "D"

            st.markdown(
                f'<div class="status-row">'
                f'<span class="status-dot"></span>'
                f'<span class="status-text">{label}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="stats-grid">'
                f'<div class="stat-box"><div class="stat-val">{pct:.0f}%</div>'
                f'<div class="stat-lbl">Score</div></div>'
                f'<div class="stat-box"><div class="stat-val">{grade_letter}</div>'
                f'<div class="stat-lbl">Letter</div></div>'
                f'<div class="stat-box"><div class="stat-val">KNN</div>'
                f'<div class="stat-lbl">Model</div></div>'
                f'</div>',
                unsafe_allow_html=True
            )

    # ── GRADE MODE ──
    else:
        st.markdown('<div class="model-tag">Gaussian Naive Bayes Classifier</div>', unsafe_allow_html=True)
        st.markdown(
            "<h3 style='margin:0 0 1rem; font-size:1.3rem;'>Grade Classifier</h3>",
            unsafe_allow_html=True
        )

        gpa_val = st.number_input("Enter GPA for classification", 0.0, 4.0, 3.0, step=0.01)

        if st.button("Predict Grade Class"):
            features_nb = np.array([[age, gender, ethnicity, parent_edu, study_time,
                                     absences, tut, parent_supp, ext, spt, mus, vol, gpa_val]])
            grade_res   = nb.predict(features_nb)
            grade_map   = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'F'}
            final_grade = grade_map.get(int(grade_res[0]), "?")

            grade_styles = {
                'A': ('#D4AF37', 'rgba(212,175,55,0.08)', 'rgba(212,175,55,0.25)', 'Outstanding'),
                'B': ('#C9A84C', 'rgba(201,168,76,0.08)',  'rgba(201,168,76,0.22)',  'Above Average'),
                'C': ('#B8922A', 'rgba(184,146,42,0.08)',  'rgba(184,146,42,0.22)',  'Average'),
                'D': ('#8A6E1E', 'rgba(138,110,30,0.08)',  'rgba(138,110,30,0.2)',   'Below Average'),
                'F': ('#5A4010', 'rgba(90,64,16,0.1)',     'rgba(90,64,16,0.3)',     'Failing'),
            }
            gc_col, gc_bg, gc_brd, gc_desc = grade_styles.get(
                final_grade, ('#D4AF37','rgba(212,175,55,0.08)','rgba(212,175,55,0.25)','Unknown')
            )

            st.markdown(
                f'<div class="grade-box" style="background:{gc_bg}; border:1px solid {gc_brd};">'
                f'<div class="grade-lbl">Predicted Grade</div>'
                f'<div class="grade-letter" style="color:{gc_col};">{final_grade}</div>'
                f'<div class="grade-desc" style="color:{gc_col};">{gc_desc}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="stats-grid" style="margin-top:12px;">'
                f'<div class="stat-box"><div class="stat-val">{gpa_val:.2f}</div>'
                f'<div class="stat-lbl">GPA In</div></div>'
                f'<div class="stat-box"><div class="stat-val">{final_grade}</div>'
                f'<div class="stat-lbl">Grade</div></div>'
                f'<div class="stat-box"><div class="stat-val">NB</div>'
                f'<div class="stat-lbl">Model</div></div>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown('</div>', unsafe_allow_html=True)

# ── FOOTER ──
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
st.caption("EduPredict Pro  ·  Models: sc.pkl · nb.pkl · model.pkl  ·  KNN + Naive Bayes")