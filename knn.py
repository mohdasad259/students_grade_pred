import streamlit as st
import joblib
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="EduPredict Pro",
    page_icon="🎓",
    layout="wide"
)

# --- HIGH CONTRAST & VISIBLE CUSTOM CSS ---
st.markdown("""
    <style>
    /* Main Background - Deep Dark Blue for better focus */
    .stApp {
        background-color: #0F172A;
    }
    
    /* Global Text Color for visibility */
    .stMarkdown, p, span {
        color: #F8FAF8 !important;
    }

    /* Titles - Bright Golden Yellow for contrast */
    h1, h2, h3 {
        color: #FACC15 !important;
        font-weight: 800 !important;
    }

    /* Input Labels - Cyan Blue for clarity */
    label {
        color: #22D3EE !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }

    /* Expander Styling - Dark Grey with Border */
    [data-testid="stExpander"] {
        background-color: #1E293B !important;
        border: 2px solid #334155 !important;
        border-radius: 12px !important;
    }

    /* Prediction Container - Solid Box with White Border */
    .prediction-container {
        padding: 30px;
        border-radius: 20px;
        background-color: #1E293B;
        border: 3px solid #FACC15;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        margin-top: 20px;
    }

    /* Vibrant Multi-Color Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3.5em;
        background: #2563EB; /* Solid Blue */
        color: white !important;
        font-size: 1.2rem !important;
        font-weight: bold;
        border: 2px solid #FFFFFF;
        transition: 0.3s;
    }
    
    .stButton>button:hover {
        background: #FACC15 !important; /* Changes to Yellow on hover */
        color: #0F172A !important;
        border: 2px solid #0F172A;
    }

    /* Success Message Contrast */
    .stAlert {
        background-color: #064E3B !important;
        color: #D1FAE5 !important;
        border: 1px solid #10B981 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    sc = joblib.load('sc.pkl')
    nb = joblib.load('nb.pkl')
    knn = joblib.load('model.pkl')
    return sc, nb, knn

try:
    sc, nb, knn = load_assets()
except Exception as e:
    st.error(f"⚠️ Error: Model files not found! {e}")
    st.stop()

# --- HEADER ---
st.title("🎓 STUDENT PERFORMANCE ANALYTICS")
st.markdown("<p style='color: #94A3B8;'>Fill the profile on the left to see predictions on the right.</p>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid #334155;'>", unsafe_allow_html=True)

# --- MAIN LAYOUT ---
left_col, right_col = st.columns([1, 1.2], gap="large")

with left_col:
    st.subheader("📝 Student Profile")
    
    with st.expander("👤 Physical & Demographic", expanded=True):
        a1, a2 = st.columns(2)
        age = a1.number_input("Age", 15, 20, 18)
        gender = a2.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x==1 else "Female")
        ethnicity = st.selectbox("Ethnicity", [0, 1, 2, 3], 
                                 format_func=lambda x: ["Caucasian", "African American", "Asian", "Other"][x])
    
    with st.expander("📖 Academic Background", expanded=True):
        study_time = st.slider("Weekly Study Time (Hrs)", 0.0, 25.0, 10.0)
        absences = st.number_input("Total Absences", 0, 30, 2)
        parent_edu = st.selectbox("Parental Education Level", [0, 1, 2, 3, 4],
                                  format_func=lambda x: ["None", "High School", "Some College", "Bachelor's", "Higher"][x])
        parent_supp = st.select_slider("Parental Support Level", options=[0, 1, 2, 3, 4],
                                      format_func=lambda x: ["None", "Low", "Moderate", "High", "Very High"][x])

    with st.expander("⚽ Extracurricular & Activities"):
        c1, c2 = st.columns(2)
        tutoring = c1.checkbox("Tutoring")
        extra = c2.checkbox("Extracurricular")
        sports = c1.checkbox("Sports")
        music = c2.checkbox("Music")
        volunteering = st.checkbox("Volunteering")
        
        # Mapping to numbers for models
        tut, ext, spt, mus, vol = map(int, [tutoring, extra, sports, music, volunteering])

with right_col:
    st.subheader("🔮 Prediction Dashboard")
    
    # Choice Radio with high contrast text
    choice = st.radio("Select Prediction Type:", ["Predict GPA (KNN)", "Predict Grade (Naive Bayes)"], horizontal=True)
    
    st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
    
    if choice == "Predict GPA (KNN)":
        st.write("### 📈 GPA Estimator")
        gc = st.selectbox("Current Grade Class", [0, 1, 2, 3, 4], help="0 is Top, 4 is Bottom.")
        
        if st.button("CALCULATE GPA"):
            features = np.array([[age, gender, ethnicity, parent_edu, study_time, absences, 
                                 tut, parent_supp, ext, spt, mus, vol, gc]])
            scaled_f = sc.transform(features)
            gpa_res = knn.predict(scaled_f)
            
            st.success(f"## Predicted GPA: {gpa_res[0]:.2f}")
            st.progress(min(gpa_res[0]/4.0, 1.0))

    else:
        st.write("### 🎯 Grade Classifier")
        gpa_val = st.number_input("Enter GPA for classification", 0.0, 4.0, 3.0)
        
        if st.button("PREDICT GRADE CLASS"):
            features_nb = np.array([[age, gender, ethnicity, parent_edu, study_time, absences, 
                                    tut, parent_supp, ext, spt, mus, vol, gpa_val]])
            grade_res = nb.predict(features_nb)
            
            grade_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'F'}
            final_grade = grade_map.get(int(grade_res[0]), "Unknown")
            
            # High-visibility colors for grades
            grade_colors = {'A': '#10B981', 'B': '#3B82F6', 'C': '#FACC15', 'D': '#F97316', 'F': '#EF4444'}
            current_color = grade_colors.get(final_grade, "#FFFFFF")
            
            st.markdown(f"""
                <div style="text-align:center; padding: 10px;">
                    <p style="font-size: 1.5rem; color: #94A3B8;">Student will likely achieve:</p>
                    <h1 style="font-size: 120px; color: {current_color} !important; margin: 0;">{final_grade}</h1>
                </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("<hr style='border: 0.5px solid #334155;'>", unsafe_allow_html=True)
st.caption("✅ Professional ML Deployment | High Contrast Version | Models: sc.pkl, nb.pkl, model.pkl")