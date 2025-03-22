#import the libraries
import math
import numpy as np
import pickle
import streamlit as st
import pandas as pd

#SET PAGE WIDE
st.set_page_config(
    page_title='IPL Score Predictor',
    layout="wide",
    initial_sidebar_state="collapsed"
)

#Get the ML model and label encoders
filename = 'ml_model.pkl'
encoders_filename = 'label_encoders.pkl'
try:
    model = pickle.load(open(filename,'rb'))
    encoders = pickle.load(open(encoders_filename,'rb'))
except FileNotFoundError:
    st.error("Model files not found. Please run train_model.py first to train the model.")
    st.stop()

# Add custom CSS and JavaScript for better UI
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), 
                    url("https://4.bp.blogspot.com/-F6aZF5PMwBQ/Wrj5h204qxI/AAAAAAAABao/4QLn48RP3x0P8Ry0CcktxilJqRfv1IfcACLcBGAs/s1600/GURU%2BEDITZ%2Bbackground.jpg");
        background-attachment: fixed;
        background-size: cover;
        font-family: 'Poppins', sans-serif;
    }
    
    .main-title {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3em;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
        padding: 20px;
    }
    
    .custom-div {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 30px;
        margin: 20px 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .team-select {
        background: rgba(0, 0, 0, 0.5);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    /* Style for all text elements */
    .element-container {
        color: white !important;
    }
    
    /* Style for labels and text */
    label, .stTextInput > label, .stNumberInput > label, .stSelectbox > label {
        color: #FFFFFF !important;
        font-weight: 500 !important;
        font-size: 1.1em !important;
        margin-bottom: 8px !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    /* Style for selectbox */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.9) !important;
        border-radius: 8px !important;
        border: none !important;
        color: #000000 !important;
        font-weight: 500;
    }
    
    /* Style for number input */
    .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.9) !important;
        border-radius: 8px !important;
        border: none !important;
        color: #000000 !important;
        font-weight: 500;
        padding: 8px 12px !important;
    }
    
    /* Style for slider */
    .stSlider > div > div > div {
        color: #4ECDC4 !important;
    }
    
    /* Style for button */
    .stButton > button {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 12px 24px !important;
        border-radius: 8px !important;
        transition: transform 0.2s !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
    }
    
    /* Style for expander */
    .streamlit-expanderContent {
        background: rgba(0, 0, 0, 0.5) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Style for success messages */
    .stSuccess {
        background: linear-gradient(90deg, rgba(40, 167, 69, 0.8), rgba(46, 204, 113, 0.8)) !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 16px !important;
        text-align: center !important;
        font-weight: 600 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    /* Style for error messages */
    .stError {
        background: linear-gradient(90deg, rgba(220, 53, 69, 0.8), rgba(231, 76, 60, 0.8)) !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 16px !important;
        text-align: center !important;
        font-weight: 600 !important;
    }
    
    .custom-info {
        color: #4ECDC4 !important;
        background: rgba(78, 205, 196, 0.1) !important;
        border: 1px solid #4ECDC4 !important;
        border-radius: 10px !important;
        padding: 16px !important;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2em;
        }
        .custom-div {
            padding: 20px;
        }
    }
    </style>
    
    <script>
    // Add smooth scrolling
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });
    </script>
""", unsafe_allow_html=True)

# Title with custom styling
st.markdown('<h1 class="main-title">IPL Score Predictor 2022</h1>', unsafe_allow_html=True)

# Description in a custom styled container
with st.expander("ℹ️ About this app"):
    st.markdown("""
    <div class="custom-info">
    This ML-powered application predicts IPL match scores with high accuracy. For reliable predictions:
    <ul>
        <li>Minimum 5 overs should be completed</li>
        <li>Input current match statistics accurately</li>
        <li>Consider recent team performance</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Create a custom container for the form
st.markdown('<div class="custom-div">', unsafe_allow_html=True)

# Teams selection in a custom styled section
st.markdown('<div class="team-select">', unsafe_allow_html=True)
teams = ['Chennai Super Kings', 'Delhi Daredevils', 'Kings XI Punjab',
         'Kolkata Knight Riders', 'Mumbai Indians', 'Rajasthan Royals',
         'Royal Challengers Bangalore', 'Sunrisers Hyderabad']

col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('🏏 Select Batting Team', teams)
with col2:
    bowling_team = st.selectbox('🎯 Select Bowling Team', teams)

if bowling_team == batting_team:
    st.error('⚠️ Bowling and Batting teams must be different!')
    st.stop()

st.markdown('</div>', unsafe_allow_html=True)

# Match statistics input
st.markdown('<div class="match-stats">', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    overs = st.number_input('🕒 Current Over', min_value=5.1, max_value=19.5, value=5.1, step=0.1)
    if overs-math.floor(overs)>0.5:
        st.error('⚠️ Invalid over format! One over has 6 balls only.')
with col2:
    runs = st.number_input('🏃 Current Runs', min_value=0, max_value=354, step=1, format='%i')

# Wickets with custom styling
wickets = st.slider('🎯 Wickets Fallen', 0, 9, help='Slide to select number of wickets')

col3, col4 = st.columns(2)
with col3:
    runs_in_prev_5 = st.number_input('📈 Runs (Last 5 Overs)', min_value=0, max_value=runs, step=1, format='%i')
with col4:
    wickets_in_prev_5 = st.number_input('📉 Wickets (Last 5 Overs)', min_value=0, max_value=wickets, step=1, format='%i')

st.markdown('</div>', unsafe_allow_html=True)

# Prediction section
input_df = pd.DataFrame({
    'bat_team': [batting_team],
    'bowl_team': [bowling_team],
    'runs': [runs],
    'wickets': [wickets],
    'overs': [overs],
    'runs_last_5': [runs_in_prev_5],
    'wickets_last_5': [wickets_in_prev_5]
})

# Transform teams using label encoders
input_df['bat_team'] = encoders['bat_team'].transform(input_df['bat_team'])
input_df['bowl_team'] = encoders['bowl_team'].transform(input_df['bowl_team'])

# Center the predict button
col1, col2, col3 = st.columns([1,2,1])
with col2:
    predict_button = st.button('🎯 Predict Score')

if predict_button:
    prediction = model.predict(input_df)
    my_prediction = int(round(prediction[0]))
    
    # Display prediction with custom styling
    st.markdown(f"""
        <div style='text-align: center; padding: 20px;'>
            <h2 style='color: #4ECDC4; margin-bottom: 10px;'>Prediction Results</h2>
            <div class='custom-div' style='display: inline-block; padding: 20px 40px;'>
                <h3 style='color: white; margin: 0;'>Predicted Score Range</h3>
                <h1 style='color: #FF6B6B; margin: 10px 0;'>{my_prediction-5} - {my_prediction+5}</h1>
                <p style='color: #4ECDC4; margin: 0;'>runs</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style='text-align: center; padding: 20px; color: rgba(255,255,255,0.5);'>
        Made with ❤️ for Cricket Fans
    </div>
""", unsafe_allow_html=True)