import streamlit as st
import pandas as pd
import joblib
import base64

# Set page config
st.set_page_config(page_title="Diet Recommender", layout="centered")

# --- Background image setup only ---
def set_background(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }}

        .main-container {{
            background-color: rgba(255, 255, 255, 0.9);  /* Opaque white */
            padding: 2rem;
            border-radius: 20px;
            box-shadow: 0 0 25px rgba(0, 0, 0, 0.3);
            max-width: 700px;
            margin: 3rem auto;
            border: 1px solid #ccc;
        }}

        .stButton > button {{
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }}

        .stTextInput > div > div > input,
        .stNumberInput > div > input {{
            background-color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# --- Start and End container for content ---
def start_container():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

def end_container():
    st.markdown('</div>', unsafe_allow_html=True)

# Apply background styling
set_background("diet_bg.jpg")

# Start container
start_container()

# --- Load model and encoders ---
@st.cache_resource
def load_artifacts():
    scaler = joblib.load('scaler.pkl')
    model = joblib.load('diet_model.pkl')
    le_Calories = joblib.load('le_Calories.pkl')
    le_breakfast = joblib.load('le_breakfast.pkl')
    le_lunch = joblib.load('le_lunch.pkl')
    le_dinner = joblib.load('le_dinner.pkl')
    return scaler, model, le_Calories, le_breakfast, le_lunch, le_dinner

scaler, model, le_Calories, le_breakfast, le_lunch, le_dinner = load_artifacts()

# --- Title and inputs ---
st.title("🥗 Personalized Diet Recommendation System")
st.write("Enter your health details to receive a customized Indian meal plan:")

age = st.number_input('Age', min_value=1, max_value=120, value=25)
gender = st.selectbox('Gender', ['Male', 'Female'])
height_cm = st.number_input('Height (cm)', min_value=50, max_value=250, value=170)
weight_kg = st.number_input('Weight (kg)', min_value=10, max_value=200, value=65)
systolic_bp = st.number_input('Systolic Blood Pressure (mmHg)', min_value=50, max_value=200, value=120)
diastolic_bp = st.number_input('Diastolic Blood Pressure (mmHg)', min_value=30, max_value=140, value=80)
blood_sugar = st.number_input('Blood Sugar (mg/dL)', min_value=50, max_value=400, value=100)

# --- Button to predict ---
if st.button('Get Diet Plan'):
    input_df = pd.DataFrame([{
        'Age': age,
        'Gender': 1 if gender.lower() == 'male' else 0,
        'Height_cm': height_cm,
        'Weight_kg': weight_kg,
        'Systolic_BP': systolic_bp,
        'Diastolic_BP': diastolic_bp,
        'Blood_Sugar_mg_dL': blood_sugar,
    }])

    input_scaled = scaler.transform(input_df)
    predictions = model.predict(input_scaled)[0]

    calories = le_Calories.inverse_transform([predictions[0]])[0]
    breakfast = le_breakfast.inverse_transform([predictions[1]])[0]
    lunch = le_lunch.inverse_transform([predictions[2]])[0]
    dinner = le_dinner.inverse_transform([predictions[3]])[0]

    st.subheader("🎯 Your Personalized Diet Plan")
    st.write(f"**Estimated Calories:** ~{int(calories)} kcal")
    st.write(f"**Breakfast:** {breakfast}")
    st.write(f"**Lunch:** {lunch}")
    st.write(f"**Dinner:** {dinner}")

# End container
end_container()
