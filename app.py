import streamlit as st
import pickle
import pandas as pd

# --- Page Config ---
st.set_page_config(
    page_title="House Price Predictor | Lipsa Mahakul",
    page_icon="🏠",
    layout="centered"
)

# --- Custom CSS for Decoration ---
st.markdown("""
<style>
   .main {
        background-color: #f8f9ff;
    }
   .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
   .css-1d391kg {
        background: white;
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
   .prediction-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        border: 2px solid #ff9a76;
        margin-top: 20px;
    }
   .metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        text-align: center;
    }
    h1 {
        color: white!important;
        text-align: center;
        font-family: 'Poppins', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Model ---
@st.cache_resource
def load_model():
    with open('house_price_model.pkl', 'rb') as f:
        return pickle.load(f)

try:
    model = load_model()
except:
    st.error("Model file not found! Please first run the code to save house_price_model.pkl")
    st.stop()

# --- Header ---
st.markdown("<h1>🏠 House Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white; font-size: 18px;'>Enter your dream house details and get instant price prediction<br>Built by <b>Lipsa Mahakul</b></p>", unsafe_allow_html=True)

# --- Main Card ---
with st.container():
    st.markdown("### 📋 Enter House Details")

    col1, col2 = st.columns(2)

    with col1:
        area = st.slider("📐 Area (sq ft)", 500, 10000, 2500, step=100, help="Total area of house")
        bedrooms = st.selectbox("🛏️ Bedrooms", [1,2,3,4,5,6], index=2)
        bathrooms = st.selectbox("🚿 Bathrooms", [1,2,3,4,5], index=1)

    with col2:
        stories = st.selectbox("🏢 Stories / Floors", [1,2,3,4], index=1)
        parking = st.selectbox("🚗 Parking Spaces", [0,1,2,3], index=1)
        st.markdown(" ")
        st.markdown(" ")
        predict_btn = st.button("💰 PREDICT PRICE", use_container_width=True, type="primary")

# --- Prediction ---
if predict_btn:
    # IMPORTANT: Keep same order as training: area, bedrooms, bathrooms, stories, parking
    input_df = pd.DataFrame({
        'area': [area],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'stories': [stories],
        'parking': [parking]
    })

    prediction = model.predict(input_df)[0]

    st.markdown(f"""
    <div class="prediction-box">
        <h2 style='color: #333; margin:0;'>🎉 Predicted Price</h2>
        <h1 style='color: #d63031!important; font-size: 42px; margin:10px 0;'>₹ {prediction:,.0f}</h1>
        <p style='color: #636e72; font-size: 16px;'>For {area} sq ft, {bedrooms}BHK House</p>
    </div>
    """, unsafe_allow_html=True)

    # Show metrics
    st.markdown("### 📊 Your House Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Price / sq ft", f"₹ {prediction/area:,.0f}")
    c2.metric("Total Rooms", f"{bedrooms + bathrooms}")
    c3.metric("Model R² Score", "0.53")

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: white;'>Made with ❤️ using Python, Scikit-Learn & Streamlit | <a href='https://github.com/LipsaMahakul' style='color:white;'>GitHub: LipsaMahakul</a></p>", unsafe_allow_html=True)