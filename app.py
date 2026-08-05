import streamlit as st
import pandas as pd
import joblib

# Load the saved models
@st.cache_resource # Caches the models so they aren't reloaded on every interaction
def load_models():
    rf_model = joblib.load('rf_model.joblib')
    scaler = joblib.load('scaler.joblib')
    interpolator = joblib.load('interpolator.joblib')
    return rf_model, scaler, interpolator

rf_model, scaler, interpolator = load_models()

def get_stability_limit(f, mu, e_bin):
    return f([[mu, e_bin]])[0]

st.title("Exoplanet Orbital Stability Predictor")
st.write("Predict the semimajor axis of a planet and evaluate its stability in a binary star system.")

# Example Inputs button
if st.button("Load Example Inputs"):
    st.session_state['star_1'] = 2.15
    st.session_state['star_2'] = 1.72
    st.session_state['period'] = 50.0
    st.session_state['e_bin'] = 0.01
    st.session_state['p_bin'] = 75.0
else:
    # Initialize session state if not present
    if 'star_1' not in st.session_state:
        st.session_state['star_1'] = 1.0
        st.session_state['star_2'] = 1.0
        st.session_state['period'] = 365.0
        st.session_state['e_bin'] = 0.5
        st.session_state['p_bin'] = 100.0

# Sidebar for User Inputs
st.sidebar.header("System Parameters")

star_1_mass = st.sidebar.number_input("Star 1 Mass (M_sun)", min_value=0.01, value=st.session_state['star_1'], step=0.1)
star_2_mass = st.sidebar.number_input("Star 2 Mass (M_sun)", min_value=0.01, value=st.session_state['star_2'], step=0.1)
planet_period = st.sidebar.number_input("Planet Orbital Period (days)", min_value=0.1, value=st.session_state['period'], step=1.0)
e_bin = st.sidebar.number_input("Binary Eccentricity (e_bin)", min_value=0.0, max_value=0.8, value=st.session_state['e_bin'], step=0.01)
p_bin = st.sidebar.number_input("Binary Period (days)", min_value=1.0, value=st.session_state['p_bin'], step=1.0)

st.header("Evaluation Results")

# 1. Processed user inputs
M_tot = star_1_mass + star_2_mass
mu = star_1_mass / (star_1_mass + star_2_mass)
a_bin = (p_bin**2 * M_tot)**(1./3)

# 2. SMA Evaluation Model (Random Forest)
# Columns must match the exact feature names used in training ("st_mass", "pl_orbper")
user_input_df = pd.DataFrame([[star_1_mass, planet_period]], columns=["st_mass", "pl_orbper"])
user_input_scaled = scaler.transform(user_input_df)
a_predicted = rf_model.predict(user_input_scaled)[0]

# 3. CSP Stability Model (Interpolator)
a_c = get_stability_limit(interpolator, mu, e_bin) * a_bin
pl_valid = a_c > a_predicted

# Display metrics
col1, col2 = st.columns(2)
col1.metric("Predicted Semimajor Axis", f"{a_predicted:.4f} AU")
col2.metric("Critical Stability Limit (a_c)", f"{a_c:.4f} AU")

st.subheader("Stability Conclusion")
if pl_valid:
    st.success("This orbit is stable! The predicted semimajor axis is inside the critical stability limit.")
else:
    st.error("This orbit is unstable! The predicted semimajor axis exceeds the critical stability limit.")