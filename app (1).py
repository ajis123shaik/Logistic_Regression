import streamlit as st
import numpy as np
import pandas as pd
from joblib import load

# -------------------------------
# Load trained model
# -------------------------------
MODEL_FILE = "logistic_model.joblib"  # use .joblib now

try:
    model = load(MODEL_FILE)
except FileNotFoundError:
    st.error(f"❌ Model file '{MODEL_FILE}' not found! Make sure it is in the app folder.")
    st.stop()

# -------------------------------
# Streamlit App Config
# -------------------------------
st.set_page_config(page_title="🚢 Titanic Survival Prediction", layout="wide")
st.title("🚢 Titanic Survival Prediction")
st.markdown(
    """
    This app predicts **the probability of survival** for Titanic passengers.  
    Fill in the details in the sidebar and click **Predict**.
    """
)

# -------------------------------
# Sidebar - User Input
# -------------------------------
pclass = st.sidebar.selectbox("🎟 Passenger Class", [1, 2, 3], index=2)
sex = st.sidebar.radio("👤 Sex", ["Male", "Female"])
age = st.sidebar.slider("🎂 Age", 0, 80, 25)
sibsp = st.sidebar.number_input("👫 Siblings/Spouses Aboard", 0, 10, 0)
parch = st.sidebar.number_input("👪 Parents/Children Aboard", 0, 10, 0)
fare = st.sidebar.number_input("💰 Fare Paid", 0.0, 600.0, 32.0, step=1.0)
embarked = st.sidebar.selectbox("🛳 Port of Embarkation", ["C = Cherbourg", "Q = Queenstown", "S = Southampton"])

sex_val = 1 if sex == "Male" else 0
embarked_val = {"C = Cherbourg": 0, "Q = Queenstown": 1, "S = Southampton": 2}[embarked]

features = np.array([[pclass, sex_val, age, sibsp, parch, fare, embarked_val]])

# -------------------------------
# Prediction Section
# -------------------------------
if st.button("🔮 Predict Survival"):
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    st.subheader("📌 User Input Summary")
    st.table(pd.DataFrame({
        "Passenger Class": [pclass],
        "Sex": [sex],
        "Age": [age],
        "Siblings/Spouses": [sibsp],
        "Parents/Children": [parch],
        "Fare": [fare],
        "Embarked": [embarked]
    }))

    st.subheader("📝 Prediction Result")
    if prediction == 1:
        st.success("✅ Passenger is predicted to **Survive**")
    else:
        st.error("❌ Passenger is predicted **Not to Survive**")

    st.subheader("📊 Prediction Probabilities")
    st.write("**Legend:** 🟥 = Not Survived (0), 🟩 = Survived (1)")

    st.write(f"🟥 Not Survived: {probabilities[0]:.2%}")
    st.progress(int(probabilities[0]*100))

    st.write(f"🟩 Survived: {probabilities[1]:.2%}")
    st.progress(int(probabilities[1]*100))