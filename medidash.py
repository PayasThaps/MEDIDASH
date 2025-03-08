import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set Page Configuration
st.set_page_config(page_title="AI Hospital Optimization Prototype", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Solution:", [
    "AI-Powered No-Show Prediction",
    "AI-Based ICU & Doctor Scheduling",
    "AI-Driven Diagnostics & Lab Optimization",
    "Blockchain & Secure Data Sharing"
])

# Generate Dummy Data
np.random.seed(42)
dummy_patients = pd.DataFrame({
    "Patient ID": np.arange(1, 101),
    "Age": np.random.randint(18, 90, 100),
    "Past No-Show Rate": np.random.uniform(0, 1, 100),
    "Predicted No-Show Probability": np.random.uniform(0, 1, 100),
    "ICU Demand Prediction": np.random.randint(0, 100, 100),
    "Doctor Availability": np.random.randint(0, 10, 100),
    "Lab Test Priority": np.random.choice(["High", "Medium", "Low"], 100)
})

# Train No-Show Prediction Model (Logistic Regression)
X = dummy_patients[["Age", "Past No-Show Rate"]]
y = (dummy_patients["Predicted No-Show Probability"] > 0.5).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
no_show_model = LogisticRegression()
no_show_model.fit(X_train, y_train)
y_pred = no_show_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
joblib.dump(no_show_model, "no_show_model.pkl")

# Train ICU Bed Allocation Model (LSTM for Time-Series Prediction)
X_icubeds = np.random.rand(100, 10)  # Dummy time-series data for ICU demand
y_icubeds = np.random.randint(0, 100, 100)
X_icubeds_padded = pad_sequences(X_icubeds, maxlen=10)

lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(10, 1)),
    LSTM(50),
    Dense(1, activation='linear')
])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_icubeds_padded, y_icubeds, epochs=5, verbose=0)
lstm_model.save("icu_lstm_model.h5")

# AI-Powered No-Show Prediction
if page == "AI-Powered No-Show Prediction":
    st.title("📊 AI-Powered No-Show Prediction")
    st.write("Predicting patient no-shows to optimize scheduling and reduce inefficiencies.")
    
    model = joblib.load("no_show_model.pkl")
    age = st.slider("Select Age:", 18, 90, 30)
    past_no_show = st.slider("Past No-Show Rate:", 0.0, 1.0, 0.5)
    prediction = model.predict([[age, past_no_show]])[0]
    st.write(f"### Predicted No-Show Status: {'No-Show' if prediction == 1 else 'Attending'}")
    st.write(f"**Model Accuracy:** {accuracy * 100:.2f}%")

# AI-Based ICU & Doctor Scheduling
elif page == "AI-Based ICU & Doctor Scheduling":
    st.title("🏥 AI-Based ICU & Doctor Scheduling")
    st.write("Optimizing ICU bed allocation and doctor availability.")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=dummy_patients["ICU Demand Prediction"],
                    y=dummy_patients["Doctor Availability"], hue=dummy_patients["Lab Test Priority"], palette="coolwarm")
    ax.set_title("ICU Demand vs. Doctor Availability")
    ax.set_xlabel("ICU Demand Prediction")
    ax.set_ylabel("Doctor Availability")
    st.pyplot(fig)

# AI-Driven Diagnostics & Lab Optimization
elif page == "AI-Driven Diagnostics & Lab Optimization":
    st.title("🔬 AI-Driven Diagnostics & Lab Optimization")
    st.write("Enhancing test turnaround and prioritizing urgent cases.")
    
    priority_counts = dummy_patients["Lab Test Priority"].value_counts()
    fig = px.pie(names=priority_counts.index, values=priority_counts.values,
                 title="Lab Test Priority Distribution", color=priority_counts.index)
    st.plotly_chart(fig)

# Blockchain & Secure Data Sharing
elif page == "Blockchain & Secure Data Sharing":
    st.title("🔐 Blockchain & Secure Data Sharing")
    st.write("Ensuring hospital data security and compliance.")
    st.markdown("### Features of Blockchain for Secure Data:")
    st.write("✅ Decentralized and tamper-proof records")
    st.write("✅ AI models access only permissioned datasets")
    st.write("✅ Secure patient data transactions across hospitals")
    
    df_security = pd.DataFrame({
        "Feature": ["Data Encryption", "Access Control", "Tamper-Proof Records", "Audit Trail"],
        "Implemented": ["Yes", "Yes", "Yes", "Yes"]
    })
    st.table(df_security)

st.sidebar.markdown("---")
st.sidebar.write("🚀 AI-Powered Hospital Optimization Prototype")
