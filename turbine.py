import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.metrics import mean_squared_error

# Set Streamlit page configuration
st.set_page_config(page_title="Wind Turbine Anomaly Detection", layout="wide")

# Load Dataset
def load_data():
    uploaded_file = st.file_uploader("Upload your wind turbine dataset (CSV)", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.write(data.head())
        return data
    else:
        st.warning("Please upload a dataset.")
        return None

# Preprocess Data
def preprocess_data(data, features):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[features])
    return data_scaled, scaler

# Build Autoencoder Model
def build_autoencoder(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(32, activation='relu'),
        Dense(64, activation='relu'),
        Dense(input_dim, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Train Autoencoder
def train_autoencoder(data, model, epochs=50, batch_size=32):
    history = model.fit(data, data, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)
    return history

# Detect Anomalies
def detect_anomalies(data, model, threshold=0.02):
    reconstructed = model.predict(data)
    loss = np.mean(np.square(data - reconstructed), axis=1)
    anomalies = loss > threshold
    return anomalies, loss

# Visualize Results
def visualize_anomalies(data, anomalies, loss):
    plt.figure(figsize=(10, 6))
    plt.plot(loss, label='Reconstruction Loss')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    plt.title("Anomaly Detection")
    plt.xlabel("Sample Index")
    plt.ylabel("Reconstruction Loss")
    plt.legend()
    st.pyplot(plt)

# Streamlit UI
def main():
    st.title("Advanced Anomaly Detection in Wind Turbine Operations")

    # Load Dataset
    data = load_data()
    if data is not None:
        features = st.multiselect("Select features for anomaly detection", options=data.columns)
        if features:
            data_scaled, scaler = preprocess_data(data, features)

            # Train Autoencoder
            st.write("### Training Autoencoder...")
            autoencoder = build_autoencoder(data_scaled.shape[1])
            epochs = st.slider("Select number of epochs", 10, 100, 50)
            batch_size = st.slider("Select batch size", 16, 128, 32)
            history = train_autoencoder(data_scaled, autoencoder, epochs, batch_size)

            # Detect Anomalies
            threshold = st.slider("Set anomaly threshold", 0.01, 0.1, 0.02)
            anomalies, loss = detect_anomalies(data_scaled, autoencoder, threshold)

            # Display Results
            st.write("### Anomaly Detection Results")
            data['Anomaly'] = anomalies
            st.write(data)

            # Visualize Anomalies
            visualize_anomalies(data, anomalies, loss)

if __name__ == "__main__":
    main()
