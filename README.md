# Smart Solar PV Diagnostic System (AI-Driven)

This project implements a hybrid AI-logic system to monitor and diagnose solar panel faults in real-time.

## 📁 Project Structure
- `main_live_system.py`: The production script for live ThingSpeak telemetry.
- `model_trainer.py`: The Machine Learning training script (Random Forest).
- `main_diagnostic.py`: Simulation and validation tool for fault scenarios.
- `data_generator.py`: Synthetic data generator for training and testing.

## 🛠️ Features
- **Fault Detection:** Identifies Soiling, Shading, Tracker Failures, and Disconnections.
- **AI-Powered:** Uses Random Forest to predict expected power based on irradiance and temperature.
- **Cloud Integrated:** Real-time data fetching via ThingSpeak API.

