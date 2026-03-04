# AI-Powered Solar PV Fault Diagnosis System ☀️🔋

An advanced diagnostic framework that leverages **Hybrid Machine Learning** (Random Forest + KNN) and **Electrical Voltage Profiling** to detect and classify faults in Photovoltaic systems in real-time.

## 🚀 Overview
Unlike traditional monitoring, this system doesn't just look at power output; it analyzes the relationship between **Open-Circuit Voltage ($V_{oc}$)** and **Load Voltage ($V_{load}$)**. This allows the system to distinguish between environmental issues (shading/dust) and electrical failures (high resistance/wiring defects).

## 🧠 Hybrid AI Methodology
The system employs a dual-model validation approach:
1.  **Primary Model (Random Forest):** Handles non-linear relationships between Irradiance, Temperature, and Power.
2.  **Validator Model (KNN):** Cross-checks the prediction against historical data points to ensure high confidence.

## 🛠 Features
- **$V_{oc}$ vs $V_{load}$ Analysis:** Detects critical voltage drops and wiring resistance.
- **Smart Fault Classification:** - **Healthy:** Optimal operation.
    - **Soiling/Dust:** Significant current drop with stable $V_{oc}$.
    - **Shading/Damage:** Significant drop in both $V_{oc}$ and Power.
    - **Wiring Fault:** High $V_{drop}$ ratio ($V_{oc} - V_{load}$).
- **IoT Ready:** Designed to fetch live data from **ThingSpeak** via ESP32.

## 📂 Project Structure
| File | Description |
| :--- | :--- |
| `data_generator.py` | Generates a synthetic dataset based on real PV cell physics including $V_{oc}$ & $V_{load}$. |
| `model_trainer.py` | Trains and exports the Hybrid AI models (`.pkl` files). |
| `main_diagnostic.py` | The "Brain" that processes sensor data and outputs real-time diagnosis. |

## 📊 How it Works
1.  **Data Acquisition:** Sensors (Current, Voltage, LDR, DHT11) send data to ThingSpeak.
2.  **AI Prediction:** The system predicts the "Ideal Power" for current weather conditions.
3.  **Deviation Analysis:** If (Actual Power < Predicted Power), the diagnostic logic identifies the root cause based on the **Voltage Drop Ratio**.

## 🛠 Installation & Usage
1. Clone the repository.
2. Run `data_generator.py` to create the dataset.
3. Run `model_trainer.py` to train the RF and KNN models.
4. Run `main_diagnostic.py` to start the live monitoring.

---
**Developed with Wit & Logic.** 🚀

