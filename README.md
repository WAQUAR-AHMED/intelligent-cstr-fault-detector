# 🧠 Intelligent Fault Detection in CSTR using Deep Learning

This project presents an intelligent deep learning-based system for fault detection in a **Continuous Stirred Tank Reactor (CSTR)** — a common type of chemical reactor used in process industries. By leveraging time-series sensor data and advanced neural network models, we aim to detect and classify operational faults in real time, enabling predictive maintenance and improved process safety.

---

## 🔍 Motivation

Faults in chemical reactors can cause significant operational, economic, and safety hazards. Traditional threshold-based or statistical monitoring methods often fail in detecting nonlinear or subtle faults. In this project, we use deep learning to detect such anomalies from process data.

---

## ⚗️ About CSTR

A **Continuous Stirred Tank Reactor (CSTR)** is a vessel used in chemical engineering where a reaction takes place under constant agitation, and reactants are continuously fed into and products removed from the reactor.

### 🧪 Governing Equations

The dynamic behavior of a non-isothermal CSTR (single, exothermic reaction \( A \rightarrow B \)) can be modeled using the following ordinary differential equations:
![image](https://github.com/user-attachments/assets/7799259c-586c-484b-9c05-e519d8741431)



**********************Where:
• CA: Concentration of species A in reactor (mol/L)
• T : Reactor temperature (K)
• CA0: Inlet concentration of A (mol/L)
• T0: Inlet temperature (K)
• Tc: Coolant temperature (K)
• q: Inlet volumetric flow rate (L/min)
• V : Reactor volume (L)
• k0: Pre-exponential factor (1/min)
• E: Activation energy (J/mol)
• R: Gas constant (8.314 J/mol·K)
• ∆H: Heat of reaction (J/mol)
-• U : Heat transfer coefficient (J/min·m2·K)
-• A: Heat transfer area (m2)
---******

## 🛠️ Project Overview

| Feature                    | Details                                                      |
|----------------------------|--------------------------------------------------------------|
| 🧠 Model                   | LSTM / CNN-LSTM for sequence classification                  |
| 🧪 Dataset                 | Simulated fault data from a CSTR process (Kaggle)            |
| 🔎 Problem Type            | Multiclass time-series classification                        |
| 📈 Classes (Faults)        | Normal, Fault-1, Fault-2, ..., Fault-n                        |
| 🌐 Deployment              | Streamlit or Gradio app (optional, demo-ready)               |
| 🧪 Use Case                | Predictive maintenance & real-time fault detection           |

---

## ⚠️ Types of Faults Detected

Each fault represents an abnormal operating condition such as:
![image](https://github.com/user-attachments/assets/76f3b078-66b4-4e5a-8d57-87ca6d38fde1)

*Note: Faults vary depending on simulation settings from the dataset.*

---

## 🧠 Model Architecture

We use **LSTM-based deep neural networks** to capture temporal dependencies in sensor data:

- Input: Time-series vectors of sensor readings (e.g., \( T \), \( C_A \), \( T_c \), flow rate, etc.)
- Layers:
  - LSTM layer(s)
  - Dropout
  - Dense + Softmax (for classification)
- Output: Probability distribution over possible faults

---

## 📊 Performance Metrics

| Class        | Precision | Recall | F1-Score |
|--------------|-----------|--------|----------|
| Normal       | 0.91      | 0.95   | 0.93     |
| Fault 1      | 0.88      | 0.87   | 0.87     |
| Fault 2      | 0.95      | 0.90   | 0.92     |
| ...          | ...       | ...    | ...      |


        fault #   precision    recall  f1-score   support

         0.0       0.87      0.91      0.89        43
         1.0       1.00      0.87      0.93        46
         2.0       1.00      1.00      1.00        47
         3.0       1.00      1.00      1.00        35
         4.0       1.00      1.00      1.00        54
         5.0       1.00      1.00      1.00        49
         6.0       0.93      1.00      0.96        41
         7.0       1.00      1.00      1.00        45
         8.0       0.59      0.98      0.74        45
         9.0       1.00      0.85      0.92        48
        10.0       1.00      0.83      0.91        30
        11.0       0.70      0.85      0.76        46
        12.0       0.59      0.23      0.33        43

    accuracy                           0.89       572
    macro avg      0.90      0.89      0.88       572
    weighted avg   0.90      0.89      0.88       572


> You may also include training/validation loss plots, confusion matrices, and ROC curves.


##Confusion Matrix
![image](https://github.com/user-attachments/assets/48b657b8-d3e9-47ea-bbc7-14246f51a451)


---

## 📁 Project Structure

