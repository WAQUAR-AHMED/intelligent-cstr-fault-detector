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

**1. Material Balance (for reactant A):**

\[
\frac{dC_A}{dt} = \frac{F}{V}(C_{A0} - C_A) - k_0 e^{-\frac{E}{RT}} C_A
\]

**2. Energy Balance:**

\[
\frac{dT}{dt} = \frac{F}{V}(T_{0} - T) + \frac{(-\Delta H)}{\rho C_p} k_0 e^{-\frac{E}{RT}} C_A - \frac{UA}{\rho C_p V}(T - T_c)
\]

Where:
- \( C_A \): Concentration of A (mol/L)
- \( T \): Reactor temperature (K)
- \( T_c \): Coolant temperature (K)
- \( F \): Inlet flow rate (L/min)
- \( V \): Reactor volume (L)
- \( k_0 \): Pre-exponential factor (1/min)
- \( E \): Activation energy (J/mol)
- \( R \): Gas constant (8.314 J/mol·K)
- \( \Delta H \): Heat of reaction (J/mol)
- \( \rho \), \( C_p \): Density and specific heat of fluid
- \( UA \): Heat transfer coefficient × area

---

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

- **Fault 1**: Sudden change in inlet temperature
- **Fault 2**: Valve failure affecting coolant flow rate
- **Fault 3**: Sensor drift in reactor temperature
- **Fault 4**: Heat exchanger fouling (UA drop)
- **Fault 5**: Catalyst deactivation (drop in \( k_0 \))
- **Normal**: Stable operation with expected input/output

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

> You may also include training/validation loss plots, confusion matrices, and ROC curves.

---

## 📁 Project Structure

