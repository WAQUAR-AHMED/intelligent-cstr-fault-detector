# 🧠 Intelligent Fault Detection in CSTR using Deep Learning

A deep learning-based intelligent system to detect and classify faults in a Continuous Stirred Tank Reactor (CSTR). This project combines Chemical Engineering knowledge and modern machine learning to monitor reactor performance and detect anomalies in real time.

---

## 📌 Project Highlights

- 📊 Dataset: CSTR fault dataset from Kaggle (sensor-based process data)
- 🤖 Model: LSTM / CNN-LSTM / other deep architectures
- 🎯 Task: Multiclass fault classification (Normal, Fault-1, Fault-2, etc.)
- 🛠️ Technologies: Python, TensorFlow/Keras (or PyTorch), Pandas, NumPy, Scikit-learn
- 🌐 Optional Web Interface: Streamlit / Gradio

---

## 📁 Directory Structure
├── model/ # Trained model files
├── src/ # Model training, evaluation, and prediction code
├── app/ # Optional web app for demo
├── data/ # Sample or link to dataset
├── requirements.txt # Dependencies
└── README.md


---

## ⚙️ Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/cstr-fault-detection-deeplearning.git
   cd cstr-fault-detection-deeplearning

2.**Install Dependencies**
pip install -r requirements.txt

3.**Run Training**
python src/train.py

4.**Launch Demo App (Optional)**
streamlit run app/app.py

![image](https://github.com/user-attachments/assets/ba7ca656-ef05-4c80-b2f5-08279a1a104c)


📎 Dataset
Source: Kaggle CSTR Fault Dataset ← (put the actual link)

Sensor features like temperature, pressure, concentration, flow rates, etc.

📌 Applications
Industrial process monitoring

Chemical plant automation

Fault tolerance and predictive maintenance

🙌 Acknowledgements
Kaggle community

TensorFlow/PyTorch docs

Academic references on process control and soft sensors

📬 Contact
For collaborations or suggestions: [your_email@example.com]
