# 🐔 Chicken Disease Classifier

A deep learning-based image classification system for identifying potential diseases in chickens, powered by a Convolutional Neural Network (CNN).  
This app uses a Streamlit frontend, backed by Docker, AWS ECR, and GitHub Actions for end-to-end MLOps deployment.

---

## 🚀 Features

- 📷 Upload a chicken image and classify it as **Healthy** or **Infected**
- 🧠 Trained CNN model with TensorFlow
- 📈 Visual confidence chart of prediction
- 💻 Streamlit-powered UI with custom styling
- 🐳 Dockerized deployment with GitHub Actions
- ☁️ CI/CD pipeline deploying to **AWS ECR + ECS**

---

## 🧰 Tech Stack

| Component | Tech |
|:--|:--|
| Model | CNN (TensorFlow / Keras) |
| UI | Streamlit |
| CI/CD | GitHub Actions |
| Containerization | Docker |
| Deployment | Amazon ECR (Elastic Container Registry) |
| Infra | AWS ECS, IAM (through GitHub Secrets) |

---

## 📸 Demo

<p align="center">
  <img src="assets/demo.png" width="600" alt="App Screenshot"/>
</p>

---

## 🧪 How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/nishantsingh-ds/chicken-disease-classifier.git
cd chicken-disease-classifier

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run Streamlit app
streamlit run app.py
