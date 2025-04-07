# 🧠 Disease Prediction and Medicine Recommendation using Machine Learning

This project is a **Machine Learning-based Disease Diagnosis and Medicine Recommendation System**. It uses user-reported symptoms to predict possible diseases and recommend appropriate medicines along with additional health information like precautions, recovery advice, and diet suggestions.

## 🚀 Features

- ✅ Predicts disease based on symptoms using multiple ML models (Decision Tree, Random Forest, Naïve Bayes, SVM)
- ✅ Majority voting system to finalize the most accurate disease prediction
- ✅ Recommends medicines based on the diagnosed disease using Collaborative Filtering
- ✅ Displays disease description, medication details, precautions, recovery tips, and diet suggestions
- ✅ Real-time prediction through a Flask-powered web interface

---

## 🧩 Tech Stack

- **Frontend**: React / Flutter *(Optional - can be integrated later)*
- **Backend**: Flask (Python)
- **ML Libraries**: Scikit-learn, Pandas, NumPy, TensorFlow *(for optimization)*
- **ML Models Used**:
  - Decision Tree
  - Random Forest
  - Naïve Bayes
  - Support Vector Machine (SVM)
  - Collaborative Filtering for medicine recommendation

---

## 📁 Dataset

The dataset includes:
- Disease names and symptoms
- Descriptions of diseases
- Recommended medicines
- Precautions and workout plans
- Recovery advice and diet suggestions

### Data Preprocessing
- Symptoms are encoded into numerical format using a separate symptom mapping.
- Trained models use either text-based or numerical input depending on the algorithm.

---

## ⚙️ How It Works

1. **User inputs symptoms**
2. **ML Models predict disease** using majority voting
3. **Collaborative Filtering** suggests medicines for the predicted disease
4. System displays:
   - Predicted disease name
   - Medicine name(s)
   - Disease description
   - Precautions & recovery advice
   - Diet and workout suggestions

---

## 🔧 Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/disease-prediction-ml.git
   cd disease-prediction-ml
2. Create a virtual environment and install dependencies:

   python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

3. Train the models:
   - Run the training script(s) in models/ folder (e.g. train_decision_tree.py, train_random_forest.py, etc.)
  
4. Open your browser and navigate to:
   http://127.0.0.1:5000

## 📬 Contact

For queries or feedback, contact:
📧 tamimakunji@gmail.com
   
