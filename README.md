# ML_Supervised_Shipment
# Shipment Delivery Prediction

## Project Overview

This machine learning project predicts whether a shipment will reach its destination on time based on shipment, customer, and product-related features.

The project follows a complete machine learning workflow including data preprocessing, model training, evaluation, hyperparameter tuning, feature importance analysis, model persistence, and deployment using Streamlit.

---

## Business Problem

Late deliveries can negatively impact customer satisfaction and logistics efficiency. The objective of this project is to build a classification model that predicts shipment delays and helps businesses make proactive decisions.

---

## Dataset Features

The dataset contains the following features:

* ID
* Warehouse Block
* Mode of Shipment
* Customer Care Calls
* Customer Rating
* Cost of Product
* Prior Purchases
* Product Importance
* Gender
* Discount Offered
* Weight in Grams

### Target Variable

* **Reached.on.Time_Y.N**

  * 0 → Reached on time
  * 1 → Delayed

---

## Project Workflow

### 1. Data Preprocessing

* Missing value handling
* Numerical feature scaling
* Categorical feature encoding
* Scikit-Learn preprocessing pipeline

### 2. Train-Test Split

* 80% Training Data
* 20% Testing Data
* Stratified sampling

### 3. Model Training

The following classification models were trained:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* Gradient Boosting Classifier
* AdaBoost Classifier

### 4. Model Evaluation

Evaluation metrics used:

* Accuracy
* Precision
* Recall
* F1 Score

### 5. Hyperparameter Tuning

GridSearchCV was used to optimize the best-performing models based on F1 Score.

### 6. Feature Importance

Permutation Importance was used to identify the most influential features affecting delivery predictions.

### 7. Model Deployment

The final model was deployed using Streamlit for interactive predictions.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Joblib
* Streamlit
* Jupyter Notebook

---

## Project Structure

```text
Shipment_Delivery_Prediction/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── notebooks/
│   ├── eda.ipynb
│   └── model_training.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── tuning.py
│   └── utils.py
│
├── model.pkl
├── app.py
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd Shipment_Delivery_Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Application

Launch the Streamlit application:

```bash
streamlit run app.py
```

Then open the URL displayed in the terminal.

---

## Results

* Multiple machine learning models were compared.
* Hyperparameter tuning improved model performance.
* The final model was selected based on F1 Score.
* A Streamlit application was developed for real-time predictions.

---

## Future Improvements

* Deploy on Streamlit Cloud
* Add XGBoost and LightGBM models
* Implement model monitoring
* Build automated retraining pipelines

---

## Author

**Mohammed Ameen**

Machine Learning & Data Science Enthusiast

---

## License

This project is created for educational and portfolio purposes.
