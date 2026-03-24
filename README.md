# 🚀 Customer Churn Prediction (with Code, Charts & Insights)

![Banner](https://images.unsplash.com/photo-1551288049-bebda4e38f71?q=80\&w=1600\&auto=format\&fit=crop)

---

## 📌 Project Overview

This project predicts **customer churn** using machine learning and provides deep insights through **visualizations and data analysis**.

The goal is to help businesses understand:

* Why customers leave
* Which customers are at risk
* How to improve retention

---

## 📊 Dataset Summary

* 👥 Total Customers: **7043**
* 📌 Features: **21 columns**
* 🎯 Target: **Churn (Yes / No)**

---

## 🧹 Data Preprocessing (Code Included)

```python
# Load dataset
data = pd.read_csv("data.csv")

# Drop unnecessary column
data.drop("customerID", axis=1, inplace=True)

# Handle missing values
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data.drop(labels=data[data["tenure"] == 0].index, axis=0, inplace=True)
data.fillna(data["TotalCharges"].mean(), inplace=True)

# Convert SeniorCitizen to categorical
data['SeniorCitizen'] = data['SeniorCitizen'].map({0: "No", 1: "Yes"})
```

---

## 📈 Exploratory Data Analysis (Charts + Code)

### 🔥 1. Churn Distribution

```python
import plotly.graph_objects as go

fig = go.Figure(data=[go.Pie(
    labels=data['Churn'].value_counts().index,
    values=data['Churn'].value_counts().values,
    hole=0.4)])
fig.update_layout(title="Churn Distribution")
fig.show()
```

### 💡 Insight:

* Majority customers **do not churn**
* But a significant portion still leaves → retention needed

---

### 📊 2. Churn vs Contract Type

```python
import plotly.express as px

fig = px.histogram(data, x="Churn", color="Contract", barmode="group",
                   title="Churn by Contract Type")
fig.show()
```

### 💡 Insight:

* Customers with **monthly contracts churn more**
* Long-term contracts increase retention

---

### 💸 3. Monthly Charges Distribution

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.kdeplot(data.MonthlyCharges[data["Churn"] == 'No'], label="No Churn")
sns.kdeplot(data.MonthlyCharges[data["Churn"] == 'Yes'], label="Churn")
plt.legend()
plt.title("Monthly Charges vs Churn")
plt.show()
```

### 💡 Insight:

* Higher **monthly charges → higher churn probability**

---

### ⏳ 4. Tenure vs Churn

```python
fig = px.box(data, x='Churn', y='tenure', title="Tenure vs Churn")
fig.show()
```

### 💡 Insight:

* **New customers churn more**
* Long-term customers are more loyal

---

## 🔍 Feature Engineering

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])
```

---

## ⚙️ Model Building (Code Included)

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = data.drop("Churn", axis=1)
y = data["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

---

## 🤖 Models Used

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}
```

---

## 📊 Model Evaluation

```python
from sklearn.metrics import accuracy_score

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(name, "Accuracy:", accuracy_score(y_test, preds))
```

---

## 🏆 Final Results

| Model             | Accuracy | ROC-AUC |
| ----------------- | -------- | ------- |
| Voting Classifier | ⭐ 79.95% | 84.82   |
| Gradient Boosting | 79.36%   | 84.63   |
| AdaBoost          | 79.93%   | 84.39   |

---

## 📉 Key Business Insights

✔ Customers with **monthly contracts are high-risk**
✔ High **monthly charges increase churn**
✔ Customers without **tech support churn more**
✔ **New users are most likely to leave**

---

## 🚀 Tech Stack

* Python 🐍
* Pandas, NumPy
* Matplotlib, Seaborn, Plotly
* Scikit-learn, XGBoost, CatBoost

---

## ▶️ How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn plotly xgboost catboost
jupyter notebook
```

---

## 🌟 Future Improvements

* Hyperparameter tuning (GridSearchCV)
* Deployment (Streamlit / Flask)
* Real-time prediction system
* Deep learning models

---

## 💡 Conclusion

This project demonstrates how machine learning + data visualization can help businesses:

* Reduce churn 📉
* Improve retention 📈
* Increase revenue 💰

---

## 🙌 Support

If you found this useful:
⭐ Star this repo

💬 Give feedback

---

![Footer](https://images.unsplash.com/photo-1504384308090-c894fdcc538d?q=80\&w=1600)
