import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Load Data with caching for performance optimization
@st.cache_data
def load_data():
    data = pd.read_csv('diabetes.csv')  # Make sure the 'diabetes.csv' is in the same directory
    return data

# Load data
data = load_data()

# Fix: Convert 'Outcome' column to numeric and handle NaN
data['Outcome'] = pd.to_numeric(data['Outcome'], errors='coerce')
data['Outcome'].fillna(0, inplace=True)  # Assuming 0 means no diabetes

# App Title and Intro
st.title("🩺 Diabetes Prediction App")
st.markdown("""
    This app helps visualize and predict diabetes using the **PIMA Indian Diabetes Dataset**. 
    The k-Nearest Neighbors (kNN) model is used to classify whether a person has diabetes or not based on several factors like glucose levels, age, and BMI.
""")

# Show Dataset Info
st.header("📊 Dataset Overview")
st.write("Here is a quick look at the dataset and some basic statistics:")

# Show data preview
st.dataframe(data.head())

# Show basic stats and missing values
st.write("### Basic Statistics of Dataset:")
st.write(data.describe())

st.write("### Checking for Missing Values:")
st.write(data.isna().sum())

# Display Count Plot of Outcome
st.header("🔍 Distribution of Diabetes Outcomes")
st.write("""
    The **Outcome** column indicates whether a person has diabetes (1) or not (0).
    Let's visualize the distribution of outcomes.
""")
fig, ax = plt.subplots(figsize=(12,6))
sns.countplot(x='Outcome', data=data, ax=ax)
ax.set_title('Diabetes Outcome Count')
st.pyplot(fig)

# Boxplots of all features
st.header("📦 Feature Boxplots")
st.write("""
    Boxplots help visualize the spread of data and detect outliers for each feature. Below are the boxplots for various features.
""")
fig, axes = plt.subplots(3, 3, figsize=(12,12))
axes = axes.flatten()
for i, col in enumerate(['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']):
    sns.boxplot(x=col, data=data, ax=axes[i])
plt.tight_layout()
st.pyplot(fig)

# Pairplot for feature correlation
st.header("👯‍♂️ Pairplot of Features")
st.write("""
    Pairplots show relationships between different features. Here we plot them based on diabetes outcomes.
""")
fig = sns.pairplot(data=data, hue='Outcome')
st.pyplot(fig)

# Histograms of features
st.header("📊 Distribution of Features")
st.write("""
    Below, you can see the distribution of each feature using histograms with KDE (Kernel Density Estimate).
""")
fig, axes = plt.subplots(3, 3, figsize=(12,12))
axes = axes.flatten()
for i, col in enumerate(['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']):
    sns.histplot(x=col, data=data, kde=True, ax=axes[i])
plt.tight_layout()
st.pyplot(fig)

# Correlation heatmap
st.header("🧠 Correlation Heatmap")
st.write("""
    This heatmap shows how features are correlated with each other. Higher correlation means stronger relationships.
""")
fig, ax = plt.subplots(figsize=(12,12))
sns.heatmap(data.corr(), vmin=-1.0, center=0, cmap='RdBu_r', annot=True, ax=ax)
st.pyplot(fig)

# Feature Scaling and Model Training
st.header("⚙️ Data Preprocessing and Model Training")

# Standardize features
sc_X = StandardScaler()
X = pd.DataFrame(sc_X.fit_transform(data.drop(['Outcome'], axis=1)),
                 columns=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])
y = data['Outcome']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Model training using kNN
k_values = st.slider("Choose the value of k for kNN", 1, 15, 5)
knn = KNeighborsClassifier(k_values)
knn.fit(X_train, y_train)
train_accuracy = knn.score(X_train, y_train)
test_accuracy = knn.score(X_test, y_test)

st.write(f"Training Accuracy for k={k_values}: {train_accuracy * 100:.2f}%")
st.write(f"Test Accuracy for k={k_values}: {test_accuracy * 100:.2f}%")

# Confusion Matrix
st.header("📉 Confusion Matrix")
y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay(cm).plot(cmap='Blues', ax=ax)
st.pyplot(fig)

# Classification Report
st.header("📈 Classification Report")
st.write(classification_report(y_test, y_pred))

# Footer and Conclusion
st.markdown("""
    ## 🎯 Conclusion:
    - The app shows how we can predict diabetes based on various features using k-Nearest Neighbors.
    - You can experiment with different values of k to observe how the model's accuracy changes.
    - The confusion matrix and classification report give a detailed insight into model performance.

    🧑‍🔬 **Feel free to try other models and tweak parameters for better results.**
""")
