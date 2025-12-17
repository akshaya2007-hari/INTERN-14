import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# -------------------------------
# App Title
# -------------------------------
st.title("❤️ Heart Risk Prediction using KNN")
st.write("Predict heart risk based on BP and Cholesterol")

# -------------------------------
# Dataset
# -------------------------------
data = {
    'BP': [120,130,140,150,160,170,180,190,200,210],
    'Cholestrol': [200,210,220,230,240,250,260,270,280,290],
    'Heart Risk': [0,0,0,0,1,1,1,1,1,1]
}

df = pd.DataFrame(data)

st.subheader("📋 Dataset")
st.dataframe(df)

# -------------------------------
# Split features and target
# -------------------------------
X = df[['BP', 'Cholestrol']]
y = df['Heart Risk']

# -------------------------------
# Select K value
# -------------------------------
k = st.slider("Select number of neighbors (k)", 1, 5, 3)

# -------------------------------
# Train KNN Model
# -------------------------------
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X, y)

# -------------------------------
# User Input
# -------------------------------
st.subheader("🧮 Enter Patient Details")

bp = st.number_input("Blood Pressure (BP)", min_value=80, max_value=250, value=160)
chol = st.number_input("Cholesterol Level", min_value=150, max_value=350, value=250)

new_data = pd.DataFrame({
    'BP': [bp],
    'Cholestrol': [chol]
})

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Heart Risk"):
    prediction = knn.predict(new_data)[0]

    if prediction == 0:
        st.success("✅ NO RISK")
    else:
        st.error("⚠️ AT RISK")

# -------------------------------
# Visualization
# -------------------------------
st.subheader("📊 Data Visualization")

fig, ax = plt.subplots()
scatter = ax.scatter(df['BP'], df['Cholestrol'], c=df['Heart Risk'])
ax.scatter(bp, chol, color='red', s=200, marker='X')

ax.set_xlabel("Blood Pressure")
ax.set_ylabel("Cholesterol")
ax.set_title("Heart Risk Classification (KNN)")

st.pyplot(fig)
