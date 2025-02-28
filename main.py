import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import streamlit as st
from PIL import Image

df = pd.read_csv('fruit_data.csv')

X = df[['R', 'G', 'B']].values  # RGB features
y = df['class'].values  # Labels

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model accuracy: {accuracy * 100:.2f}%")

st.title("Fruit Classifier")
st.write("Upload an image of a fruit to classify it.")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image:
    img = Image.open(uploaded_image).resize((64, 64))  
    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

    img_array = np.array(img)
    mean_rgb = np.mean(img_array, axis=(0, 1)).reshape(1, -1)  

    prediction = svclassifier.predict(mean_rgb)
    class_name = le.inverse_transform(prediction)[0]

    st.write(f"Predicted class: {class_name}")
