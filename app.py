import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import numpy as np


st.title("Iris Flower Species Prediction")


st.write(
    "This is a simple web app to predict the species of an Iris flower "
    "based on its sepal length, sepal width, petal length, and petal width."
)
st.write(
    "Use the sliders in the sidebar to provide the input features of the flower."
)


st.sidebar.header("Input Features")

def user_input_features():
    """
    Creates sliders in the sidebar for user to input flower features.
    Returns a pandas DataFrame with the user's input.
    """
    sepal_length = st.sidebar.slider('Sepal length (cm)', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width (cm)', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length (cm)', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width (cm)', 0.1, 2.5, 0.2)
    
 
    data = {
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }
   
    features = pd.DataFrame(data, index=[0])
    return features


df_input = user_input_features()


iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
Y = pd.Series(iris.target, name='species')


species_names = iris.target_names


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(iris.data, iris.target)


prediction = model.predict(df_input)
prediction_proba = model.predict_proba(df_input)


st.subheader("Your Input Parameters")
st.write(df_input)


st.subheader("Prediction")
predicted_species = species_names[prediction[0]]
st.write(f"The model predicts the flower is a **{predicted_species}**.")


st.subheader("Prediction Probability")
st.write(
    "The chart below shows the probability for each flower species."
)

prob_df = pd.DataFrame(
    prediction_proba,
    columns=species_names,
    index=['Probability']
)
st.bar_chart(prob_df.T)



st.subheader("Model Feature Importance")
st.write(
    "This chart shows which features the model found most important "
    "for making its predictions."
)


feature_importances = pd.DataFrame(
    model.feature_importances_,
    index=iris.feature_names,
    columns=['Importance']
).sort_values('Importance', ascending=False)


st.bar_chart(feature_importances)

