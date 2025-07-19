# Iris Flower Species Prediction Web App

Welcome to the **Iris Flower Species Prediction** app! 🌸

This interactive web application lets you predict the species of an Iris flower based on its physical measurements. Built with Streamlit and powered by a Random Forest Classifier, it's a fun and educational way to explore machine learning in action.

## Features
- **User-Friendly Interface:** Use sliders to input sepal and petal measurements.
- **Instant Prediction:** Get the predicted Iris species with a single click.
- **Probability Chart:** Visualize the model's confidence for each species.
- **Feature Importance:** See which features matter most to the model.

## How to Run the App
1. **Clone this repository** (or download the code).
2. **Install the required packages:**
   ```bash
   pip install streamlit pandas scikit-learn numpy
   ```
3. **Start the app:**
   ```bash
   streamlit run app.py
   ```
4. **Interact:**
   - Adjust the sliders in the sidebar to set the flower's measurements.
   - View the predicted species and model insights on the main page.

## About the Dataset
This app uses the classic [Iris dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html), which contains 150 samples of iris flowers, each described by four features:
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

The goal is to classify each sample as one of three species:
- Setosa
- Versicolor
- Virginica



## Credits
- Built with [Streamlit](https://streamlit.io/)
- Machine learning by [scikit-learn](https://scikit-learn.org/)
- Data from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris)

---

Enjoy exploring the world of Iris flowers and machine learning! 🌱
