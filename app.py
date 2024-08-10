import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split

# Load the dataset
@st.cache
def load_data():
    df = pd.read_csv(r'C:\Users\User\Downloads\DL-ASSIGN 4\data (2).csv')
    df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    
    # Split the dataset
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42)
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Calculate statistics for user input sliders
    feature_stats = {col: {
        'min': X[col].min(),
        'max': X[col].max(),
        'mean': X[col].mean()
    } for col in X.columns}
    
    return X_test, y_test, scaler, feature_stats

@st.cache
def load_model():
    # Load the saved model
    with open('ann_model.pkl', 'rb') as file:
        ann_model = pickle.load(file)
    return ann_model

def main():
    st.title("Breast Cancer Prediction App")

    # Load data and model
    X_test, y_test, scaler, feature_stats = load_data()
    ann_model = load_model()

    st.sidebar.header("User Input Features")

    def user_input_features():
        data = {}
        for feature, stats in feature_stats.items():
            min_val = stats['min']
            max_val = stats['max']
            mean_val = stats['mean']
            data[feature] = st.sidebar.slider(feature, float(min_val), float(max_val), float(mean_val))
        return pd.DataFrame(data, index=[0])

    input_df = user_input_features()
    input_scaled = scaler.transform(input_df)
    prediction = ann_model.predict(input_scaled)
    prediction_proba = ann_model.predict_proba(input_scaled)

    st.subheader("Prediction")
    if prediction[0] == 1:
        st.write("The tumor is **Malignant (Cancerous)**")
    else:
        st.write("The tumor is **Benign (Not Cancerous)**")

    st.subheader("Prediction Probability")
    st.write(f"Probability of Malignant: {prediction_proba[0][1]:.2f}")
    st.write(f"Probability of Benign: {prediction_proba[0][0]:.2f}")

    st.subheader("Model Performance")
    y_pred = ann_model.predict(X_test)
    st.write("Accuracy on test set: ", accuracy_score(y_test, y_pred))
    st.text("Classification Report on test set:")
    st.text(classification_report(y_test, y_pred))

    st.subheader("Confusion Matrix")
    conf_matrix = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'], ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

if __name__ == "__main__":
    main()
