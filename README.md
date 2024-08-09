**Breast Cancer Prediction Using Artificial Neural Networks (ANN) and Streamlit**
This project aims to develop a machine learning model to predict whether a tumor is benign or malignant based on the Breast Cancer Wisconsin (Diagnostic) dataset. The project also includes an interactive web application built using Streamlit, where users can input data and obtain predictions.

*Project Overview*
This project leverages an Artificial Neural Network (ANN) to predict breast cancer. The goal is to classify whether the breast cancer is benign or malignant based on features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.

The project is divided into several steps:
Data preprocessing
Feature selection
Model development and training
Model evaluation
Web app development using Streamlit
Model deployment

*Project Structure*
The project contains the following files and directories:
├── breast_cancer_prediction.ipynb  # Jupyter notebook for data analysis and model development
├── app.py                           # Streamlit app script for deployment
├── requirements.txt                 # List of dependencies
└── README.md                        # Project documentation (this file)

*Description of Files*
-breast_cancer_prediction.ipynb: Contains the complete workflow, including data loading, preprocessing, feature selection, model training, and evaluation.
-app.py: Implements the Streamlit web application, allowing users to interact with the model and make predictions.
-requirements.txt: Lists all the Python packages required to run the project.
-README.md: Provides an overview and detailed instructions on how to set up and use the project.

*Setup Instructions*
1. Create a Virtual Environment
To keep your development environment clean and organized, it's recommended to create a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
2. Install Dependencies
Install the necessary Python packages listed in the requirements.txt file:
pip install -r requirements.txt
3. Dataset Acquisition
The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) dataset. You can obtain the dataset from:
UCI Machine Learning Repository
Kaggle

*Data Preprocessing*
Data preprocessing is crucial to ensure that the dataset is in a suitable format for model training. The steps involved include:

-Loading the Dataset: Load the dataset into a Pandas DataFrame.
-Handling Missing Values: Drop any columns or rows with missing data.
-Encoding Categorical Variables: Encode categorical variables (e.g., diagnosis) to numerical values.
-Feature Scaling: Normalize the feature set using techniques like StandardScaler.
-Data Splitting: Split the dataset into training, validation, and test sets.
-Feature Selection
-Feature selection is performed to identify the most relevant features for the model. This helps in reducing the model complexity and improving performance.

SelectKBest: A method from sklearn.feature_selection used to select the top K features based on statistical tests.

from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=10)
X_new = selector.fit_transform(X, y)

**Model Development**
1. Model Architecture
An Artificial Neural Network (ANN) is built using the MLPClassifier from sklearn.neural_network. The architecture includes:
An input layer corresponding to the number of selected features.
One or more hidden layers with ReLU activation functions.
An output layer with a sigmoid or softmax activation function depending on the classification task.
2. Model Training
The model is trained on the training dataset. Grid Search Cross-Validation is used to tune hyperparameters such as:
Number of hidden layers and units.
Learning rate.
Regularization parameters.
3. Model Evaluation
The model is evaluated using the test dataset. Key evaluation metrics include:
Accuracy: Overall accuracy of the model.
Confusion Matrix: A matrix showing the true positives, true negatives, false positives, and false negatives.
Precision, Recall, and F1-Score: These metrics provide insights into the model's performance, particularly for imbalanced classes.
