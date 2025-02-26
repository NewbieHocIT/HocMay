import streamlit as st
import os
from src import linear_regression 
from src import processing
from src import svm_mnist
from src import decision_tree_mnist
from src import clustering
from src import mlflow_web
# Sidebar navigation
st.sidebar.title("App Selection")
option = st.sidebar.selectbox("Chọn lựa chọn phù hợp:", ["Pre Processing", "Linear Regression", "SVM Mnist", "Decision Tree Mnist",  "Clustering", "ML-Flow"])

if(option == 'Pre Processing'):
    processing.display()
elif(option == 'Linear Regression'):
    linear_regression.display()
elif(option == 'SVM Mnist'):
    svm_mnist.display()
elif(option == 'Decision Tree Mnist'):
    decision_tree_mnist.display()
elif(option == 'Clustering'):
    clustering.display()
elif(option == 'ML-Flow'):
    mlflow_web.display()