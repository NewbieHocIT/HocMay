import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import pickle
import mlflow
import dagshub
import os

# Khởi tạo kết nối với DagsHub
def init_mlflow():
    try:
        dagshub.init(repo_owner='NewbieHocIT', repo_name='MocMayvsPython', mlflow=True)
        os.environ['MLFLOW_TRACKING_USERNAME'] = 'NewbieHocIT'
        os.environ['MLFLOW_TRACKING_PASSWORD'] = '681dda9a41f9271a144aa94fa8624153a3c95696'
        mlflow.set_tracking_uri("https://dagshub.com/NewbieHocIT/MocMayvsPython.mlflow")
        print("Kết nối MLflow thành công!")
    except Exception as e:
        st.warning("Không thể kết nối với MLflow hoặc DagsHub. Vui lòng kiểm tra cài đặt.")

def load_data():
    # Load tập train
    train_data = pd.read_csv("data/mnist/train.csv")
    X_train = train_data.iloc[:, 1:].values / 255.0
    y_train = train_data.iloc[:, 0].values

    # Load tập test (nếu có)
    try:
        test_data = pd.read_csv("data/mnist/test.csv")
        X_test = test_data.values / 255.0
        y_test = None  # Giả sử tập test không có nhãn
    except FileNotFoundError:
        X_test, y_test = None, None

    return train_data, X_train, y_train, X_test, y_test

def show_sample_images():
    train_data = pd.read_csv("data/mnist/train.csv")
    unique_labels = train_data.iloc[:, 0].unique()
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    label_count = 0
    
    for i, ax in enumerate(axes.flat):
        if label_count >= len(unique_labels):
            break
        sample = train_data[train_data.iloc[:, 0] == unique_labels[label_count]].iloc[0, 1:].values.reshape(28, 28)
        ax.imshow(sample, cmap='gray')
        ax.set_title(f"Label: {unique_labels[label_count]}", fontsize=10)
        ax.axis("off")
        label_count += 1
    st.pyplot(fig)

def plot_label_distribution(y):
    fig, ax = plt.subplots(figsize=(8, 5))
    pd.Series(y).value_counts().sort_index().plot(kind="bar", ax=ax, color="skyblue")
    ax.set_title("Label Distribution in Dataset")
    ax.set_xlabel("Digit Label")
    ax.set_ylabel("Count")
    st.pyplot(fig)

def log_experiment(model_name, model, X_val, y_val, y_val_pred, train_size, val_size, test_size):
    """ Log thí nghiệm với MLflow """
    try:
        init_mlflow()
        experiment_name = "MNIST_SVM_Classification"
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name=model_name) as run:
            # Log các tham số
            mlflow.log_param("Train Size (%)", train_size)
            mlflow.log_param("Validation Size (%)", val_size)
            mlflow.log_param("Test Size (%)", test_size)
            mlflow.log_param("Model", "SVM")
            mlflow.log_param("Kernel", "linear")

            # Tính toán và log các metrics
            accuracy = accuracy_score(y_val, y_val_pred)
            class_report = classification_report(y_val, y_val_pred, output_dict=True)
            
            mlflow.log_metric("Validation Accuracy", accuracy)
            for label, metrics in class_report.items():
                if label.isdigit():  # Chỉ log metrics cho các lớp số
                    mlflow.log_metric(f"Precision_{label}", metrics['precision'])
                    mlflow.log_metric(f"Recall_{label}", metrics['recall'])
                    mlflow.log_metric(f"F1-Score_{label}", metrics['f1-score'])

            # Log mô hình
            mlflow.sklearn.log_model(model, "svm_mnist_model")

            st.success(f"✅ Mô hình được log vào thí nghiệm: {experiment_name}")
            st.write(f"- Validation Accuracy: {accuracy:.4f}")
            st.write(f"- Classification Report:")
            st.dataframe(pd.DataFrame(class_report).transpose())
    except Exception as e:
        st.error(f"Lỗi khi log thí nghiệm: {e}")

def display():
    st.title("🖼️ MNIST Classification using SVM")
    st.header("📌 Step 1: Understanding Data")
    st.write("Below are some sample images from the dataset:")

    show_sample_images()

    st.write("🔹 The pixel values are normalized by dividing by 255 to scale them between 0 and 1, which helps improve model performance and convergence speed.")
    train_data, X_train, y_train, X_test, y_test = load_data()
    st.write("📊 First few rows of the dataset:")
    st.dataframe(train_data.head())
    st.write(f"📏 Dataset Shape: {train_data.shape}")

    st.write("📊 Label Distribution:")
    plot_label_distribution(y_train)

    if st.button("Proceed to Training 🚀"):
        st.session_state['train_ready'] = True

    if 'train_ready' in st.session_state:
        st.header("📌 Step 2: Training Model")
        
        # Phần chia dữ liệu với thanh trượt
        col1, col2 = st.columns(2)
        with col1:
            train_size = st.slider("🔹 Chọn tỷ lệ dữ liệu Train (%)", min_value=0, max_value=100, step=1, value=70, key="train_size")
        with col2:
            val_size = st.slider("🔸 Chọn tỷ lệ Validation (%)", min_value=0, max_value=100 - train_size, step=1, value=15, key="val_size")
        test_size = 100 - train_size - val_size  # Phần còn lại cho test

        if train_size == 0 or val_size == 0 or test_size == 0:
            st.error("🚨 Tỷ lệ Train/Validation/Test không được bằng 0%. Hãy chọn lại.")
        else:
            train_ratio = train_size / 100
            val_ratio = val_size / 100
            test_ratio = test_size / 100

            st.write(f"📌 **Tập Train:** {train_size}%")
            st.write(f"📌 **Tập Validation:** {val_size}%")
            st.write(f"📌 **Tập Test:** {test_size}%")

            if st.button("Train Model 🎯"):
                # Chia tập train thành tập train và tập validation
                X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=val_ratio, random_state=42)
                
                st.write("⏳ Training SVM Model...")
                model = SVC(kernel='linear')
                model.fit(X_train_final, y_train_final)
                
                with open("svm_mnist_model.pkl", "wb") as model_file:
                    pickle.dump(model, model_file)
                st.session_state['model'] = model
                st.session_state['X_val'] = X_val
                st.session_state['y_val'] = y_val
                
                # Đánh giá trên tập validation
                y_val_pred = model.predict(X_val)
                accuracy = accuracy_score(y_val, y_val_pred)
                class_report = classification_report(y_val, y_val_pred, output_dict=True)
                
                st.success(f"✅ Model Accuracy (Validation): {accuracy:.4f}")
                st.subheader("📊 Classification Report (Validation)")
                st.dataframe(pd.DataFrame(class_report).transpose())

                # Log thí nghiệm
                model_name = "SVM_MNIST_Model"
                log_experiment(model_name, model, X_val, y_val, y_val_pred, train_size, val_size, test_size)

    if 'model' in st.session_state:
        st.header("📌 Step 4: Predict Custom Digit")
        uploaded_file = st.file_uploader("📤 Upload grayscale image of a digit", type=["png", "jpg", "jpeg"])
        
        if uploaded_file is not None:
            from PIL import Image
            image = Image.open(uploaded_file).convert("L").resize((28, 28))
            image_array = np.array(image) / 255.0
            image_flatten = image_array.flatten().reshape(1, -1)
            
            st.image(image, caption="Uploaded Image", width=100)
            prediction = st.session_state['model'].predict(image_flatten)[0]
            st.success(f"🔮 Predicted Label: {prediction}")

if __name__ == "__main__":
    display()