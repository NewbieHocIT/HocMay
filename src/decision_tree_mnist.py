import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import pickle
import mlflow
import dagshub
import os

# Khởi tạo MLflow và DagsHub
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
    X = train_data.iloc[:, 1:].values / 255.0
    y = train_data.iloc[:, 0].values
    return train_data, X, y

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

def display():
    st.title("🖼️ MNIST Classification using Decision Tree")
    st.header("📌 Step 1: Understanding Data")
    st.write("Below are some sample images from the dataset:")

    show_sample_images()

    st.write("🔹 The pixel values are normalized by dividing by 255 to scale them between 0 and 1, which helps improve model performance and convergence speed.")
    train_data, X, y = load_data()
    st.write("📊 First few rows of the dataset:")
    st.dataframe(train_data.head())
    st.write(f"📏 Dataset Shape: {train_data.shape}")

    st.write("📊 Label Distribution:")
    plot_label_distribution(y)

    if st.button("Proceed to Training 🚀"):
        st.session_state['train_ready'] = True

    if 'train_ready' in st.session_state:
        st.header("📌 Step 2: Training Model")
        
        # Phần chia dữ liệu với thanh trượt
        col1, col2 = st.columns(2)
        with col1:
            train_size = st.slider("🔹 Chọn tỷ lệ dữ liệu Train (%)", min_value=0.0, max_value=100.0, value=70.0, step=1.0, key="train_size")
        with col2:
            val_size = st.slider("🔸 Chọn tỷ lệ Validation (%)", min_value=0.0, max_value=100.0 - train_size, value=15.0, step=1.0, key="val_size")
        test_size = 100.0 - train_size - val_size  # Phần còn lại cho test

        if train_size == 0 or val_size == 0 or test_size == 0:
            st.error("🚨 Tỷ lệ Train/Validation/Test không được bằng 0%. Hãy chọn lại.")
        else:
            train_ratio = train_size / 100.0
            val_ratio = val_size / 100.0
            test_ratio = test_size / 100.0

            st.write(f"📌 **Tập Train:** {train_size}%")
            st.write(f"📌 **Tập Validation:** {val_size}%")
            st.write(f"📌 **Tập Test:** {test_size}%")

            if st.button("Train Model 🎯"):
                # Chia tập train thành tập train và tập validation
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_ratio, random_state=42)
                
                st.write("⏳ Training Decision Tree Model...")
                model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
                model.fit(X_train, y_train)
                
                with open("decision_tree_mnist_model.pkl", "wb") as model_file:
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

                # Log thí nghiệm vào MLflow
                try:
                    init_mlflow()
                    # Thiết lập thí nghiệm với tên cụ thể
                    mlflow.set_experiment("decision_tree_mnist")
                    with mlflow.start_run():
                        # Log các thông số
                        mlflow.log_param("train_size", train_size)
                        mlflow.log_param("val_size", val_size)
                        mlflow.log_param("test_size", test_size)
                        mlflow.log_param("max_depth", 3)
                        mlflow.log_param("criterion", "entropy")

                        # Log các metrics
                        mlflow.log_metric("accuracy", accuracy)
                        for label, metrics in class_report.items():
                            if label.isdigit():
                                mlflow.log_metric(f"precision_{label}", metrics['precision'])
                                mlflow.log_metric(f"recall_{label}", metrics['recall'])
                                mlflow.log_metric(f"f1_score_{label}", metrics['f1-score'])

                        # Log mô hình
                        mlflow.sklearn.log_model(model, "decision_tree_model")
                        st.success("✅ Thí nghiệm đã được log vào MLflow trong thư mục 'decision_tree_mnist'!")
                except Exception as e:
                    st.error(f"🚨 Lỗi khi log thí nghiệm: {e}")

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