import numpy as np
import pandas as pd
import streamlit as st
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_score, f1_score, recall_score
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

def train_model(X_train, y_train, X_valid, y_valid, model_type='multiple', degree=2):
    # Đặt tên experiment cụ thể
    experiment_name = "Regression_Experiment"  # Thay đổi tên experiment ở đây
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():
        if model_type == 'multiple':
            model = LinearRegression()
            poly = None
        elif model_type == 'polynomial':
            poly = PolynomialFeatures(degree=degree)
            X_train = poly.fit_transform(X_train)
            X_valid = poly.transform(X_valid)
            model = LinearRegression()
        else:
            raise ValueError("Loại mô hình không hợp lệ. Chọn 'multiple' hoặc 'polynomial'.")
        
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_valid_pred = model.predict(X_valid)
        
        y_train_pred_binary = np.round(y_train_pred)
        y_valid_pred_binary = np.round(y_valid_pred)
        
        train_precision = precision_score(y_train, y_train_pred_binary, average='weighted', zero_division=0)
        valid_precision = precision_score(y_valid, y_valid_pred_binary, average='weighted', zero_division=0)
        train_f1 = f1_score(y_train, y_train_pred_binary, average='weighted')
        valid_f1 = f1_score(y_valid, y_valid_pred_binary, average='weighted')
        train_recall = recall_score(y_train, y_train_pred_binary, average='weighted')
        valid_recall = recall_score(y_valid, y_valid_pred_binary, average='weighted')
        
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("degree", degree if model_type == "polynomial" else None)
        mlflow.log_metric("train_precision", train_precision)
        mlflow.log_metric("valid_precision", valid_precision)
        mlflow.log_metric("train_f1", train_f1)
        mlflow.log_metric("valid_f1", valid_f1)
        mlflow.log_metric("train_recall", train_recall)
        mlflow.log_metric("valid_recall", valid_recall)
        mlflow.sklearn.log_model(model, "model")
        
    return model, train_precision, valid_precision, train_f1, valid_f1, train_recall, valid_recall, poly

def display():
    st.title("Mô phỏng Hồi quy với MLflow Tracking")

    df = pd.read_csv('./data/processed_data.csv')
    df = df.iloc[:, 1:]
    if df is not None:
        st.write("Xem trước dữ liệu:", df.head())
        
        target_col = st.selectbox("Chọn cột mục tiêu", df.columns)
        
        if target_col:
            col1, col2 = st.columns(2)

            with col1:
                train_size = st.slider("🔹 Chọn tỷ lệ dữ liệu Train (%)", min_value=0, max_value=100, step=1, value=70, key="train_size")
            test_size = 100 - train_size  # Phần còn lại cho test

            if train_size == 0 or train_size == 100:
                st.error("🚨 Train/Test không được bằng 0% hoặc 100%. Hãy chọn lại.")
                st.stop()

            X = df.drop(columns=[target_col])
            y = df[target_col]

            # Chia tập train & test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

            # Hiển thị kích thước train/test
            st.write(f"📌 **Tập Train:** {train_size}% ({X_train.shape[0]} mẫu)")
            st.write(f"📌 **Tập Test:** {test_size}% ({X_test.shape[0]} mẫu)")

            # **Chia tiếp tập train thành train/val**
            val_size = st.slider("🔸 Chọn tỷ lệ Validation (%) (trên tập Train)", min_value=0, max_value=100, step=1, value=20, key="val_size")
            
            val_ratio = val_size / 100  # Tính phần trăm validation từ tập train
            train_final_size = 1 - val_ratio  # Phần còn lại là train

            if val_size == 100:
                st.error("🚨 Tập train không thể có 0 mẫu, hãy giảm Validation %.")  
                st.stop()

            # Chia train thành train/val
            X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=val_ratio, random_state=42)

            # Hiển thị kích thước train/val/test
            st.subheader("📊 Kích thước các tập dữ liệu")
            st.write(f"📌 **Tập Train Cuối:** {round(train_final_size * train_size, 2)}% ({X_train_final.shape[0]} mẫu)")
            st.write(f"📌 **Tập Validation:** {round(val_size * train_size / 100, 2)}% ({X_val.shape[0]} mẫu)")
            st.write(f"📌 **Tập Test:** {test_size}% ({X_test.shape[0]} mẫu)")

            model_type = st.selectbox("Chọn loại mô hình", ["multiple", "polynomial"])
            degree = st.slider("Bậc của hồi quy đa thức", 2, 5, 2) if model_type == "polynomial" else None

            if st.button("Huấn luyện mô hình"):
                model, train_precision, valid_precision, train_f1, valid_f1, train_recall, valid_recall, poly = train_model(
                    X_train_final, y_train_final, X_val, y_val, model_type=model_type, degree=degree
                )
                
                # Lưu model và poly vào session_state để sử dụng sau
                st.session_state.model = model
                st.session_state.poly = poly
                
                st.write("Kết quả huấn luyện:")
                st.write(f"- Train Precision: {train_precision:.2f}")
                st.write(f"- Validation Precision: {valid_precision:.2f}")
                st.write(f"- Train F1 Score: {train_f1:.2f}")
                st.write(f"- Validation F1 Score: {valid_f1:.2f}")
                st.write(f"- Train Recall: {train_recall:.2f}")
                st.write(f"- Validation Recall: {valid_recall:.2f}")

            # Phần input dự đoán
            st.subheader("📝 Nhập thông tin dự đoán")
            col1, col2, col3 = st.columns(3)
            with col1:
                pclass = st.selectbox("Pclass", [1, 2, 3])
                sex = st.selectbox("Sex", ["male", "female"])
            with col2:
                age = st.slider("Age", 0, 100, 25)
                sibsp = st.slider("SibSp", 1, 4, 1)
            with col3:
                embarked = st.selectbox("Embarked", ["C", "S", "Q"])
                fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=50.0)
                parch = st.selectbox("Parch", [0, 1, 2, 3, 4, 5])

            if st.button("Dự đoán"):
                if 'model' not in st.session_state or st.session_state.model is None:
                    st.error("🚨 Vui lòng huấn luyện mô hình trước khi dự đoán.")
                else:
                    # Xử lý dữ liệu đầu vào
                    sex = 1 if sex == "male" else 0
                    embarked = {"C": 0, "S": 1, "Q": 2}[embarked]
                    input_data = np.array([[pclass, sex, age, sibsp, embarked, fare, parch]])
                    
                    # Biến đổi dữ liệu nếu là mô hình đa thức
                    if 'poly' in st.session_state and st.session_state.poly:
                        input_data = st.session_state.poly.transform(input_data)
                    
                    # Dự đoán
                    prediction = st.session_state.model.predict(input_data)
                    prediction_binary = 1 if prediction[0] >= 0.5 else 0  # Chuyển đổi thành nhị phân
                    result = "Sống" if prediction_binary == 1 else "Chết"
                    st.success(f"**Dự đoán:** {result}")

if __name__ == "__main__":
    display()