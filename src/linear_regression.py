import numpy as np
import pandas as pd
import streamlit as st
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_score, f1_score, recall_score
from src import processing
from datetime import datetime
import os

# Khởi tạo kết nối với DagsHub
def mlflow_input():
    DAGSHUB_MLFLOW_URI = "https://dagshub.com/NewbieHocIT/MocMayvsPython.mlflow"
    st.session_state['mlflow_url'] = DAGSHUB_MLFLOW_URI
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)

    os.environ["MLFLOW_TRACKING_USERNAME"] = "NewbieHocIT"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "681dda9a41f9271a144aa94fa8624153a3c95696"

    mlflow.set_experiment("Regression_Experiment")


def train_model(X_train, y_train, X_valid, y_valid, model_type='multiple', degree=2, run_name=None):
    """
    Hàm huấn luyện mô hình hồi quy và log kết quả vào MLflow.
    """
    with mlflow.start_run(run_name=run_name):
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

        # Huấn luyện mô hình
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_valid_pred = model.predict(X_valid)

        # Chuyển đổi dự đoán thành nhị phân (0 hoặc 1)
        y_train_pred_binary = np.round(y_train_pred)
        y_valid_pred_binary = np.round(y_valid_pred)

        # Tính toán các chỉ số đánh giá
        train_precision = precision_score(y_train, y_train_pred_binary, average='weighted', zero_division=0)
        valid_precision = precision_score(y_valid, y_valid_pred_binary, average='weighted', zero_division=0)
        train_f1 = f1_score(y_train, y_train_pred_binary, average='weighted')
        valid_f1 = f1_score(y_valid, y_valid_pred_binary, average='weighted')
        train_recall = recall_score(y_train, y_train_pred_binary, average='weighted')
        valid_recall = recall_score(y_valid, y_valid_pred_binary, average='weighted')

        # Log các tham số và chỉ số vào MLflow
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

    # Đọc dữ liệu
    df = pd.read_csv('./data/processed_data.csv')
    df = df.iloc[:, 1:]  # Bỏ cột đầu tiên (index)

    if df is not None:
        st.write("Xem trước dữ liệu:", df.head())

        # Chọn cột mục tiêu
        target_col = st.selectbox("Chọn cột mục tiêu", df.columns)

        if target_col:
            # Chọn tỷ lệ tập Test
            test_size = st.slider("🔹 Chọn tỷ lệ dữ liệu Test (%)", min_value=0, max_value=50, step=1, value=20)
            
            # Chọn tỷ lệ tập Validation
            val_size = st.slider("🔸 Chọn tỷ lệ dữ liệu Validation (%)", min_value=0, max_value=50, step=1, value=15)

            # Tính tỷ lệ tập Train
            train_size = 100 - test_size - val_size

            # Kiểm tra nếu tỷ lệ không hợp lệ
            if train_size <= 0:
                st.error("🚨 Tổng Test + Validation không được vượt quá 100%. Hãy chọn lại.")
                st.stop()

            # Chia dữ liệu thành Train/Test
            X = df.drop(columns=[target_col])
            y = df[target_col]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100, random_state=42)

            # Chia tiếp Train thành Train/Validation
            val_ratio = val_size / (train_size + val_size)  # Tỷ lệ Validation trên tổng Train + Validation
            X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=val_ratio, random_state=42)

            # Hiển thị kích thước train/val/test
            st.subheader("📊 Kích thước các tập dữ liệu")
            st.write(f"📌 **Tập Train:** {train_size}% ({X_train_final.shape[0]} mẫu)")
            st.write(f"📌 **Tập Validation:** {val_size}% ({X_val.shape[0]} mẫu)")
            st.write(f"📌 **Tập Test:** {test_size}% ({X_test.shape[0]} mẫu)")

            # Chọn loại mô hình
            model_type = st.selectbox("Chọn loại mô hình", ["multiple", "polynomial"])
            degree = st.slider("Bậc của hồi quy đa thức", 2, 5, 2) if model_type == "polynomial" else None

            # Đặt tên cho mô hình
            model_name = st.text_input("Đặt tên cho mô hình (tùy chọn)")

            # Huấn luyện mô hình
            if st.button("Huấn luyện mô hình"):
                if not model_name:
                    st.error("🚨 Vui lòng đặt tên cho mô hình.")
                    return

                model, train_precision, valid_precision, train_f1, valid_f1, train_recall, valid_recall, poly = train_model(
                    X_train_final, y_train_final, X_val, y_val, model_type=model_type, degree=degree, run_name=model_name
                )

                # Lưu mô hình vào danh sách
                if 'models' not in st.session_state:
                    st.session_state.models = {}
                st.session_state.models[model_name] = {
                    "model": model,
                    "poly": poly,
                    "model_type": model_type,
                    "degree": degree
                }

                # Hiển thị kết quả huấn luyện
                st.write("Kết quả huấn luyện:")
                st.write(f"- Train Precision: {train_precision:.2f}")
                st.write(f"- Validation Precision: {valid_precision:.2f}")
                st.write(f"- Train F1 Score: {train_f1:.2f}")
                st.write(f"- Validation F1 Score: {valid_f1:.2f}")
                st.write(f"- Train Recall: {train_recall:.2f}")
                st.write(f"- Validation Recall: {valid_recall:.2f}")


import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))



def predict():
    st.subheader("📝 Nhập thông tin dự đoán")
    
    # Kiểm tra xem có mô hình nào đã được huấn luyện không
    if 'models' not in st.session_state or not st.session_state.models:
        st.error("🚨 Vui lòng huấn luyện ít nhất một mô hình trước khi dự đoán.")
        return

    # Chọn mô hình từ danh sách
    model_names = list(st.session_state.models.keys())
    selected_model_name = st.selectbox("Chọn mô hình đã huấn luyện", model_names)

    # Lấy thông tin mô hình đã chọn
    selected_model = st.session_state.models[selected_model_name]
    model = selected_model["model"]
    poly = selected_model["poly"]
    model_type = selected_model["model_type"]
    degree = selected_model.get("degree", None)

    # Hiển thị thông tin mô hình
    st.write("### Thông tin mô hình đã chọn:")
    st.write(f"- Tên mô hình: {selected_model_name}")
    st.write(f"- Loại mô hình: {model_type}")
    if model_type == "polynomial":
        st.write(f"- Bậc đa thức: {degree}")

    # Tạo các trường nhập liệu
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

    # Nút dự đoán
    if st.button("Dự đoán"):
        # Xử lý dữ liệu đầu vào
        sex = 1 if sex == "male" else 0
        embarked = {"C": 0, "S": 1, "Q": 2}[embarked]
        input_data = np.array([[pclass, sex, age, sibsp, embarked, fare, parch]])

        # Biến đổi dữ liệu nếu là mô hình đa thức
        if model_type == "polynomial" and poly is not None:
            input_data = poly.transform(input_data)

        prediction = model.predict(input_data)[0]
        prediction = sigmoid(prediction)  # Đưa về khoảng [0, 1]

        prediction_binary = 1 if prediction >= 0.5 else 0
        confidence = round(abs(prediction - 0.5) * 200, 2)

        result = "Sống" if prediction_binary == 1 else "Chết"
        st.success(f"**Dự đoán:** {result}  \n🔍 **Độ tin cậy:** {confidence}%")


def show_experiment_selector():
    st.title("📊 MLflow Experiments")

    experiment_name = "Regression_Experiment"
    
    # Lấy danh sách experiment
    experiments = mlflow.search_experiments()
    selected_experiment = next((exp for exp in experiments if exp.name == experiment_name), None)

    if not selected_experiment:
        st.error(f"❌ Experiment '{experiment_name}' không tồn tại!")
        return

    st.subheader(f"📌 Experiment: {experiment_name}")
    st.write(f"**Experiment ID:** {selected_experiment.experiment_id}")
    st.write(f"**Trạng thái:** {'Active' if selected_experiment.lifecycle_stage == 'active' else 'Deleted'}")
    st.write(f"**Vị trí lưu trữ:** {selected_experiment.artifact_location}")

    # Lấy danh sách runs trong experiment
    runs = mlflow.search_runs(experiment_ids=[selected_experiment.experiment_id])

    if runs.empty:
        st.warning("⚠ Không có runs nào trong experiment này.")
        return

    st.write("### 🏃‍♂️ Các Runs gần đây:")

    # Tạo danh sách run name và map với run_id
    run_dict = {}
    for _, run in runs.iterrows():
        run_name = run.get("tags.mlflow.runName", f"Run {run['run_id'][:8]}")
        run_dict[run_name] = run["run_id"]  # Map run_name -> run_id

    # Chọn run theo tên
    selected_run_name = st.selectbox("🔍 Chọn một run:", list(run_dict.keys()), key="runname")
    selected_run_id = run_dict[selected_run_name]

    # Lấy thông tin của run đã chọn
    selected_run = mlflow.get_run(selected_run_id)

    if selected_run:
        st.subheader(f"📌 Thông tin Run: {selected_run_name}")
        st.write(f"**Run ID:** {selected_run_id}")
        st.write(f"**Trạng thái:** {selected_run.info.status}")
        
        start_time_ms = selected_run.info.start_time
        start_time = datetime.fromtimestamp(start_time_ms / 1000).strftime("%Y-%m-%d %H:%M:%S") if start_time_ms else "Không có thông tin"

        st.write(f"**Thời gian chạy:** {start_time}")

        params = selected_run.data.params
        metrics = selected_run.data.metrics

        if params:
            st.write("### ⚙️ Parameters:")
            st.json(params)

        if metrics:
            st.write("### 📊 Metrics:")
            st.json(metrics)

    else:
        st.warning("⚠ Không tìm thấy thông tin cho run này.")


def LinearRegressionApp():
    st.title("🖊️ Linear Regression App")

    tab1, tab2, tab3, tab4 = st.tabs(["📘 Processing ", "⚙️ Huấn luyện", "🔢 Dự đoán", "🔥Mlflow"])

    with tab1:
        processing.display()
    with tab2:
        display()
    with tab3:
        predict()  # Gọi hàm dự đoán
    with tab4:
        show_experiment_selector()


def LinearApp():
    mlflow_input()
    LinearRegressionApp()


if __name__ == "__main__":
    mlflow_input()
    LinearRegressionApp()