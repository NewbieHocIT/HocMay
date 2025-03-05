import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from PIL import Image, ImageOps
import joblib
import pandas as pd
import os
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_openml
# Load dữ liệu MNIST
def load_mnist():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data, mnist.target.astype(int)
    return X, y

def data():
    st.header("MNIST Dataset")
    st.write("""
      **MNIST** là một trong những bộ dữ liệu nổi tiếng và phổ biến nhất trong cộng đồng học máy, 
      đặc biệt là trong các nghiên cứu về nhận diện mẫu và phân loại hình ảnh.
  
      - Bộ dữ liệu bao gồm tổng cộng **70.000 ảnh chữ số viết tay** từ **0** đến **9**, 
        mỗi ảnh có kích thước **28 x 28 pixel**.
      - Chia thành:
        - **Training set**: 60.000 ảnh để huấn luyện.
        - **Test set**: 10.000 ảnh để kiểm tra.
      - Mỗi hình ảnh là một chữ số viết tay, được chuẩn hóa và chuyển thành dạng grayscale (đen trắng).
  
      Dữ liệu này được sử dụng rộng rãi để xây dựng các mô hình nhận diện chữ số.
      """)

    st.subheader("Một số hình ảnh từ MNIST Dataset")
    st.image("mnit.png", caption="Một số hình ảnh từ MNIST Dataset", use_container_width=True)

    st.subheader("Ứng dụng thực tế của MNIST")
    st.write("""
      Bộ dữ liệu MNIST đã được sử dụng trong nhiều ứng dụng nhận dạng chữ số viết tay, chẳng hạn như:
      - Nhận diện số trên các hoá đơn thanh toán, biên lai cửa hàng.
      - Xử lý chữ số trên các bưu kiện gửi qua bưu điện.
      - Ứng dụng trong các hệ thống nhận diện tài liệu tự động.
    """)

    st.subheader("Ví dụ về các mô hình học máy với MNIST")
    st.write("""
      Các mô hình học máy phổ biến đã được huấn luyện với bộ dữ liệu MNIST bao gồm:
      - **Logistic Regression**
      - **Decision Trees**
      - **K-Nearest Neighbors (KNN)**
      - **Support Vector Machines (SVM)**
      - **Convolutional Neural Networks (CNNs)**
    """)
def plot_tree_metrics():
    accuracies = [
        0.4759, 0.5759, 0.6593, 0.7741, 0.8241, 0.8259, 0.8481, 0.8574, 0.8537, 0.8463,
        0.8463, 0.8426, 0.8426, 0.8426, 0.8426, 0.8426, 0.8426, 0.8426
    ]
    tree_depths = [
        3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
    ]

    data = pd.DataFrame({
        "Tree Depth": tree_depths,
        "Accuracy": accuracies
    })

    st.subheader("Độ chính xác theo chiều sâu cây quyết định")
    st.line_chart(data.set_index('Tree Depth'))
def split_data():
    st.title("📌 Chia dữ liệu Train/Validation/Test")

    # Tải dữ liệu MNIST
    X, y = load_mnist()
    total_samples = X.shape[0]

    num_samples = st.slider(
        "Chọn số lượng ảnh để train (⚠️ Số lượng lớn sẽ lâu hơn):", 
        1000, total_samples, 10000, 
        key="classification_num_samples_slider"  # Thêm key duy nhất
    )
    st.session_state.total_samples = num_samples

    # Chọn tỉ lệ cho tập train và validation
    train_ratio = st.slider(
        "📌 Chọn % dữ liệu Train", 
        50, 90, 70, 
        key="classification_train_ratio_slider"  # Thêm key duy nhất
    )
    val_ratio = st.slider(
        "📌 Chọn % dữ liệu Validation", 
        10, 40, 15, 
        key="classification_val_ratio_slider"  # Thêm key duy nhất
    )
    test_ratio = 100 - train_ratio - val_ratio

    if test_ratio < 10:
        st.warning("⚠️ Tỉ lệ dữ liệu Test quá thấp (dưới 10%). Hãy điều chỉnh lại tỉ lệ Train và Validation.")
    else:
        st.write(f"📌 **Tỷ lệ phân chia:** Train={train_ratio}%, Validation={val_ratio}%, Test={test_ratio}%")
    
    if st.button("✅ Xác nhận & Lưu", key="classification_confirm_button"):  # Thêm key duy nhất
        X_selected, _, y_selected, _ = train_test_split(X, y, train_size=num_samples, stratify=y, random_state=42)

        stratify_option = y_selected if len(np.unique(y_selected)) > 1 else None
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X_selected, y_selected, test_size=test_ratio/100, stratify=stratify_option, random_state=42
        )

        stratify_option = y_train_full if len(np.unique(y_train_full)) > 1 else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=val_ratio / (train_ratio + val_ratio),
            stratify=stratify_option, random_state=42
        )

        st.session_state.X_train = X_train
        st.session_state.X_val = X_val
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_val = y_val
        st.session_state.y_test = y_test
        st.session_state.test_size = X_test.shape[0]
        st.session_state.val_size = X_val.shape[0]
        st.session_state.train_size = X_train.shape[0]
    
        summary_df = pd.DataFrame({
            "Tập dữ liệu": ["Train", "Validation", "Test"],
            "Số lượng mẫu": [X_train.shape[0], X_val.shape[0], X_test.shape[0]]
        })
        st.success("✅ Dữ liệu đã được chia thành công!")
        st.table(summary_df)
        
def mlflow_input():
    DAGSHUB_MLFLOW_URI = "https://dagshub.com/NewbieHocIT/MocMayvsPython.mlflow"
    st.session_state['mlflow_url'] = DAGSHUB_MLFLOW_URI
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)

    os.environ["MLFLOW_TRACKING_USERNAME"] = "NewbieHocIT"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "681dda9a41f9271a144aa94fa8624153a3c95696"

    mlflow.set_experiment("Classification")

def train():
    mlflow_input()
    if "X_train" in st.session_state:
        X_train = st.session_state.X_train 
        X_val = st.session_state.X_val 
        X_test = st.session_state.X_test 
        y_train = st.session_state.y_train 
        y_val = st.session_state.y_val 
        y_test = st.session_state.y_test 
    else:
        st.error("⚠️ Chưa có dữ liệu! Hãy chia dữ liệu trước.")
        return

    X_train = X_train.reshape(-1, 28 * 28) / 255.0
    X_test = X_test.reshape(-1, 28 * 28) / 255.0

    st.header("⚙️ Chọn mô hình & Huấn luyện")

    model_choice = st.selectbox(
        "Chọn mô hình:", 
        ["Decision Tree", "SVM"], 
        key="classification_model_choice_selectbox"  # Thêm key duy nhất
    )

    if model_choice == "Decision Tree":
        st.markdown("""
        - **🌳 Decision Tree (Cây quyết định)** giúp chia dữ liệu thành các nhóm bằng cách đặt câu hỏi nhị phân dựa trên đặc trưng.
        - **Tham số cần chọn:**  
            - **max_depth**: Giới hạn độ sâu tối đa của cây.  
        """)
        max_depth = st.slider(
            "max_depth", 
            1, 20, 5, 
            key="classification_max_depth_slider"  # Thêm key duy nhất
        )
        model = DecisionTreeClassifier(max_depth=max_depth)

    elif model_choice == "SVM":
        st.markdown("""
        - **🛠️ SVM (Support Vector Machine)** là mô hình tìm siêu phẳng tốt nhất để phân tách dữ liệu.
        """)
        C = st.slider(
            "C (Regularization)", 
            0.1, 10.0, 1.0, 
            key="classification_C_slider"  # Thêm key duy nhất
        )
        kernel = st.selectbox(
            "Kernel", 
            ["linear", "sigmoid"], 
            key="classification_kernel_selectbox"  # Thêm key duy nhất
        )
        model = SVC(C=C, kernel=kernel)
    
    n_folds = st.slider(
        "Chọn số folds (KFold Cross-Validation):", 
        min_value=2, max_value=10, value=5, 
        key="classification_n_folds_slider"  # Thêm key duy nhất
    )
    
    run_name = st.text_input(
        "🔹 Nhập tên Run:", 
        "Default_Run", 
        key="classification_run_name_input"  # Thêm key duy nhất
    )
    st.session_state["run_name"] = run_name if run_name else "default_run"
    
    if st.button("Huấn luyện mô hình", key="classification_train_button"):  # Thêm key duy nhất
        with mlflow.start_run(run_name=f"Train_{st.session_state['run_name']}"):
            mlflow.log_param("test_size", st.session_state.test_size)
            mlflow.log_param("val_size", st.session_state.val_size)
            mlflow.log_param("train_size", st.session_state.train_size)
            mlflow.log_param("num_samples", st.session_state.total_samples)
            st.write("⏳ Đang chạy Cross-Validation...")
            cv_scores = cross_val_score(model, X_train, y_train, cv=n_folds)
            mean_cv_score = cv_scores.mean()
            std_cv_score = cv_scores.std()
            
            st.success(f"📊 **Cross-Validation Accuracy**: {mean_cv_score:.4f}")

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            st.success(f"✅ Độ chính xác trên test set: {acc:.4f}")

            mlflow.log_param("model", model_choice)
            if model_choice == "Decision Tree":
                mlflow.log_param("max_depth", max_depth)
            elif model_choice == "SVM":
                mlflow.log_param("C", C)
                mlflow.log_param("kernel", kernel)

            mlflow.log_metric("test_accuracy", acc)
            mlflow.log_metric("cv_accuracy_mean", mean_cv_score)
            mlflow.log_metric("cv_accuracy_std", std_cv_score)
            mlflow.sklearn.log_model(model, model_choice.lower())

        if "models" not in st.session_state:
            st.session_state["models"] = []

        model_name = model_choice.lower().replace(" ", "_")
        if model_choice == "SVM":
            model_name += f"_{kernel}"

        existing_model = next((item for item in st.session_state["models"] if item["name"] == model_name), None)

        if existing_model:
            count = 1
            new_model_name = f"{model_name}_{count}"
            while any(item["name"] == new_model_name for item in st.session_state["models"]):
                count += 1
                new_model_name = f"{model_name}_{count}"
            model_name = new_model_name
            st.warning(f"⚠️ Mô hình được lưu với tên: {model_name}")

        st.session_state["models"].append({"name": model_name, "model": model})
        st.write(f"🔹 Mô hình đã được lưu với tên: {model_name}")
        st.write(f"Tổng số mô hình hiện tại: {len(st.session_state['models'])}")

        st.write("📋 Danh sách các mô hình đã lưu:")
        model_names = [model["name"] for model in st.session_state["models"]]
        st.write(", ".join(model_names))

        st.success(f"✅ Đã log dữ liệu cho **Train_{st.session_state['run_name']}**!")
        st.markdown(f"🔗 [Truy cập MLflow UI]({st.session_state['mlflow_url']})")

def show_experiment_selector():
    st.title("📊 MLflow Experiments")

    experiment_name = "Classification"
    
    experiments = mlflow.search_experiments()
    selected_experiment = next((exp for exp in experiments if exp.name == experiment_name), None)

    if not selected_experiment:
        st.error(f"❌ Experiment '{experiment_name}' không tồn tại!")
        return

    st.subheader(f"📌 Experiment: {experiment_name}")
    st.write(f"**Experiment ID:** {selected_experiment.experiment_id}")
    st.write(f"**Trạng thái:** {'Active' if selected_experiment.lifecycle_stage == 'active' else 'Deleted'}")
    st.write(f"**Vị trí lưu trữ:** {selected_experiment.artifact_location}")

    runs = mlflow.search_runs(experiment_ids=[selected_experiment.experiment_id])

    if runs.empty:
        st.warning("⚠ Không có runs nào trong experiment này.")
        return

    st.write("### 🏃‍♂️ Các Runs gần đây:")

    run_info = []
    for _, run in runs.iterrows():
        run_id = run["run_id"]
        run_params = mlflow.get_run(run_id).data.params
        run_name = run_params.get("run_name", f"Run {run_id[:8]}")
        run_info.append((run_name, run_id))

    run_name_to_id = dict(run_info)
    run_names = list(run_name_to_id.keys())

    selected_run_name = st.selectbox("🔍 Chọn một run:", run_names)
    selected_run_id = run_name_to_id[selected_run_name]

    selected_run = mlflow.get_run(selected_run_id)

    if selected_run:
        st.subheader(f"📌 Thông tin Run: {selected_run_name}")
        st.write(f"**Run ID:** {selected_run_id}")
        st.write(f"**Trạng thái:** {selected_run.info.status}")
        start_time_ms = selected_run.info.start_time

        if start_time_ms:
            start_time = datetime.fromtimestamp(start_time_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")
        else:
            start_time = "Không có thông tin"

        st.write(f"**Thời gian chạy:** {start_time}")

        params = selected_run.data.params
        metrics = selected_run.data.metrics

        if params:
            st.write("### ⚙️ Parameters:")
            st.json(params)

        if metrics:
            st.write("### 📊 Metrics:")
            st.json(metrics)

        dataset_path = f"{selected_experiment.artifact_location}/{selected_run_id}/artifacts/dataset.csv"
    else:
        st.warning("⚠ Không tìm thấy thông tin cho run này.")

def du_doan():
    st.title("🔢 Dự đoán chữ số viết tay")

    if "models" not in st.session_state or not st.session_state["models"]:
        st.error("⚠️ Chưa có mô hình nào được huấn luyện. Hãy huấn luyện mô hình trước.")
        return

    model_names = [model["name"] for model in st.session_state["models"]]
    selected_model_name = st.selectbox(
        "🔍 Chọn mô hình đã huấn luyện:", 
        model_names, 
        key="classification_model_selectbox"  # Thêm key duy nhất
    )
    selected_model = next(model["model"] for model in st.session_state["models"] if model["name"] == selected_model_name)

    uploaded_file = st.file_uploader(
        "📤 Tải lên ảnh chữ số viết tay (28x28 pixel)", 
        type=["png", "jpg", "jpeg"], 
        key="classification_file_uploader"  # Thêm key duy nhất
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")
        image = ImageOps.invert(image)
        image = image.resize((28, 28))
        st.image(image, caption="Ảnh đã tải lên", use_column_width=True)

        img_array = np.array(image).reshape(1, -1) / 255.0
        prediction = selected_model.predict(img_array)
        st.success(f"🔢 Dự đoán: **{prediction[0]}**")

def Classification():
    st.title("🖊️ MNIST Classification App")

    tab1, tab2, tab3, tab4 = st.tabs(["📘 Data", "⚙️ Huấn luyện", "🔢 Dự đoán", "🔥Mlflow"])

    with tab1:
        data()
        
    with tab2:
        split_data()
        train()
        
    with tab3:
        du_doan()   
    with tab4:
        show_experiment_selector()  

if __name__ == "__main__":
    Classification()