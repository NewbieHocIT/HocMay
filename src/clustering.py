import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image, ImageOps
import pandas as pd
import os
import mlflow
from datetime import datetime
from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
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


def reduce_dimensions(X, method='PCA', n_components=2):
    if method == 'PCA':
        reducer = PCA(n_components=n_components)
    elif method == 't-SNE':
        reducer = TSNE(n_components=n_components, perplexity=30, n_iter=300)
    else:
        raise ValueError("Phương pháp giảm chiều không hợp lệ. Chọn 'PCA' hoặc 't-SNE'.")
    
    X_reduced = reducer.fit_transform(X)
    return X_reduced
def split_data():
    st.title("📌 Chia dữ liệu Train/Validation/Test")

    # Tải dữ liệu MNIST
    X, y = load_mnist()
    total_samples = X.shape[0]

    num_samples = st.slider(
        "Chọn số lượng ảnh để train (⚠️ Số lượng lớn sẽ lâu hơn):", 
        1000, total_samples, 10000, 
        key="clustering_num_samples_slider"  # Thêm key duy nhất
    )
    st.session_state.total_samples = num_samples

    # Chọn tỉ lệ cho tập train và validation
    train_ratio = st.slider(
        "📌 Chọn % dữ liệu Train", 
        50, 90, 70, 
        key="clustering_train_ratio_slider"  # Thêm key duy nhất
    )
    val_ratio = st.slider(
        "📌 Chọn % dữ liệu Validation", 
        10, 40, 15, 
        key="clustering_val_ratio_slider"  # Thêm key duy nhất
    )
    test_ratio = 100 - train_ratio - val_ratio

    if test_ratio < 10:
        st.warning("⚠️ Tỉ lệ dữ liệu Test quá thấp (dưới 10%). Hãy điều chỉnh lại tỉ lệ Train và Validation.")
    else:
        st.write(f"📌 **Tỷ lệ phân chia:** Train={train_ratio}%, Validation={val_ratio}%, Test={test_ratio}%")
    
    if st.button("✅ Xác nhận & Lưu", key="clustering_confirm_button"):  # Thêm key duy nhất
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

    mlflow.set_experiment("Clustering")

def train_model(model, X_train_reduced, y_train, X_test_reduced, y_test, model_choice, reduction_method, n_components, n_clusters=None, eps=None, min_samples=None):
    """
    Hàm huấn luyện mô hình và log kết quả vào MLflow.
    """
    with mlflow.start_run(run_name=f"Train_{st.session_state['run_name']}"):
        # Log các tham số
        mlflow.log_param("test_size", st.session_state.test_size)
        mlflow.log_param("val_size", st.session_state.val_size)
        mlflow.log_param("train_size", st.session_state.train_size)
        mlflow.log_param("num_samples", st.session_state.total_samples)
        mlflow.log_param("reduction_method", reduction_method)
        mlflow.log_param("n_components", n_components)

        if model_choice == "K-means":
            mlflow.log_param("n_clusters", n_clusters)
        elif model_choice == "DBSCAN":
            mlflow.log_param("eps", eps)
            mlflow.log_param("min_samples", min_samples)

        st.write("⏳ Đang huấn luyện mô hình...")
        model.fit(X_train_reduced)
        labels = model.labels_

        # Tính toán silhouette score
        if len(np.unique(labels)) > 1:
            silhouette_avg = silhouette_score(X_train_reduced, labels)
            st.success(f"📊 **Silhouette Score**: {silhouette_avg:.4f}")
            mlflow.log_metric("silhouette_score", silhouette_avg)
        else:
            st.warning("⚠ Không thể tính silhouette score vì chỉ có một cụm.")

        # Lưu mô hình vào session_state
        if "models" not in st.session_state:
            st.session_state["models"] = []

        model_name = model_choice.lower().replace(" ", "_")
        if model_choice == "DBSCAN":
            model_name += f"_eps{eps}_min_samples{min_samples}"
        elif model_choice == "K-means":
            model_name += f"_n_clusters{n_clusters}"

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
        ["K-means", "DBSCAN"], 
        key="clustering_model_choice_selectbox"  # Thêm key duy nhất
    )

    # Giảm chiều dữ liệu
    reduction_method = st.selectbox(
        "Chọn phương pháp giảm chiều dữ liệu:", 
        ["PCA", "t-SNE"], 
        key="clustering_reduction_method_selectbox"  # Thêm key duy nhất
    )
    n_components = st.slider(
        "Số chiều sau khi giảm:", 
        2, 
        50 if reduction_method == "PCA" else 3,  # Giới hạn t-SNE tối đa là 3
        2,
        key="clustering_n_components_slider"  # Thêm key duy nhất
    )

    # Lưu vào session_state
    st.session_state.reduction_method = reduction_method
    st.session_state.n_components = n_components

    X_train_reduced = reduce_dimensions(X_train, method=reduction_method, n_components=n_components)
    X_test_reduced = reduce_dimensions(X_test, method=reduction_method, n_components=n_components)

    if model_choice == "K-means":
        st.markdown("""
        - **K-means** là một thuật toán phân cụm dựa trên khoảng cách giữa các điểm dữ liệu.
        - **Tham số cần chọn:**  
            - **n_clusters**: Số lượng cụm.  
        """)
        n_clusters = st.slider(
            "n_clusters", 
            2, 20, 10, 
            key="clustering_n_clusters_slider"  # Thêm key duy nhất
        )
        model = KMeans(n_clusters=n_clusters)

    elif model_choice == "DBSCAN":
        st.markdown("""
        - **DBSCAN** là một thuật toán phân cụm dựa trên mật độ.
        """)
        eps = st.slider(
            "eps (Khoảng cách tối đa giữa hai điểm để coi là lân cận)", 
            0.1, 1.0, 0.5, 
            key="clustering_eps_slider"  # Thêm key duy nhất
        )
        min_samples = st.slider(
            "min_samples (Số lượng điểm tối thiểu trong một lân cận)", 
            1, 20, 5, 
            key="clustering_min_samples_slider"  # Thêm key duy nhất
        )
        model = DBSCAN(eps=eps, min_samples=min_samples)

    if st.button("Huấn luyện mô hình", key="clustering_train_button"):  # Thêm key duy nhất
        st.session_state["run_name"] = "training_run_1"  # Tạo giá trị cho run_name

        # Gọi hàm huấn luyện mô hình
        train_model(
            model=model,
            X_train_reduced=X_train_reduced,
            y_train=y_train,
            X_test_reduced=X_test_reduced,
            y_test=y_test,
            model_choice=model_choice,
            reduction_method=reduction_method,
            n_components=n_components,
            n_clusters=n_clusters if model_choice == "K-means" else None,
            eps=eps if model_choice == "DBSCAN" else None,
            min_samples=min_samples if model_choice == "DBSCAN" else None
        )


def du_doan():
    st.title("🔢 Dự đoán chữ số viết tay")

    if "models" not in st.session_state or not st.session_state["models"]:
        st.error("⚠️ Chưa có mô hình nào được huấn luyện. Hãy huấn luyện mô hình trước.")
        return

    model_names = [model["name"] for model in st.session_state["models"]]
    selected_model_name = st.selectbox(
        "🔍 Chọn mô hình đã huấn luyện:", 
        model_names, 
        key="clustering_model_selectbox"  # Thêm key duy nhất
    )
    selected_model = next(model["model"] for model in st.session_state["models"] if model["name"] == selected_model_name)

    uploaded_file = st.file_uploader(
        "📤 Tải lên ảnh chữ số viết tay (28x28 pixel)", 
        type=["png", "jpg", "jpeg"], 
        key="clustering_file_uploader"  # Thêm key duy nhất
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")
        image = ImageOps.invert(image)
        image = image.resize((28, 28))
        st.image(image, caption="Ảnh đã tải lên", use_column_width=True)

        img_array = np.array(image).reshape(1, -1) / 255.0
        prediction = selected_model.predict(img_array)
        st.success(f"🔢 Dự đoán: **{prediction[0]}**")

def show_experiment_selector():
    st.title("📊 MLflow Experiments - DAGsHub")

    experiment_name = "Clustering"
    
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

    # Thêm key duy nhất cho selectbox
    selected_run_name = st.selectbox(
        "🔍 Chọn một run:", 
        run_names, 
        key="clustering_experiment_selectbox"  # Thêm key duy nhất
    )
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
        st.write("### 📂 Dataset:")
        st.write(f"📥 [Tải dataset]({dataset_path})")
    else:
        st.warning("⚠ Không tìm thấy thông tin cho run này.")


def Clusterting():
    st.title(" MNIST Clustering App")

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
    Clusterting()


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


def reduce_dimensions(X, method='PCA', n_components=2):
    if method == 'PCA':
        reducer = PCA(n_components=n_components)
    elif method == 't-SNE':
        reducer = TSNE(n_components=n_components, perplexity=30, n_iter=300)
    else:
        raise ValueError("Phương pháp giảm chiều không hợp lệ. Chọn 'PCA' hoặc 't-SNE'.")
    
    X_reduced = reducer.fit_transform(X)
    return X_reduced
def split_data():
    st.title("📌 Chia dữ liệu Train/Validation/Test")

    # Tải dữ liệu MNIST
    X, y = load_mnist()
    total_samples = X.shape[0]

    num_samples = st.slider(
        "Chọn số lượng ảnh để train (⚠️ Số lượng lớn sẽ lâu hơn):", 
        1000, total_samples, 10000, 
        key="clustering_num_samples_slider"  # Thêm key duy nhất
    )
    st.session_state.total_samples = num_samples

    # Chọn tỉ lệ cho tập train và validation
    train_ratio = st.slider(
        "📌 Chọn % dữ liệu Train", 
        50, 90, 70, 
        key="clustering_train_ratio_slider"  # Thêm key duy nhất
    )
    val_ratio = st.slider(
        "📌 Chọn % dữ liệu Validation", 
        10, 40, 15, 
        key="clustering_val_ratio_slider"  # Thêm key duy nhất
    )
    test_ratio = 100 - train_ratio - val_ratio

    if test_ratio < 10:
        st.warning("⚠️ Tỉ lệ dữ liệu Test quá thấp (dưới 10%). Hãy điều chỉnh lại tỉ lệ Train và Validation.")
    else:
        st.write(f"📌 **Tỷ lệ phân chia:** Train={train_ratio}%, Validation={val_ratio}%, Test={test_ratio}%")
    
    if st.button("✅ Xác nhận & Lưu", key="clustering_confirm_button"):  # Thêm key duy nhất
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

    mlflow.set_experiment("Clustering")

def train_model(model, X_train_reduced, y_train, X_test_reduced, y_test, model_choice, reduction_method, n_components, n_clusters=None, eps=None, min_samples=None):
    """
    Hàm huấn luyện mô hình và log kết quả vào MLflow.
    """
    with mlflow.start_run(run_name=f"Train_{st.session_state['run_name']}"):
        # Log các tham số
        mlflow.log_param("test_size", st.session_state.test_size)
        mlflow.log_param("val_size", st.session_state.val_size)
        mlflow.log_param("train_size", st.session_state.train_size)
        mlflow.log_param("num_samples", st.session_state.total_samples)
        mlflow.log_param("reduction_method", reduction_method)
        mlflow.log_param("n_components", n_components)

        if model_choice == "K-means":
            mlflow.log_param("n_clusters", n_clusters)
        elif model_choice == "DBSCAN":
            mlflow.log_param("eps", eps)
            mlflow.log_param("min_samples", min_samples)

        st.write("⏳ Đang huấn luyện mô hình...")
        model.fit(X_train_reduced)
        labels = model.labels_

        # Tính toán silhouette score
        if len(np.unique(labels)) > 1:
            silhouette_avg = silhouette_score(X_train_reduced, labels)
            st.success(f"📊 **Silhouette Score**: {silhouette_avg:.4f}")
            mlflow.log_metric("silhouette_score", silhouette_avg)
        else:
            st.warning("⚠ Không thể tính silhouette score vì chỉ có một cụm.")

        # Lưu mô hình vào session_state
        if "models" not in st.session_state:
            st.session_state["models"] = []

        model_name = model_choice.lower().replace(" ", "_")
        if model_choice == "DBSCAN":
            model_name += f"_eps{eps}_min_samples{min_samples}"
        elif model_choice == "K-means":
            model_name += f"_n_clusters{n_clusters}"

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
        ["K-means", "DBSCAN"], 
        key="clustering_model_choice_selectbox"  # Thêm key duy nhất
    )

    # Giảm chiều dữ liệu
    reduction_method = st.selectbox(
        "Chọn phương pháp giảm chiều dữ liệu:", 
        ["PCA", "t-SNE"], 
        key="clustering_reduction_method_selectbox"  # Thêm key duy nhất
    )
    n_components = st.slider(
        "Số chiều sau khi giảm:", 
        2, 
        50 if reduction_method == "PCA" else 3,  # Giới hạn t-SNE tối đa là 3
        2,
        key="clustering_n_components_slider"  # Thêm key duy nhất
    )

    # Lưu vào session_state
    st.session_state.reduction_method = reduction_method
    st.session_state.n_components = n_components

    X_train_reduced = reduce_dimensions(X_train, method=reduction_method, n_components=n_components)
    X_test_reduced = reduce_dimensions(X_test, method=reduction_method, n_components=n_components)

    if model_choice == "K-means":
        st.markdown("""
        - **K-means** là một thuật toán phân cụm dựa trên khoảng cách giữa các điểm dữ liệu.
        - **Tham số cần chọn:**  
            - **n_clusters**: Số lượng cụm.  
        """)
        n_clusters = st.slider(
            "n_clusters", 
            2, 20, 10, 
            key="clustering_n_clusters_slider"  # Thêm key duy nhất
        )
        model = KMeans(n_clusters=n_clusters)

    elif model_choice == "DBSCAN":
        st.markdown("""
        - **DBSCAN** là một thuật toán phân cụm dựa trên mật độ.
        """)
        eps = st.slider(
            "eps (Khoảng cách tối đa giữa hai điểm để coi là lân cận)", 
            0.1, 1.0, 0.5, 
            key="clustering_eps_slider"  # Thêm key duy nhất
        )
        min_samples = st.slider(
            "min_samples (Số lượng điểm tối thiểu trong một lân cận)", 
            1, 20, 5, 
            key="clustering_min_samples_slider"  # Thêm key duy nhất
        )
        model = DBSCAN(eps=eps, min_samples=min_samples)

    if st.button("Huấn luyện mô hình", key="clustering_train_button"):  # Thêm key duy nhất
        st.session_state["run_name"] = "training_run_1"  # Tạo giá trị cho run_name

        # Gọi hàm huấn luyện mô hình
        train_model(
            model=model,
            X_train_reduced=X_train_reduced,
            y_train=y_train,
            X_test_reduced=X_test_reduced,
            y_test=y_test,
            model_choice=model_choice,
            reduction_method=reduction_method,
            n_components=n_components,
            n_clusters=n_clusters if model_choice == "K-means" else None,
            eps=eps if model_choice == "DBSCAN" else None,
            min_samples=min_samples if model_choice == "DBSCAN" else None
        )


def du_doan():
    st.title("🔢 Dự đoán chữ số viết tay")

    if "models" not in st.session_state or not st.session_state["models"]:
        st.error("⚠️ Chưa có mô hình nào được huấn luyện. Hãy huấn luyện mô hình trước.")
        return

    model_names = [model["name"] for model in st.session_state["models"]]
    selected_model_name = st.selectbox(
        "🔍 Chọn mô hình đã huấn luyện:", 
        model_names, 
        key="clustering_model_selectbox"  # Thêm key duy nhất
    )
    selected_model = next(model["model"] for model in st.session_state["models"] if model["name"] == selected_model_name)

    uploaded_file = st.file_uploader(
        "📤 Tải lên ảnh chữ số viết tay (28x28 pixel)", 
        type=["png", "jpg", "jpeg"], 
        key="clustering_file_uploader"  # Thêm key duy nhất
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")
        image = ImageOps.invert(image)
        image = image.resize((28, 28))
        st.image(image, caption="Ảnh đã tải lên", use_column_width=True)

        img_array = np.array(image).reshape(1, -1) / 255.0
        prediction = selected_model.predict(img_array)
        st.success(f"🔢 Dự đoán: **{prediction[0]}**")

def show_experiment_selector():
    st.title("📊 MLflow Experiments")

    experiment_name = "Clustering"
    
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

    # Thêm key duy nhất cho selectbox
    selected_run_name = st.selectbox(
        "🔍 Chọn một run:", 
        run_names, 
        key="clustering_experiment_selectbox"  # Thêm key duy nhất
    )
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
        st.write("### 📂 Dataset:")
        st.write(f"📥 [Tải dataset]({dataset_path})")
    else:
        st.warning("⚠ Không tìm thấy thông tin cho run này.")


def Clusterting():
    st.title(" MNIST Clustering App")

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
    Clusterting()
