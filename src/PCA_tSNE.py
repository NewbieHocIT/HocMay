import streamlit as st
import pandas as pd
import os
import plotly.express as px
from datetime import datetime
import joblib
import mlflow
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import time

import pickle

def load_mnist():
    """
    Đọc dữ liệu MNIST từ file .pkl.
    """
    with open("data/mnist/X.pkl", "rb") as f:
        X = pickle.load(f)
    with open("data/mnist/y.pkl", "rb") as f:
        y = pickle.load(f)
    return X, y

# Thiết lập MLflow
def mlflow_input():
    DAGSHUB_MLFLOW_URI = "https://dagshub.com/NewbieHocIT/MocMayvsPython.mlflow"
    st.session_state['mlflow_url'] = DAGSHUB_MLFLOW_URI
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
    os.environ["MLFLOW_TRACKING_USERNAME"] = "NewbieHocIT"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "681dda9a41f9271a144aa94fa8624153a3c95696"
    mlflow.set_experiment("PCA-tSNE")

# Giảm chiều dữ liệu
def reduce_dimensions(X, method='PCA', n_components=2):
    if method == 'PCA':
        reducer = PCA(n_components=n_components)
    elif method == 't-SNE':
        reducer = TSNE(n_components=n_components, perplexity=30, n_iter=300)
    else:
        raise ValueError("Phương pháp giảm chiều không hợp lệ. Chọn 'PCA' hoặc 't-SNE'.")
    X_reduced = reducer.fit_transform(X)
    return X_reduced

# Trực quan hóa dữ liệu
def visualize_data(X_reduced, y, n_components):
    # Khởi tạo thanh tiến trình
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Bước 1: Chuẩn bị dữ liệu
    status_text.text("⏳ Đang chuẩn bị dữ liệu để vẽ...")
    df = pd.DataFrame(X_reduced, columns=[f"Component {i+1}" for i in range(n_components)])
    df['Digit'] = y.astype(str)  # Chuyển nhãn thành chuỗi để coi là phân loại
    for i in range(0, 30):
        time.sleep(0.01)
        progress_bar.progress(i + 1)

    # Nếu số chiều > 3, cho phép người dùng chọn 3 chiều để biểu diễn
    if n_components > 3:
        st.warning("⚠️ Số chiều > 3. Vui lòng chọn 3 chiều để biểu diễn.")
        col1, col2, col3 = st.columns(3)
        with col1:
            x_axis = st.selectbox("Chọn trục X", df.columns[:-1], index=0, key="x_axis")
        with col2:
            y_axis = st.selectbox("Chọn trục Y", df.columns[:-1], index=1, key="y_axis")
        with col3:
            z_axis = st.selectbox("Chọn trục Z", df.columns[:-1], index=2, key="z_axis")
    else:
        x_axis = df.columns[0]
        y_axis = df.columns[1]
        z_axis = df.columns[2] if n_components >= 3 else None

    # Bước 2: Tạo biểu đồ
    status_text.text("⏳ Đang tạo biểu đồ...")
    if n_components >= 3:
        fig = px.scatter_3d(
            df,
            x=x_axis,
            y=y_axis,
            z=z_axis,
            color='Digit',
            title="3D Visualization of Reduced Data",
            hover_data={
                x_axis: ':.2f',
                y_axis: ':.2f',
                z_axis: ':.2f',
                'Digit': True
            },
            opacity=0.7,
            color_discrete_sequence=px.colors.qualitative.Set1,
        )
        fig.update_traces(marker=dict(size=3))
        fig.update_layout(
            scene=dict(
                xaxis_title=x_axis,
                yaxis_title=y_axis,
                zaxis_title=z_axis,
                bgcolor='rgba(0,0,0,0)',
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            title_x=0.5,
            legend_title_text='Digit',
            showlegend=True,
        )
    else:
        fig = px.scatter(
            df,
            x=x_axis,
            y=y_axis,
            color='Digit',
            title="2D Visualization of Reduced Data",
            color_discrete_sequence=px.colors.qualitative.Set1,
            hover_data={
                x_axis: ':.2f',
                y_axis: ':.2f',
                'Digit': True
            },
        )
        fig.update_traces(marker=dict(size=5))
    for i in range(30, 70):
        time.sleep(0.01)
        progress_bar.progress(i + 1)

    # Bước 3: Hiển thị biểu đồ
    status_text.text("⏳ Đang hiển thị biểu đồ...")
    st.plotly_chart(fig, use_container_width=True)
    for i in range(70, 100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)

    status_text.text("✅ Hoàn thành trực quan hóa!")
    progress_bar.progress(100)
def show_experiment_selector():
    st.title("📊 MLflow Experiments")

    experiment_name = "PCA-tSNE"
    
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
    selected_run_name = st.selectbox("🔍 Chọn một run:", list(run_dict.keys()),key="runname")
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
# Hàm chính để chạy ứng dụng
def run_pca_tsne():
    st.title("📌 Giảm chiều dữ liệu")

    # Đọc dữ liệu
    X, y = load_mnist()
    total_samples = X.shape[0]

    # Chọn số lượng mẫu
    num_samples = st.number_input("📌 Nhập số lượng ảnh:", min_value=1000, max_value=70000, value=10000, step=1000)

    # Nếu chọn toàn bộ dữ liệu, không cần giảm
    if num_samples == total_samples:
        X_selected, y_selected = X, y
    else:
        X_selected, _, y_selected, _ = train_test_split(
            X, y, train_size=num_samples, stratify=y, random_state=42
        )

    st.success(f"✅ Đã chọn {num_samples} mẫu từ {total_samples} dữ liệu.")

    # Chọn phương pháp giảm chiều
    reduction_method = st.selectbox(
        "Chọn phương pháp giảm chiều:",
        ["PCA", "t-SNE"],
        key="pca_tsne_reduction_method_selectbox"
    )

    # Chọn số chiều
    n_components = st.number_input(
        "Chọn số chiều sau khi giảm:",
        min_value=1,
        max_value=784,
        value=3,
        step=1,
        key="pca_tsne_n_components_slider"
    )

    # Nút giảm chiều
    if st.button("Giảm chiều", key="pca_tsne_reduce_button"):
        # Giảm chiều dữ liệu
        X_reduced = reduce_dimensions(X_selected, method=reduction_method, n_components=n_components)

        # Lưu kết quả vào session_state để sử dụng lại
        st.session_state['X_reduced'] = X_reduced
        st.session_state['y_selected'] = y_selected
        st.session_state['n_components'] = n_components
        st.session_state['visualized'] = False  # Đánh dấu chưa trực quan hóa

        st.success("✅ Đã giảm chiều dữ liệu thành công!")

    # Kiểm tra nếu dữ liệu đã được giảm chiều
    if 'X_reduced' in st.session_state:
        # Trực quan hóa dữ liệu chỉ khi chưa được vẽ
        if not st.session_state.get('visualized', False):
            st.subheader("Trực quan hóa dữ liệu sau khi giảm chiều")
            visualize_data(st.session_state['X_reduced'], st.session_state['y_selected'], st.session_state['n_components'])
            st.session_state['visualized'] = True  # Đánh dấu đã trực quan hóa

        # Phần đặt tên run và logging vào MLflow
        st.subheader("Log vào MLflow")
        run_name = st.text_input(
            "🔹 Nhập tên Run:",
            "Default_Run",
            key="pca_tsne_run_name_input"
        )

        if st.button("Log vào MLflow", key="pca_tsne_log_button"):
            mlflow_input()
            with mlflow.start_run(run_name=run_name):
                mlflow.log_param("num_samples", num_samples)
                mlflow.log_param("reduction_method", reduction_method)
                mlflow.log_param("n_components", n_components)
                st.success(f"✅ Đã log dữ liệu vào MLflow với tên Run: {run_name}")

def pce():
    st.title("🖊️ PCA và t-SNE ")
    tab1, tab2= st.tabs(["📘 Giảm chiều dữ liệu","🔥Mlflow"])

    with tab1:
        run_pca_tsne()
        
    with tab2:
        show_experiment_selector()
if __name__ == "__main__":
    pce()