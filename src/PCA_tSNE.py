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

# Giảm chiều dữ liệu với callback tiến trình
def reduce_dimensions(X, method='PCA', n_components=2, progress_callback=None):
    if method == 'PCA':
        reducer = PCA(n_components=n_components)
    elif method == 't-SNE':
        reducer = TSNE(n_components=n_components, perplexity=30, n_iter=300)
    else:
        raise ValueError("Phương pháp giảm chiều không hợp lệ. Chọn 'PCA' hoặc 't-SNE'.")
    
    # Giả lập tiến trình trong quá trình giảm chiều (vì PCA/t-SNE không có callback gốc)
    X_reduced = reducer.fit_transform(X)
    if progress_callback:
        progress_callback(50)  # Giảm chiều hoàn tất, đạt 50%
    return X_reduced

# Trực quan hóa dữ liệu với tiến trình
def visualize_data(X_reduced, y, n_components, progress_bar, status_text):
    with st.spinner("⏳ Đang chuẩn bị dữ liệu để vẽ..."):
        df = pd.DataFrame(X_reduced, columns=[f"Component {i+1}" for i in range(n_components)])
        df['Digit'] = y.astype(str)
        progress_bar.progress(60)  # Chuẩn bị dữ liệu xong, 60%
        time.sleep(0.1)  # Đảm bảo spinner hiển thị

    # Nếu số chiều > 3, chọn trục
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

    status_text.text("⏳ Đang tạo biểu đồ...")
    progress_bar.progress(80)  # Bắt đầu tạo biểu đồ, 80%
    if n_components >= 3:
        fig = px.scatter_3d(
            df,
            x=x_axis,
            y=y_axis,
            z=z_axis,
            color='Digit',
            title="3D Visualization of Reduced Data",
            hover_data={x_axis: ':.2f', y_axis: ':.2f', z_axis: ':.2f', 'Digit': True},
            opacity=0.7,
            color_discrete_sequence=px.colors.qualitative.Set1,
        )
        fig.update_traces(marker=dict(size=3))
        fig.update_layout(
            scene=dict(xaxis_title=x_axis, yaxis_title=y_axis, zaxis_title=z_axis, bgcolor='rgba(0,0,0,0)'),
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
            hover_data={x_axis: ':.2f', y_axis: ':.2f', 'Digit': True},
        )
        fig.update_traces(marker=dict(size=5))

    status_text.text("⏳ Đang hiển thị biểu đồ...")
    progress_bar.progress(90)  # Biểu đồ đã tạo xong, 90%
    st.plotly_chart(fig, use_container_width=True)

    status_text.text("✅ Hoàn thành trực quan hóa!")
    progress_bar.progress(100)

def show_experiment_selector():
    st.title("📊 MLflow Experiments")
    experiment_name = "PCA-tSNE"
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
    run_dict = {run.get("tags.mlflow.runName", f"Run {run['run_id'][:8]}"): run["run_id"] for _, run in runs.iterrows()}
    selected_run_name = st.selectbox("🔍 Chọn một run:", list(run_dict.keys()), key="runname")
    selected_run_id = run_dict[selected_run_name]
    selected_run = mlflow.get_run(selected_run_id)

    if selected_run:
        st.subheader(f"📌 Thông tin Run: {selected_run_name}")
        st.write(f"**Run ID:** {selected_run_id}")
        st.write(f"**Trạng thái:** {selected_run.info.status}")
        start_time_ms = selected_run.info.start_time
        start_time = datetime.fromtimestamp(start_time_ms / 1000).strftime("%Y-%m-%d %H:%M:%S") if start_time_ms else "Không có thông tin"
        st.write(f"**Thời gian chạy:** {start_time}")
        if selected_run.data.params:
            st.write("### ⚙️ Parameters:")
            st.json(selected_run.data.params)
        if selected_run.data.metrics:
            st.write("### 📊 Metrics:")
            st.json(selected_run.data.metrics)
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
        # Khởi tạo thanh tiến trình và trạng thái
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Bước 1: Giảm chiều
        status_text.text("⏳ Đang giảm chiều dữ liệu...")
        progress_bar.progress(10)  # Bắt đầu giảm chiều, 10%
        X_reduced = reduce_dimensions(
            X_selected, 
            method=reduction_method, 
            n_components=n_components, 
            progress_callback=lambda x: progress_bar.progress(x)
        )

        # Lưu kết quả vào session_state
        st.session_state['X_reduced'] = X_reduced
        st.session_state['y_selected'] = y_selected
        st.session_state['n_components'] = n_components
        st.session_state['visualized'] = False

        # Bước 2: Trực quan hóa
        st.subheader("Trực quan hóa dữ liệu sau khi giảm chiều")
        visualize_data(st.session_state['X_reduced'], st.session_state['y_selected'], st.session_state['n_components'], progress_bar, status_text)
        st.session_state['visualized'] = True

        st.success("✅ Đã giảm chiều và trực quan hóa dữ liệu thành công!")

    # Kiểm tra nếu dữ liệu đã được giảm chiều
    if 'X_reduced' in st.session_state and st.session_state.get('visualized', False):
        # Phần đặt tên run và logging vào MLflow
        st.subheader("Log vào MLflow")
        run_name = st.text_input(
            "🔹 Nhập tên Run:",
            "Default_Run",
            key="pca_tsne_run_name_input"
        )

        if st.button("Log vào MLflow", key="pca_tsne_log_button"):
            with st.spinner("⏳ Đang logging vào MLflow..."):
                mlflow_input()
                with mlflow.start_run(run_name=run_name):
                    mlflow.log_param("num_samples", num_samples)
                    mlflow.log_param("reduction_method", reduction_method)
                    mlflow.log_param("n_components", n_components)
                st.success(f"✅ Đã log dữ liệu vào MLflow với tên Run: {run_name}")

def pce():
    st.title("🖊️ PCA và t-SNE ")
    tab1, tab2 = st.tabs(["📘 Giảm chiều dữ liệu", "🔥 Mlflow"])

    with tab1:
        run_pca_ts
