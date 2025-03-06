import streamlit as st
import pandas as pd
import os
import plotly.express as px
import mlflow
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Load dữ liệu MNIST
def load_mnist():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data, mnist.target.astype(int)
    return X, y

# Thiết lập MLflow
def mlflow_input():
    DAGSHUB_MLFLOW_URI = "https://dagshub.com/NewbieHocIT/MocMayvsPython.mlflow"
    st.session_state['mlflow_url'] = DAGSHUB_MLFLOW_URI
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)

    os.environ["MLFLOW_TRACKING_USERNAME"] = "NewbieHocIT"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "681dda9a41f9271a144aa94fa8624153a3c95696"

    mlflow.set_experiment("PCA-tSNE")  # Đặt tên thí nghiệm là "PCA-tSNE"

# Giảm chiều dữ liệu
def reduce_dimensions(X, method='PCA', n_components=2):
    """
    Hàm giảm chiều dữ liệu sử dụng PCA hoặc t-SNE.
    
    Parameters:
    - X: Dữ liệu đầu vào (numpy array).
    - method: Phương pháp giảm chiều ('PCA' hoặc 't-SNE').
    - n_components: Số chiều sau khi giảm.
    
    Returns:
    - X_reduced: Dữ liệu sau khi giảm chiều.
    """
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
    """
    Hàm trực quan hóa dữ liệu sau khi giảm chiều sử dụng plotly.
    - Sử dụng bảng màu phong phú hơn để hiển thị các điểm dữ liệu.
    """
    # Tạo DataFrame từ dữ liệu giảm chiều
    df = pd.DataFrame(X_reduced, columns=[f"Component {i+1}" for i in range(n_components)])
    df['Digit'] = y  # Thêm cột nhãn (digit)

    # Nếu số chiều > 3, cho phép người dùng chọn 3 chiều để biểu diễn
    if n_components > 3:
        st.warning("⚠️ Số chiều > 3. Vui lòng chọn 3 chiều để biểu diễn.")
        col1, col2, col3 = st.columns(3)
        with col1:
            x_axis = st.selectbox("Chọn trục X", df.columns[:-1], index=0)
        with col2:
            y_axis = st.selectbox("Chọn trục Y", df.columns[:-1], index=1)
        with col3:
            z_axis = st.selectbox("Chọn trục Z", df.columns[:-1], index=2)
    else:
        x_axis, y_axis, z_axis = df.columns[0], df.columns[1], df.columns[2] if n_components == 3 else None

    # Tạo biểu đồ 3D hoặc 2D tùy thuộc vào số chiều
    if n_components >= 3:
        fig = px.scatter_3d(
            df, 
            x=x_axis, 
            y=y_axis, 
            z=z_axis, 
            color='Digit', 
            title="3D Visualization of Reduced Data",
            labels={'color': 'Digit'},
            color_continuous_scale=px.colors.sequential.Viridis  # Sử dụng bảng màu phong phú
        )
    else:
        fig = px.scatter(
            df, 
            x=x_axis, 
            y=y_axis, 
            color='Digit', 
            title="2D Visualization of Reduced Data",
            labels={'color': 'Digit'},
            color_continuous_scale=px.colors.sequential.Viridis  # Sử dụng bảng màu phong phú
        )

    # Hiển thị biểu đồ
    st.plotly_chart(fig, use_container_width=True)

# Hàm chính để chạy ứng dụng
def run_pca_tsne():
    st.title("PCA & t-SNE Visualization")

    # Thiết lập MLflow
    mlflow_input()

    # Tải dữ liệu MNIST
    X, y = load_mnist()

    # Chọn số lượng mẫu
    num_samples = st.slider(
        "Chọn số lượng mẫu để giảm chiều:", 
        1000, X.shape[0], 10000, 
        key="pca_tsne_num_samples_slider"
    )
    X_selected, _, y_selected, _ = train_test_split(X, y, train_size=num_samples, stratify=y, random_state=42)

    # Chọn phương pháp giảm chiều
    reduction_method = st.selectbox(
        "Chọn phương pháp giảm chiều:", 
        ["PCA", "t-SNE"], 
        key="pca_tsne_reduction_method_selectbox"
    )

    # Chọn số chiều
    n_components = st.slider(
        "Chọn số chiều sau khi giảm:", 
        2, 
        784 if reduction_method == "PCA" else 3,  # Giới hạn t-SNE tối đa là 3
        2,
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

        st.success("✅ Đã giảm chiều dữ liệu thành công!")

    # Kiểm tra nếu dữ liệu đã được giảm chiều
    if 'X_reduced' in st.session_state:
        # Trực quan hóa dữ liệu
        st.subheader("Trực quan hóa dữ liệu sau khi giảm chiều")
        visualize_data(st.session_state['X_reduced'], st.session_state['y_selected'], st.session_state['n_components'])

        # Phần đặt tên run và logging vào MLflow
        st.subheader("Log vào MLflow")
        run_name = st.text_input(
            "🔹 Nhập tên Run:", 
            "Default_Run", 
            key="pca_tsne_run_name_input"
        )

        if st.button("Log vào MLflow", key="pca_tsne_log_button"):
            with mlflow.start_run(run_name=run_name):
                mlflow.log_param("num_samples", num_samples)
                mlflow.log_param("reduction_method", reduction_method)
                mlflow.log_param("n_components", n_components)
                st.success(f"✅ Đã log dữ liệu vào MLflow với tên Run: {run_name}")

if __name__ == "__main__":
    run_pca_tsne()