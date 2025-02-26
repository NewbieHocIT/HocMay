import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import mlflow
import dagshub
import os
# Khởi tạo kết nối với DagsHub

# Load MNIST dataset
@st.cache_data
def load_mnist():
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    x = np.concatenate((x_train, x_test), axis=0)
    x = x.reshape(x.shape[0], -1) / 255.0  # Normalize and flatten
    return x

def reduce_dimensionality(data, method, n_components):
    """ Giảm chiều dữ liệu bằng PCA hoặc t-SNE """
    if method == "PCA":
        reducer = PCA(n_components=n_components)
    elif method == "t-SNE":
        reducer = TSNE(n_components=n_components, random_state=42)
    
    return reducer.fit_transform(data)

def log_experiment(model_name):
    try:
        dagshub.init(repo_owner='NewbieHocIT', repo_name='MocMayvsPython', mlflow=True)
        os.environ['MLFLOW_TRACKING_USERNAME'] = 'NewbieHocIT'
        os.environ['MLFLOW_TRACKING_PASSWORD'] = '681dda9a41f9271a144aa94fa8624153a3c95696'
        mlflow.set_tracking_uri("https://dagshub.com/NewbieHocIT/MocMayvsPython.mlflow")
        client = mlflow.MlflowClient()
        experiment_name = "Clustering"
        experiment = next((exp for exp in client.search_experiments() if exp.name == experiment_name), None)
        if experiment:
            experiment_id = experiment.experiment_id
            print(f"Experiment ID: {experiment_id}")
            mlflow.set_experiment(experiment_name)
            with mlflow.start_run(run_name = model_name) as run:
                print('Logging...')
                mlflow.log_param("Cluster", st.session_state.num_clusters)
                mlflow.log_metric("Silhouette Score", 100)
                st.success(f"✅ Mô hình được log vào thí nghiệm: {experiment_name}")

        else:
            # Nếu thí nghiệm chưa tồn tại, tạo mới
            experiment_id = mlflow.create_experiment(experiment_name)
        print("Active Run:", mlflow.active_run())
    except Exception as e:
        st.warning("Không thể kết nối với MLflow hoặc DagsHub. Vui lòng kiểm tra cài đặt.")

def display():    
        # Initialize session state variables
        if "sample_size" not in st.session_state:
            st.session_state.sample_size = 1000
        if "train_size" not in st.session_state:
            st.session_state.train_size = 0.8
        if "num_clusters" not in st.session_state:
            st.session_state.num_clusters = 10
        if "eps" not in st.session_state:
            st.session_state.eps = 3.0
        if "min_samples" not in st.session_state:
            st.session_state.min_samples = 5
        if "train_clicked" not in st.session_state:
            st.session_state.train_clicked = False
        if "split_data_clicked" not in st.session_state:
            st.session_state.split_data_clicked = False
        if "dim_reduction_clicked" not in st.session_state:
            st.session_state.dim_reduction_clicked = False
        if "algorithm" not in st.session_state:
            st.session_state.algorithm = "K-Means"
        if "dim_reduction" not in st.session_state:
            st.session_state.dim_reduction = "PCA"
        if "n_components" not in st.session_state:
            st.session_state.n_components = 50
        if "log_clicked" not in st.session_state:
            st.session_state.log_clicked = False
        if "log_success" not in st.session_state:
            st.session_state.log_success = False
        st.title("Clustering on MNIST with Dimensionality Reduction")

        # Load data
        data = load_mnist()
        df = pd.DataFrame(data)
        st.dataframe(df.head())

        # Dataset selection
        st.write("## Dataset Options")
        st.session_state.sample_size = int(st.text_input("Number of samples (max = 70000)", st.session_state.sample_size))
        st.session_state.train_size = float(st.text_input("Train set ratio", st.session_state.train_size))

        if st.button("Xác nhận chia tập dữ liệu"):
            st.session_state.split_data_clicked = True

        if st.session_state.split_data_clicked:
            x_data = load_mnist()
            x_sampled = x_data[:st.session_state.sample_size]
            x_train, x_val = train_test_split(x_sampled, test_size=1 - st.session_state.train_size, random_state=42)
            st.session_state.x_train = x_train
            st.session_state.x_val = x_val
            st.write(f"**Kích thước tập Train:** {x_train.shape}")
            st.write(f"**Kích thước tập Test:** {x_val.shape}")
            st.success("Chia tập dữ liệu thành công!")

        # Choose dimensionality reduction method
        st.write("## Dimensionality Reduction")
        st.session_state.dim_reduction = st.selectbox("Select method", ["PCA", "t-SNE"])
        st.session_state.n_components = int(st.text_input("Number of dimensions", st.session_state.n_components))

        if st.button("Xác nhận giảm chiều dữ liệu"):
            if "x_train" in st.session_state:
                st.session_state.x_train = reduce_dimensionality(st.session_state.x_train, st.session_state.dim_reduction, st.session_state.n_components)
                st.session_state.x_val = reduce_dimensionality(st.session_state.x_val, st.session_state.dim_reduction, st.session_state.n_components)
                st.session_state.dim_reduction_clicked = True
                st.write(f"**Kích thước tập Train sau giảm chiều:** {st.session_state.x_train.shape}")
                st.write(f"**Kích thước tập Validation sau giảm chiều:** {st.session_state.x_val.shape}")
                st.success(f"Giảm chiều dữ liệu bằng {st.session_state.dim_reduction} xuống {st.session_state.n_components} chiều!")
            else:
                st.warning("Vui lòng chia tập dữ liệu trước!")

        # Select clustering algorithm
        st.session_state.algorithm = st.selectbox("Select Clustering Algorithm", ["K-Means", "DBSCAN"])
        
        if st.session_state.algorithm == "K-Means":
            st.session_state.num_clusters = int(st.text_input("Number of clusters", st.session_state.num_clusters))
        elif st.session_state.algorithm == "DBSCAN":
            st.session_state.eps = float(st.text_input("Epsilon (eps)", st.session_state.eps))
            st.session_state.min_samples = int(st.text_input("Min Samples", st.session_state.min_samples))

        if st.button("Train Model"):
            if "x_train" not in st.session_state or "x_val" not in st.session_state:
                st.warning("Vui lòng chia tập dữ liệu trước!")
                return
            if not st.session_state.dim_reduction_clicked:
                st.warning("Vui lòng giảm chiều dữ liệu trước!")
                return

            # Train clustering model
            if st.session_state.algorithm == "K-Means":
                st.write("Training K-Means...")
                model = KMeans(n_clusters=st.session_state.num_clusters, random_state=42, n_init=10)
                labels = model.fit_predict(st.session_state.x_train)
                inertia = model.inertia_
            else:
                st.write("Training DBSCAN...")
                model = DBSCAN(eps=st.session_state.eps, min_samples=st.session_state.min_samples)
                labels = model.fit_predict(st.session_state.x_train)
                inertia = "N/A"

            # In ra kích thước dữ liệu sau khi huấn luyện
            st.write(f"**Kích thước dữ liệu sau khi huấn luyện:** {st.session_state.x_train.shape}")

            # Display evaluation metrics
            st.write("### Evaluation Metrics")
            st.write(f"- **Inertia (WCSS):** {inertia}")
            sil = silhouette_score(st.session_state.x_train, labels) if len(set(labels)) > 1 else 'N/A'
            st.write(f"- **Silhouette Score:** {sil}")
            st.success("Training Completed!")

        model_name = st.text_input("🏷️ Nhập tên mô hình")
        if st.button("Log Experiment Clustering"):
            log_experiment(model_name) 

        # Hiển thị trạng thái log thành công
        if st.session_state.log_success:
            st.success("🚀 Experiment đã được log thành công!")
                
display()