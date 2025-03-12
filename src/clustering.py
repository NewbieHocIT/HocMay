import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image, ImageOps
import os
import mlflow
from datetime import datetime
from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

def load_mnist():
    X = np.load("data/mnist/X.npy")
    return X

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




from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

def split_data():
    st.title("📌 Chia dữ liệu (Unsupervised Learning)")

    # Tải dữ liệu MNIST
    X = load_mnist()
    total_samples = X.shape[0]
    if "clustering_split_done" not in st.session_state:
        st.session_state.clustering_split_done = False

    # Khởi tạo các thuộc tính trong session_state nếu chưa tồn tại
    if "test_size" not in st.session_state:
        st.session_state.test_size = 0.1  # Giá trị mặc định
    if "train_size" not in st.session_state:
        st.session_state.train_size = 0
    if "total_samples" not in st.session_state:
        st.session_state.total_samples = total_samples

    # Thanh kéo chọn số lượng ảnh để sử dụng
    num_samples = st.number_input("📌 Nhập số lượng ảnh để train:", min_value=1000, max_value=70000, value=20000, step=1000)



    if st.button("✅ Xác nhận & Lưu", key="split_data_confirm_button"):
        st.session_state.clustering_split_done = True  # Đánh dấu đã chia dữ liệu
        st.success("✅ Dữ liệu đã được chia thành công!")

        st.session_state.train_size = num_samples

        # Chọn số lượng ảnh mong muốn
        X_selected = X[:num_samples]

        # Chia train/test (nếu test_size > 0)
        # Nếu không chia test, sử dụng toàn bộ dữ liệu
        st.session_state["clustering_X_train"] = X_selected
        st.session_state["clustering_X_test"] = np.array([])  # Không có tập test
        st.success(f"🔹 Dữ liệu đã sẵn sàng: {len(X_selected)} ảnh")

    if "X_train" in st.session_state:
        st.write("📌 Dữ liệu đã sẵn sàng để sử dụng!")



def mlflow_input():
    DAGSHUB_MLFLOW_URI = "https://dagshub.com/NewbieHocIT/MocMayvsPython.mlflow"
    st.session_state['mlflow_url'] = DAGSHUB_MLFLOW_URI
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)

    os.environ["MLFLOW_TRACKING_USERNAME"] = "NewbieHocIT"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "681dda9a41f9271a144aa94fa8624153a3c95696"

    mlflow.set_experiment("Clustering")


def train():
    mlflow_input()

    # Kiểm tra dữ liệu đã được chia chưa
    if "clustering_X_train" not in st.session_state or "clustering_X_test" not in st.session_state:
        st.error("⚠️ Chưa có dữ liệu! Hãy chia dữ liệu trước.")
        return

    X_train = st.session_state["clustering_X_train"]
    X_test = st.session_state["clustering_X_test"]

    # Chuẩn hóa dữ liệu
    X_train = X_train.reshape(-1, 28 * 28) / 255.0
    X_test = X_test.reshape(-1, 28 * 28) / 255.0 if X_test.size > 0 else None

    st.header("⚙️ Chọn mô hình & Huấn luyện")

    model_choice = st.selectbox("Chọn mô hình:", ["K-means", "DBSCAN"], key="clustering_model_choice_selectbox")

    if model_choice == "K-means":
        n_clusters = st.slider("n_clusters", 2, 20, 10, key="clustering_n_clusters_slider")
        model = KMeans(n_clusters=n_clusters)
    elif model_choice == "DBSCAN":
        # Tham số mặc định tốt hơn cho DBSCAN với MNIST
        eps = st.slider("eps (Khoảng cách tối đa giữa hai điểm để coi là lân cận)", 0.1, 10.0, 4.2, step=0.1, key="clustering_eps_slider")
        min_samples = st.slider("min_samples (Số lượng điểm tối thiểu trong một lân cận)", 2, 50, 10, key="clustering_min_samples_slider")
        model = DBSCAN(eps=eps, min_samples=min_samples)

    run_name = st.text_input("🔹 Nhập tên Run:", "Default_Run", key="clustering_run_name_input")
    st.session_state["run_name"] = run_name if run_name else "default_run"

    if st.button("Huấn luyện mô hình", key="clustering_train_button"):
        with mlflow.start_run(run_name=f"Train_{st.session_state['run_name']}"):
            # Các bước log param như cũ
            mlflow.log_param("test_size", st.session_state.test_size)
            mlflow.log_param("train_size", st.session_state.train_size)
            mlflow.log_param("num_samples", st.session_state.total_samples)

            progress_bar = st.progress(0)
            status_text = st.empty()



    # Huấn luyện mô hình
            status_text.text("⏳ Đang huấn luyện mô hình...")
            start_time = time.time()
            model.fit(X_train)
            training_time = time.time() - start_time
            labels = model.labels_ if hasattr(model, "labels_") else model.predict(X_train)
            st.session_state["clustering_labels"] = labels

            # Giả lập thanh tiến trình mượt mà
            total_steps = 100
            for i in range(total_steps + 1):
                progress = min(i / total_steps, 0.5)  # 0% -> 50% cho huấn luyện
                progress_bar.progress(progress)
                time.sleep(training_time / (2 * total_steps))  # Điều chỉnh tốc độ dựa trên thời gian thực

            # Tính silhouette score
            status_text.text("📊 Đang tính toán silhouette score...")
            if len(np.unique(labels)) > 1 and -1 not in labels:
                silhouette_avg = silhouette_score(X_train, labels)
                st.success(f"📊 **Silhouette Score**: {silhouette_avg:.4f}")
                mlflow.log_metric("silhouette_score", silhouette_avg)
            # Tiếp tục tăng progress từ 50% đến 80%
            for i in range(total_steps // 2, int(total_steps * 0.8)):
                progress = i / total_steps
                progress_bar.progress(progress)
                time.sleep(0.01)

            # Logging MLflow và lưu mô hình
            status_text.text("📝 Đang ghi log vào MLflow...")
            mlflow.log_param("model", model_choice)
            if model_choice == "K-means":
                mlflow.log_param("n_clusters", n_clusters)
            elif model_choice == "DBSCAN":
                mlflow.log_param("eps", eps)
                mlflow.log_param("min_samples", min_samples)
            mlflow.sklearn.log_model(model, model_choice.lower())

            # Tiến trình từ 80% đến 100%
            status_text.text("💾 Đang lưu mô hình...")
            for i in range(int(total_steps * 0.8), total_steps + 1):
                progress = i / total_steps
                progress_bar.progress(progress)
                time.sleep(0.01)

            if "clustering_models" not in st.session_state:
                st.session_state["clustering_models"] = []

            model_name = model_choice.lower().replace(" ", "_")
            if model_choice == "DBSCAN":
                model_name += f"_eps{eps}_min_samples{min_samples}"
            elif model_choice == "K-means":
                model_name += f"_n_clusters{n_clusters}"

            existing_model = next((item for item in st.session_state["clustering_models"] if item["name"] == model_name), None)

            if existing_model:
                count = 1
                new_model_name = f"{model_name}_{count}"
                while any(item["name"] == new_model_name for item in st.session_state["clustering_models"]):
                    count += 1
                    new_model_name = f"{model_name}_{count}"
                model_name = new_model_name
                st.warning(f"⚠️ Mô hình được lưu với tên: {model_name}")

            st.session_state["clustering_models"].append({"name": model_name, "model": model})

            # Hiển thị thông tin bổ sung cho DBSCAN
            if model_choice == "DBSCAN":
                num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                num_noise = list(labels).count(-1)
                st.write(f"🔢 Số lượng cụm: {num_clusters}")
                st.write(f"🔢 Số lượng điểm nhiễu: {num_noise}")

            # Hiển thị silhouette score (nếu đã tính)
            if "silhouette_avg" in locals() and len(np.unique(labels)) > 1 and -1 not in labels:
                st.write(f"📊 **Silhouette Score**: {silhouette_avg:.4f}")
            elif model_choice == "DBSCAN" and -1 in labels:
                # Tính silhouette score loại bỏ nhiễu cho DBSCAN
                if num_clusters > 1:  # Chỉ tính nếu có hơn 1 cụm hợp lệ
                    mask = labels != -1  # Lọc bỏ các điểm nhiễu
                    if mask.sum() > 0:  # Đảm bảo còn dữ liệu sau khi lọc
                        silhouette_avg_no_noise = silhouette_score(X_train[mask], labels[mask])
                        st.write(f"📊 **Silhouette Score**: {silhouette_avg_no_noise:.4f}")
                    else:
                        st.write("📊 Không thể tính Silhouette Score: Không đủ điểm dữ liệu sau khi loại bỏ nhiễu.")
                else:
                    st.write("📊 Không thể tính Silhouette Score: Chỉ có 1 cụm hoặc toàn bộ là nhiễu.")
            else:
                st.write("📊 Không thể tính Silhouette Score: Chỉ có 1 cụm.")

            st.write(f"🔹 Mô hình đã được lưu với tên: {model_name}")
            st.write(f"Tổng số mô hình hiện tại: {len(st.session_state['clustering_models'])}")

            st.write("📋 Danh sách các mô hình đã lưu:")
            model_names = [model["name"] for model in st.session_state["clustering_models"]]
            st.write(", ".join(model_names))

            st.success(f"✅ Đã log dữ liệu cho **Train_{st.session_state['run_name']}**!")
            status_text.text("💾 Đã lưu")
            progress_bar.progress(100)
            
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
import plotly.express as px

import time  # Thêm thư viện time để mô phỏng tiến trình

import plotly.express as px
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans, DBSCAN
import time

def visualize_clusters():
    st.title("🔢 Trực quan hóa các cụm từ mô hình đã huấn luyện")

    # Kiểm tra mô hình đã huấn luyện
    if "clustering_models" not in st.session_state or not st.session_state["clustering_models"]:
        st.error("⚠️ Chưa có mô hình nào được huấn luyện. Hãy huấn luyện mô hình trước.")
        return

    # Chọn mô hình
    model_names = [model["name"] for model in st.session_state["clustering_models"]]
    selected_model_name = st.selectbox("🔍 Chọn mô hình đã huấn luyện:", model_names)
    selected_model = next(model["model"] for model in st.session_state["clustering_models"] if model["name"] == selected_model_name)

    # Kiểm tra nếu đã có nhãn cụm từ quá trình huấn luyện
    if "clustering_labels" not in st.session_state:
        st.error("⚠️ Chưa có nhãn cụm được lưu. Hãy đảm bảo mô hình đã được huấn luyện và lưu nhãn.")
        return
    
    labels = st.session_state["clustering_labels"]

    # Chọn kiểu trực quan
    plot_type = st.radio("Chọn kiểu trực quan:", ["2D", "3D"])

    # Nút bắt đầu trực quan hóa
    if st.button("Bắt đầu trực quan hóa"):
        # Khởi tạo thanh tiến trình và trạng thái
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Bước 1: Giảm chiều dữ liệu bằng PCA
        status_text.text("⏳ Đang giảm chiều dữ liệu...")
        for percent_complete in range(20):  # Tăng dần từ 0% đến 20%
            time.sleep(0.05)
            progress_bar.progress(percent_complete + 1)

        # Lấy dữ liệu từ session_state
        X_train = st.session_state["clustering_X_train"]
        X_train = X_train.reshape(-1, 28 * 28) / 255.0

        # Giảm chiều xuống 3D bằng PCA
        reducer = PCA(n_components=3, random_state=42)
        X_reduced = reducer.fit_transform(X_train)

        # Bước 2: Chuẩn bị dữ liệu
        status_text.text("⏳ Đang chuẩn bị dữ liệu để vẽ biểu đồ...")
        for percent_complete in range(20, 50):  # Tăng dần từ 20% đến 50%
            time.sleep(0.05)
            progress_bar.progress(percent_complete + 1)

        # Tải nhãn gốc từ MNIST (giả định bạn có hàm load_mnist trả về X và y)
        from sklearn.datasets import fetch_openml
        X_mnist, y_mnist = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
        y_mnist = y_mnist[:len(X_train)].astype(int)  # Chỉ lấy số lượng tương ứng với X_train

        # Chuyển thành DataFrame
        df = pd.DataFrame(X_reduced, columns=['X1', 'X2', 'X3'])
        df['Cluster'] = labels.astype(str)  # Chuyển nhãn cụm thành chuỗi để Plotly coi là phân loại
        df['Original_Label'] = y_mnist  # Nhãn gốc từ MNIST

        # Bước 3: Vẽ biểu đồ
        status_text.text("⏳ Đang vẽ biểu đồ...")
        for percent_complete in range(50, 90):  # Tăng dần từ 50% đến 90%
            time.sleep(0.05)
            progress_bar.progress(percent_complete + 1)

        if plot_type == "2D":
            plt.figure(figsize=(10, 8))
            sns.scatterplot(x='X1', y='X2', hue='Cluster', data=df, palette='tab10', legend='full')
            plt.xlabel("X1")
            plt.ylabel("X2")
            plt.title("Trực quan hóa cụm bằng PCA (2D)")
            st.pyplot(plt)
        else:
            # Tùy chỉnh biểu đồ 3D với màu riêng biệt cho từng cụm
            fig = px.scatter_3d(
                df, 
                x='X1', 
                y='X2', 
                z='X3', 
                color='Cluster',  # Màu theo cụm (phân loại)
                title="Trực quan hóa cụm bằng PCA (3D)",
                hover_data={'Original_Label': True, 'Cluster': True},  # Hiển thị nhãn gốc và nhãn dự đoán khi hover
                opacity=0.7,  # Độ trong suốt để dễ nhìn
                symbol='Cluster',  # Dùng biểu tượng khác nhau cho từng cụm (tùy chọn)
            )
            # Tùy chỉnh giao diện
            fig.update_traces(marker=dict(size=5))  # Kích thước điểm
            fig.update_layout(
                scene=dict(
                    xaxis_title='X1',
                    yaxis_title='X2',
                    zaxis_title='X3',
                    bgcolor='rgba(0,0,0,0)',  # Nền trong suốt
                ),
                margin=dict(l=0, r=0, b=0, t=40),  # Giảm lề
                title_x=0.5,  # Căn giữa tiêu đề
                legend_title_text='Cụm',  # Tiêu đề legend
                coloraxis_showscale=False,  # Ẩn thanh màu gradient
            )
            st.plotly_chart(fig, use_container_width=True)

        # Bước 4: Hiển thị thông tin mô hình
        status_text.text("⏳ Đang hiển thị thông tin mô hình...")
        for percent_complete in range(90, 100):  # Tăng dần từ 90% đến 100%
            time.sleep(0.05)
            progress_bar.progress(percent_complete + 1)

        # Hiển thị thông tin mô hình
        st.write("📋 **Thông tin mô hình:**")
        st.write(f"- Tên mô hình: **{selected_model_name}**")
        st.write(f"- Loại mô hình: **{type(selected_model).__name__}**")

        if isinstance(selected_model, KMeans):
            st.write("🔢 Số lượng cụm: **{}**".format(selected_model.n_clusters))
            st.write("🔢 Tâm cụm (centroids):")
            st.write(selected_model.cluster_centers_)
        elif isinstance(selected_model, DBSCAN):
            num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            num_noise = np.sum(labels == -1)
            st.write(f"🔢 Số lượng cụm: **{num_clusters}**")
            st.write(f"🔢 Số lượng điểm nhiễu (noise): **{num_noise}**")

        # Hoàn thành
        status_text.text("✅ Hoàn thành trực quan hóa!")
        progress_bar.progress(100)
def show_experiment_selector():
    st.title("📊 MLflow Experiments")

    experiment_name = "Clustering"
    
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
    selected_run_name = st.selectbox("🔍 Chọn một run:", list(run_dict.keys()),key="run_selector_selectbox" )
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


def Clustering():
    st.title("🖊️ MNIST Clustering App")

    tab1, tab2, tab3, tab4 = st.tabs(["📘 Data", "⚙️ Huấn luyện", "🔢 Trực quan hóa", "🔥Mlflow"])

    with tab1:
        data()
        
    with tab2:
        split_data()
        train()
        
    with tab3:
        visualize_clusters()   
    with tab4:
        show_experiment_selector()  

if __name__ == "__main__":
    Clustering()