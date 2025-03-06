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
    num_samples = st.slider(
        "Chọn số lượng ảnh để sử dụng:", 
        min_value=1000, 
        max_value=total_samples, 
        value=10000
    )

    # Thanh kéo chọn tỷ lệ Train/Test (nếu cần)
    test_size = st.slider(
        "Chọn tỷ lệ test (Để đánh giá)", 
        min_value=0.0, 
        max_value=0.5, 
        value=0.1, 
        step=0.1
    )

    if st.button("✅ Xác nhận & Lưu", key="split_data_confirm_button"):
        st.session_state.clustering_split_done = True  # Đánh dấu đã chia dữ liệu
        st.success("✅ Dữ liệu đã được chia thành công!")

        st.session_state.test_size = test_size
        st.session_state.train_size = num_samples * (1 - test_size)

        # Chọn số lượng ảnh mong muốn
        X_selected = X[:num_samples]

        # Chia train/test (nếu test_size > 0)
        if test_size > 0:
            X_train, X_test = train_test_split(X_selected, test_size=test_size, random_state=42)
            st.session_state["clustering_X_train"] = X_train
            st.session_state["clustering_X_test"] = X_test
            st.success(f"🔹 Dữ liệu đã được chia: Train ({len(X_train)}), Test ({len(X_test)})")
        else:
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

    # Kiểm tra xem dữ liệu đã được chia chưa (sử dụng key "clustering_X_train")
    if "clustering_X_train" not in st.session_state or "clustering_X_test" not in st.session_state:
        st.error("⚠️ Chưa có dữ liệu! Hãy chia dữ liệu trước.")
        return

    # Lấy dữ liệu từ session_state
    X_train = st.session_state["clustering_X_train"]
    X_test = st.session_state["clustering_X_test"]

    # Chuẩn hóa dữ liệu
    X_train = X_train.reshape(-1, 28 * 28) / 255.0
    X_test = X_test.reshape(-1, 28 * 28) / 255.0 if X_test.size > 0 else None

    st.header("⚙️ Chọn mô hình & Huấn luyện")

    model_choice = st.selectbox(
        "Chọn mô hình:", 
        ["K-means", "DBSCAN"], 
        key="clustering_model_choice_selectbox"  # Thêm key duy nhất
    )

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
            0.01, 1.0, 0.5, 
            key="clustering_eps_slider"  # Thêm key duy nhất
        )
        min_samples = st.slider(
            "min_samples (Số lượng điểm tối thiểu trong một lân cận)", 
            2, 20, 5, 
            key="clustering_min_samples_slider"  # Thêm key duy nhất
        )
        model = DBSCAN(eps=eps, min_samples=min_samples)

    run_name = st.text_input(
        "🔹 Nhập tên Run:", 
        "Default_Run", 
        key="clustering_run_name_input"  # Thêm key duy nhất
    )
    st.session_state["run_name"] = run_name if run_name else "default_run"

    if st.button("Huấn luyện mô hình", key="clustering_train_button"):  # Thêm key duy nhất
        with mlflow.start_run(run_name=f"Train_{st.session_state['run_name']}"):
            mlflow.log_param("test_size", st.session_state.test_size)
            mlflow.log_param("train_size", st.session_state.train_size)
            mlflow.log_param("num_samples", st.session_state.total_samples)

            progress_bar = st.progress(0)  # Thanh tiến trình
            status_text = st.empty()  # Hiển thị trạng thái từng bước

            # Bước 1: Huấn luyện mô hình
            status_text.text("⏳ Đang huấn luyện mô hình...")
            progress_bar.progress(30)

            model.fit(X_train)
            labels = model.labels_

            # Bước 2: Tính toán silhouette score
            status_text.text("📊 Đang tính toán silhouette score...")
            progress_bar.progress(60)

            if len(np.unique(labels)) > 1:
                silhouette_avg = silhouette_score(X_train, labels)
                st.success(f"📊 **Silhouette Score**: {silhouette_avg:.4f}")
                mlflow.log_metric("silhouette_score", silhouette_avg)
            else:
                st.warning("⚠ Không thể tính silhouette score vì chỉ có một cụm.")

            # Bước 3: Logging với MLflow
            status_text.text("📝 Đang ghi log vào MLflow...")
            progress_bar.progress(80)

            mlflow.log_param("model", model_choice)
            if model_choice == "K-means":
                mlflow.log_param("n_clusters", n_clusters)
            elif model_choice == "DBSCAN":
                mlflow.log_param("eps", eps)
                mlflow.log_param("min_samples", min_samples)

            mlflow.sklearn.log_model(model, model_choice.lower())

            # Bước 4: Lưu mô hình vào session_state
            status_text.text("💾 Đang lưu mô hình...")
            progress_bar.progress(90)

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
            st.write(f"🔹 Mô hình đã được lưu với tên: {model_name}")
            st.write(f"Tổng số mô hình hiện tại: {len(st.session_state['clustering_models'])}")

            st.write("📋 Danh sách các mô hình đã lưu:")
            model_names = [model["name"] for model in st.session_state["clustering_models"]]
            st.write(", ".join(model_names))

            st.success(f"✅ Đã log dữ liệu cho **Train_{st.session_state['run_name']}**!")
            status_text.text("💾 Đã lưu")
            progress_bar.progress(100)

from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps

def du_doan():
    st.title("🔢 Dự đoán phân cụm")

    # Kiểm tra xem đã có mô hình chưa
    if "clustering_models" not in st.session_state or not st.session_state["clustering_models"]:
        st.error("⚠️ Chưa có mô hình nào được huấn luyện. Hãy huấn luyện mô hình trước.")
        return

    # Chọn mô hình
    model_names = [model["name"] for model in st.session_state["clustering_models"]]
    selected_model_name = st.selectbox("🔍 Chọn mô hình đã huấn luyện:", model_names)
    selected_model = next(model["model"] for model in st.session_state["clustering_models"] if model["name"] == selected_model_name)

    # Chọn phương thức nhập ảnh
    input_option = st.radio("🖼 Chọn phương thức nhập:", ["Tải lên ảnh", "Vẽ số"], 
                            horizontal=True,
                            key="input_option_radio"  # Thêm key
                            )

    img_array = None  # Lưu ảnh đầu vào

    if input_option == "Tải lên ảnh":
        uploaded_file = st.file_uploader("📤 Tải lên ảnh chữ số viết tay (28x28 pixel)", 
                                         type=["png", "jpg", "jpeg"],key="file_uploader" )
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert("L")
                image = ImageOps.invert(image)
                image = image.resize((28, 28))
                st.image(image, caption="Ảnh đã tải lên", use_column_width=False)

                img_array = np.array(image).reshape(1, -1) / 255.0

            except Exception as e:
                st.error(f"❌ Lỗi xử lý ảnh: {str(e)}")

    elif input_option == "Vẽ số":
        st.write("✏️ Vẽ số bên dưới (dùng chuột hoặc cảm ứng):")
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=10,
            stroke_color="white",
            background_color="black",
            width=280,
            height=280,
            drawing_mode="freedraw",
            key="canvas"
        )

        if canvas_result.image_data is not None:
            try:
                image = Image.fromarray((canvas_result.image_data[:, :, 0]).astype(np.uint8))
                image = image.resize((28, 28)).convert("L")
                image = ImageOps.invert(image)
                st.image(image, caption="Ảnh vẽ đã được xử lý", use_column_width=False)

                img_array = np.array(image).reshape(1, -1) / 255.0

            except Exception as e:
                st.error(f"❌ Lỗi xử lý ảnh vẽ tay: {str(e)}")

    # Nút dự đoán
    if img_array is not None:
        if st.button("🚀 Dự đoán",key="predict_button"):
            if isinstance(selected_model, DBSCAN):
                st.warning("⚠️ DBSCAN không hỗ trợ dự đoán trực tiếp.")
                st.write("🔢 Nhãn cụm từ quá trình huấn luyện:")
                st.write(selected_model.labels_)

                num_noise = np.sum(selected_model.labels_ == -1)
                st.write(f"🔢 Số lượng điểm nhiễu (noise): **{num_noise}**")

            elif isinstance(selected_model, KMeans):
                prediction = selected_model.predict(img_array)
                st.success(f"🔢 Dự đoán nhãn cụm: **{prediction[0]}**")
                st.write("🔢 Tâm cụm (centroids):")
                st.write(selected_model.cluster_centers_)

            else:
                st.error("⚠️ Mô hình không được hỗ trợ trong chức năng này.")

            # Hiển thị thông tin mô hình
            st.write("📋 **Thông tin mô hình:**")
            st.write(f"- Tên mô hình: **{selected_model_name}**")
            st.write(f"- Loại mô hình: **{type(selected_model).__name__}**")



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
    Clustering()