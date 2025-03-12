import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from PIL import Image
import pandas as pd
import os
import mlflow
from datetime import datetime
from sklearn.model_selection import cross_val_score
import plotly.graph_objects as go
import time
from streamlit_drawable_canvas import st_canvas

# Load dữ liệu MNIST
def load_mnist():
    X = np.load("data/mnist/X.npy")
    y = np.load("data/mnist/y.npy")
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
    st.title("📌 Chia dữ liệu Train/Test")

    # Đọc dữ liệu
    X, y = load_mnist() 
    total_samples = X.shape[0]

    # Nếu chưa có cờ "data_split_done", đặt mặc định là False
    if "data_split_done" not in st.session_state:
        st.session_state.data_split_done = False  

    # Thanh kéo chọn số lượng ảnh để train
    num_samples = st.number_input("📌 Nhập số lượng ảnh để train:", min_value=1000, max_value=70000, value=20000, step=1000)
    
    # Thanh kéo chọn tỷ lệ Train/Test
    test_size = st.slider("📌 Chọn % dữ liệu Test", 10, 50, 20)
    remaining_size = 100 - test_size
    val_size = st.slider("📌 Chọn % dữ liệu Validation (trong phần Train)", 0, 50, 15)
    st.write(f"📌 **Tỷ lệ phân chia:** Test={test_size}%, Validation={val_size}%, Train={remaining_size - val_size}%")

    if st.button("✅ Xác nhận & Lưu", key="luu"):
        st.session_state.data_split_done = True  # Đánh dấu đã chia dữ liệu
        
        if num_samples == total_samples:
            X_selected, y_selected = X, y
        else:
            X_selected, _, y_selected, _ = train_test_split(
                X, y, train_size=num_samples, stratify=y, random_state=42
            )

        # Chia train/test
        stratify_option = y_selected if len(np.unique(y_selected)) > 1 else None
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X_selected, y_selected, test_size=test_size/100, stratify=stratify_option, random_state=42
        )

        # Chia train/val
        if val_size > 0:
            stratify_option = y_train_full if len(np.unique(y_train_full)) > 1 else None
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full, y_train_full, test_size=val_size / (100 - test_size),
                stratify=stratify_option, random_state=42
            )
        else:
            X_train, y_train = X_train_full, y_train_full
            X_val, y_val = np.array([]), np.array([])  # Validation rỗng nếu val_size = 0

        # Lưu dữ liệu vào session_state
        st.session_state.total_samples = num_samples
        st.session_state["classification_X_train"] = X_train
        st.session_state["classification_X_val"] = X_val
        st.session_state["classification_X_test"] = X_test
        st.session_state["classification_y_train"] = y_train
        st.session_state["classification_y_val"] = y_val
        st.session_state["classification_y_test"] = y_test
        st.session_state.test_size = X_test.shape[0]
        st.session_state.val_size = X_val.shape[0]
        st.session_state.train_size = X_train.shape[0]

        # Hiển thị thông tin chia dữ liệu
        summary_df = pd.DataFrame({
            "Tập dữ liệu": ["Train", "Validation", "Test"],
            "Số lượng mẫu": [X_train.shape[0], X_val.shape[0], X_test.shape[0]]
        })
        st.success("✅ Dữ liệu đã được chia thành công!")
        st.table(summary_df)

    elif st.session_state.data_split_done:
        st.info("✅ Dữ liệu đã được chia. Nhấn **🔄 Chia lại dữ liệu** để thay đổi.")

def mlflow_input():
    DAGSHUB_MLFLOW_URI = "https://dagshub.com/NewbieHocIT/MocMayvsPython.mlflow"
    st.session_state['mlflow_url'] = DAGSHUB_MLFLOW_URI
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
    os.environ["MLFLOW_TRACKING_USERNAME"] = "NewbieHocIT"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "681dda9a41f9271a144aa94fa8624153a3c95696"
    mlflow.set_experiment("Classification")

def train():
    mlflow_input()

    # Kiểm tra xem dữ liệu đã được chia chưa
    required_keys = [
        "classification_X_train", "classification_X_val", "classification_X_test",
        "classification_y_train", "classification_y_val", "classification_y_test",
        "test_size", "val_size", "train_size", "total_samples"
    ]
    if not all(key in st.session_state for key in required_keys):
        st.error("⚠️ Chưa có dữ liệu! Hãy chia dữ liệu trong tab 'Huấn luyện' trước khi tiếp tục.")
        return

    # Lấy dữ liệu từ session_state
    X_train = st.session_state["classification_X_train"]
    X_val = st.session_state["classification_X_val"]
    X_test = st.session_state["classification_X_test"]
    y_train = st.session_state["classification_y_train"]
    y_val = st.session_state["classification_y_val"]
    y_test = st.session_state["classification_y_test"]

    # Chuyển đổi dữ liệu
    X_train = X_train.reshape(-1, 28 * 28) / 255.0
    X_test = X_test.reshape(-1, 28 * 28) / 255.0

    st.header("⚙️ Chọn mô hình & Huấn luyện")

    model_choice = st.selectbox(
        "Chọn mô hình:", 
        ["Decision Tree", "SVM"], 
        key="classification_model_choice_selectbox"
    )

    if model_choice == "Decision Tree":
        st.markdown("""
        - **🌳 Decision Tree (Cây quyết định)** giúp chia dữ liệu thành các nhóm bằng cách đặt câu hỏi nhị phân dựa trên đặc trưng.
        - **Tham số cần chọn:**  
            - **max_depth**: Giới hạn độ sâu tối đa của cây.  
        """)
        criterion = st.selectbox(
            "Chọn tiêu chuẩn phân nhánh (criterion):", 
            ["gini", "entropy"], 
            key="classification_criterion_selectbox"
        )
        max_depth = st.slider(
            "max_depth", 
            1, 20, 5, 
            key="classification_max_depth_slider"
        )
        model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)

    elif model_choice == "SVM":
        st.markdown("""
        - **🛠️ SVM (Support Vector Machine)** là mô hình tìm siêu phẳng tốt nhất để phân tách dữ liệu.
        """)
        C = st.slider(
            "C (Regularization)", 
            0.1, 10.0, 1.0, 
            key="classification_C_slider"
        )
        kernel = st.selectbox(
            "Kernel", 
            ["linear", "sigmoid"], 
            key="classification_kernel_selectbox"
        )
        model = SVC(C=C, kernel=kernel, probability=True)

    n_folds = st.slider(
        "Chọn số folds (KFold Cross-Validation):", 
        min_value=2, max_value=10, value=5, 
        key="classification_n_folds_slider"
    )
    
    run_name = st.text_input(
        "🔹 Nhập tên Run:", 
        "Default_Run", 
        key="classification_run_name_input"
    )
    st.session_state["run_name"] = run_name if run_name else "default_run"
    
    if st.button("Huấn luyện mô hình", key="classification_train_button"):
        with mlflow.start_run(run_name=f"Train_{st.session_state['run_name']}"):
            mlflow.log_param("test_size", st.session_state.test_size)
            mlflow.log_param("val_size", st.session_state.val_size)
            mlflow.log_param("train_size", st.session_state.train_size)
            mlflow.log_param("num_samples", st.session_state.total_samples)

            progress_bar = st.progress(0)  # Thanh tiến trình
            status_text = st.empty()  # Hiển thị trạng thái từng bước

            # Giai đoạn 1: Cross-Validation (0% -> 40%)
            status_text.text("⏳ Đang chạy Cross Validation...")
            cv_scores = cross_val_score(model, X_train, y_train, cv=n_folds)
            for i in range(0, 41, 2):  # Tăng dần từ 0% đến 40%
                progress_bar.progress(i)
                time.sleep(0.05)
            mean_cv_score = cv_scores.mean()
            std_cv_score = cv_scores.std()
            status_text.text(f"✅ Cross-Validation hoàn tất! 📊 Độ chính xác trung bình: {mean_cv_score:.4f}")
            st.info(f"📊 **Cross-Validation Accuracy**: {mean_cv_score:.4f} ± {std_cv_score:.4f}")

            # Giai đoạn 2: Huấn luyện mô hình (40% -> 70%)
            status_text.text("🛠️ Đang huấn luyện mô hình...")
            for i in range(40, 71, 2):  # Tăng từ 40% đến 70%
                progress_bar.progress(i)
                time.sleep(0.05)
            model.fit(X_train, y_train)
            status_text.text("✅ Huấn luyện hoàn tất!")

            # Giai đoạn 3: Đánh giá trên test set (70% -> 85%)
            status_text.text("📊 Đang đánh giá mô hình trên test set...")
            for i in range(70, 86, 2):  # Tăng từ 70% đến 85%
                progress_bar.progress(i)
                time.sleep(0.05)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.success(f"✅ **Độ chính xác trên test set**: {acc:.4f}")

            # Giai đoạn 4: Logging với MLflow (85% -> 100%)
            status_text.text("📝 Đang ghi log vào MLflow...")
            for i in range(85, 101, 2):  # Tăng từ 85% đến 100%
                progress_bar.progress(i)
                time.sleep(0.05)

            # Logging thông tin
            mlflow.log_param("model", model_choice)
            if model_choice == "Decision Tree":
                mlflow.log_param("criterion", criterion)
                mlflow.log_param("max_depth", max_depth)
            elif model_choice == "SVM":
                mlflow.log_param("C", C)
                mlflow.log_param("kernel", kernel)

            mlflow.log_metric("test_accuracy", acc)
            mlflow.log_metric("cv_accuracy_mean", mean_cv_score)
            mlflow.log_metric("cv_accuracy_std", std_cv_score)
            mlflow.sklearn.log_model(model, model_choice.lower())

            # Lưu mô hình vào session_state
            if "classification_models" not in st.session_state:
                st.session_state["classification_models"] = []

            model_name = model_choice.lower().replace(" ", "_")
            if model_choice == "SVM":
                model_name += f"_{kernel}"

            existing_model = next((item for item in st.session_state["classification_models"] if item["name"] == model_name), None)
            if existing_model:
                count = 1
                new_model_name = f"{model_name}_{count}"
                while any(item["name"] == new_model_name for item in st.session_state["classification_models"]):
                    count += 1
                    new_model_name = f"{model_name}_{count}"
                model_name = new_model_name
                st.warning(f"⚠️ Mô hình được lưu với tên: {model_name}")

            st.session_state["classification_models"].append({"name": model_name, "model": model})
            st.write(f"🔹 Mô hình đã được lưu với tên: {model_name}")
            st.write(f"Tổng số mô hình hiện tại: {len(st.session_state['classification_models'])}")

            st.write("📋 Danh sách các mô hình đã lưu:")
            model_names = [model["name"] for model in st.session_state["classification_models"]]
            st.write(", ".join(model_names))

            # Hoàn tất
            status_text.text("✅ Huấn luyện và logging hoàn tất!")
            st.success(f"✅ Đã log dữ liệu cho **Train_{st.session_state['run_name']}**!")

def show_experiment_selector():
    st.title("📊 MLflow Experiments")

    experiment_name = "Classification"
    
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

def preprocess_image(image):
    """Xử lý ảnh đầu vào: Chuyển về grayscale, resize, chuẩn hóa"""
    image = image.convert("L")
    image = image.resize((28, 28))  # Resize về kích thước phù hợp
    img_array = np.array(image) / 255.0  # Chuẩn hóa pixel về [0,1]
    return img_array.reshape(1, -1)  # Chuyển thành vector 1D

def du_doan():
    st.title("🔢 Dự đoán chữ số viết tay")

    # Kiểm tra xem đã có mô hình chưa
    if "classification_models" not in st.session_state or not st.session_state["classification_models"]:
        st.error("⚠️ Chưa có mô hình nào được huấn luyện. Hãy huấn luyện mô hình trước.")
        return

    # Chọn mô hình
    model_names = [model["name"] for model in st.session_state["classification_models"]]
    selected_model_name = st.selectbox("🔍 Chọn mô hình đã huấn luyện:", model_names)
    selected_model = next(model["model"] for model in st.session_state["classification_models"] if model["name"] == selected_model_name)

    # Chọn cách nhập ảnh: Tải lên hoặc Vẽ
    option = st.radio("📌 Chọn cách nhập ảnh:", ["🖼️ Tải ảnh lên", "✍️ Vẽ số"], key="input_option_radio")

    img_array = None  # Khởi tạo ảnh đầu vào

    # 1️⃣ 🖼️ Nếu tải ảnh lên
    if option == "🖼️ Tải ảnh lên":
        uploaded_file = st.file_uploader("📤 Tải ảnh chữ số viết tay (28x28 pixel)", type=["png", "jpg", "jpeg"], key="upfile")
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Ảnh đã tải lên", use_container_width=True)
            img_array = preprocess_image(image)  # Xử lý ảnh

    # 2️⃣ ✍️ Nếu vẽ số
    else:
        st.write("🎨 Vẽ số trong khung dưới đây:")
        
        canvas_result = st_canvas(
            fill_color="black",  # Màu nền
            stroke_width=10,
            stroke_color="white",
            background_color="black",
            height=250,
            width=250,
            drawing_mode="freedraw",
            key="canvas_draw"
        )

        # Khi người dùng bấm "Dự đoán"
        if st.button("Dự đoán số", key="dudoan"):
            if canvas_result.image_data is not None:
                # Chuyển đổi ảnh từ canvas thành định dạng PIL
                image = Image.fromarray((canvas_result.image_data[:, :, :3]).astype(np.uint8))
                img_array = preprocess_image(image)  # Xử lý ảnh
            else:
                st.error("⚠️ Hãy vẽ một số trước khi dự đoán!")

    # 🔍 Dự đoán nếu có ảnh đầu vào hợp lệ
    if img_array is not None:
        prediction = selected_model.predict(img_array)[0]
        probabilities = selected_model.predict_proba(img_array)[0]  # Lấy toàn bộ xác suất của các lớp

        # 🏆 Hiển thị kết quả dự đoán
        st.success(f"🔢 Dự đoán: **{prediction}**")

        # 📊 Hiển thị toàn bộ độ tin cậy theo từng lớp
        st.write("### 🔢 Độ tin cậy :")
        st.bar_chart(probabilities)

def Classification():
    st.title("🖊️ MNIST Classification App")
    tab1, tab2, tab3, tab4 = st.tabs(["📘 Data", "⚙️ Huấn luyện", "🔢 Dự đoán", "🔥 Mlflow"])

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
