import tensorflow as tf
import tensorflow as tf
Sequential = tf.keras.models.Sequential
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
Adam = tf.keras.optimizers.Adam


import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image
import pandas as pd
import os
import mlflow
from datetime import datetime
from sklearn.model_selection import cross_val_score
from streamlit_drawable_canvas import st_canvas


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
        st.session_state["neural_X_train"] = X_train
        st.session_state["neural_X_val"] = X_val
        st.session_state["neural_X_test"] = X_test
        st.session_state["neural_y_train"] = y_train
        st.session_state["neural_y_val"] = y_val
        st.session_state["neural_y_test"] = y_test
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

    mlflow.set_experiment("neural")
from tensorflow.python.keras.engine import data_adapter

def _is_distributed_dataset(ds):
    return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)

data_adapter._is_distributed_dataset = _is_distributed_dataset

from tensorflow.python.keras.callbacks import Callback

# Callback tùy chỉnh để cập nhật thanh tiến trình cho huấn luyện
class ProgressBarCallback(Callback):
    def __init__(self, total_epochs, progress_bar, status_text, max_train_progress=80):
        super(ProgressBarCallback, self).__init__()
        self.total_epochs = total_epochs
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.max_train_progress = max_train_progress  # Giới hạn tiến trình huấn luyện (80%)

    def on_epoch_begin(self, epoch, logs=None):
        progress = (epoch + 1) / self.total_epochs * self.max_train_progress
        self.progress_bar.progress(min(int(progress), self.max_train_progress))
        self.status_text.text(f"🛠️ Đang huấn luyện mô hình... Epoch {epoch + 1}/{self.total_epochs}")

    def on_train_end(self, logs=None):
        self.progress_bar.progress(self.max_train_progress)
        self.status_text.text("✅ Huấn luyện mô hình hoàn tất, đang chuẩn bị logging...")

def train():
    mlflow_input()

    # Kiểm tra xem dữ liệu đã được chia chưa
    if (
        "neural_X_train" not in st.session_state
        or "neural_X_val" not in st.session_state
        or "neural_X_test" not in st.session_state
    ):
        st.error("⚠️ Chưa có dữ liệu! Hãy chia dữ liệu trước.")
        return

    # Lấy dữ liệu từ session_state
    X_train = st.session_state["neural_X_train"]
    X_val = st.session_state["neural_X_val"]
    X_test = st.session_state["neural_X_test"]
    y_train = st.session_state["neural_y_train"]
    y_val = st.session_state["neural_y_val"]
    y_test = st.session_state["neural_y_test"]

    # Chuyển đổi dữ liệu
    X_train = X_train.reshape(-1, 28 * 28) / 255.0
    X_test = X_test.reshape(-1, 28 * 28) / 255.0
    if X_val.size > 0:
        X_val = X_val.reshape(-1, 28 * 28) / 255.0

    st.header("⚙️ Chọn mô hình & Huấn luyện")

    model_choice = st.selectbox(
        "Chọn mô hình:", 
        ["Neural Network"], 
        key="neural_model_choice_selectbox"
    )

    if model_choice == "Neural Network":
        st.markdown("""
        - **🧠 Neural Network (Mạng nơ-ron)** là một mô hình học sâu có khả năng học các đặc trưng phức tạp từ dữ liệu.
        - **Tham số cần chọn:**  
            - **Số lớp ẩn**: Số lượng lớp ẩn trong mạng.  
            - **Số node mỗi lớp**: Số lượng node trong mỗi lớp ẩn.  
            - **Hàm kích hoạt**: Hàm kích hoạt cho các lớp ẩn.  
            - **Tốc độ học**: Tốc độ học của thuật toán tối ưu.  
        """)
        
        num_layers = st.slider("Số lớp ẩn", 1, 5, 2, key="neural_num_layers_slider")
        num_nodes = st.slider("Số node mỗi lớp", 32, 256, 128, key="neural_num_nodes_slider")
        activation = st.selectbox("Hàm kích hoạt", ["relu", "sigmoid", "tanh"], key="neural_activation_selectbox")
        epochs = st.slider("Số epoch", 1, 50, 10, key="neural_epochs_slider")

        # Xây dựng mô hình
        model = Sequential()
        model.add(Dense(num_nodes, input_shape=(28 * 28,), activation=activation))
        for _ in range(num_layers - 1):
            model.add(Dense(num_nodes, activation=activation))
        model.add(Dense(10, activation='softmax'))
        
        # Biên dịch mô hình
        model.compile(optimizer=Adam(learning_rate=0.01),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
    
    run_name = st.text_input("🔹 Nhập tên Run:", "Default_Run", key="neural_run_name_input")
    st.session_state["run_name"] = run_name if run_name else "default_run"
    
    if st.button("Huấn luyện mô hình", key="neural_train_button"):
        with mlflow.start_run(run_name=f"Train_{st.session_state['run_name']}"):
            mlflow.log_param("test_size", st.session_state.test_size)
            mlflow.log_param("val_size", st.session_state.val_size)
            mlflow.log_param("train_size", st.session_state.train_size)
            mlflow.log_param("num_samples", st.session_state.total_samples)

            progress_bar = st.progress(0)  # Thanh tiến trình
            status_text = st.empty()  # Hiển thị trạng thái từng bước

            # Giai đoạn 1: Huấn luyện (chiếm 80% tiến trình)
            progress_callback = ProgressBarCallback(epochs, progress_bar, status_text, max_train_progress=80)

            if X_val.size > 0:
                history = model.fit(
                    X_train, y_train, 
                    epochs=epochs, 
                    validation_data=(X_val, y_val),
                    callbacks=[progress_callback],
                    verbose=0
                )
            else:
                history = model.fit(
                    X_train, y_train, 
                    epochs=epochs, 
                    callbacks=[progress_callback],
                    verbose=0
                )

            # Giai đoạn 2: Logging và đánh giá (chiếm 20% còn lại)
            status_text.text("📊 Đang đánh giá mô hình trên test set...")
            progress_bar.progress(85)  # Cập nhật tiến trình sau khi đánh giá test set
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

            # Hiển thị kết quả
            if X_val.size > 0:
                train_accuracy = history.history['accuracy'][-1]
                val_accuracy = history.history['val_accuracy'][-1]
                st.success(f"✅ **Độ chính xác trên tập train**: {train_accuracy:.4f}")
                st.success(f"✅ **Độ chính xác trên tập validation**: {val_accuracy:.4f}")
            else:
                train_accuracy = history.history['accuracy'][-1]
                st.success(f"✅ **Độ chính xác trên tập train**: {train_accuracy:.4f}")
            st.success(f"✅ **Độ chính xác trên test set**: {test_acc:.4f}")

            # Logging với MLflow
            status_text.text("📝 Đang ghi log vào MLflow...")
            progress_bar.progress(90)  # Cập nhật tiến trình khi bắt đầu logging

            mlflow.log_param("model", model_choice)
            mlflow.log_param("num_layers", num_layers)
            mlflow.log_param("num_nodes", num_nodes)
            mlflow.log_param("activation", activation)
            mlflow.log_param("learning_rate", 0.01)
            mlflow.log_param("epochs", epochs)

            mlflow.log_metric("test_accuracy", test_acc)
            if X_val.size > 0:
                mlflow.log_metric("train_accuracy", train_accuracy)
                mlflow.log_metric("val_accuracy", val_accuracy)

            # Lưu mô hình
            model_path = f"model_{st.session_state['run_name']}.h5"
            model.save(model_path)
            mlflow.log_artifact(model_path)
            progress_bar.progress(95)  # Cập nhật tiến trình khi lưu mô hình

            # Lưu thông tin mô hình vào session_state
            if "neural_models" not in st.session_state:
                st.session_state["neural_models"] = []

            model_name = model_choice.lower().replace(" ", "_")
            model_name += f"_{num_layers}layers_{num_nodes}nodes_{activation}"
            existing_model = next((item for item in st.session_state["neural_models"] if item["name"] == model_name), None)

            if existing_model:
                count = 1
                new_model_name = f"{model_name}_{count}"
                while any(item["name"] == new_model_name for item in st.session_state["neural_models"]):
                    count += 1
                    new_model_name = f"{model_name}_{count}"
                model_name = new_model_name
                st.warning(f"⚠️ Mô hình được lưu với tên: {model_name}")

            st.session_state["neural_models"].append({"name": model_name, "model": model})
            st.write(f"🔹 Mô hình đã được lưu với tên: {model_name}")
            st.write(f"Tổng số mô hình hiện tại: {len(st.session_state['neural_models'])}")

            # Hiển thị danh sách mô hình
            st.write("📋 Danh sách các mô hình đã lưu:")
            model_names = [model["name"] for model in st.session_state["neural_models"]]
            st.write(", ".join(model_names))

            # Hoàn tất tiến trình
            progress_bar.progress(100)
            status_text.text("✅ Huấn luyện và logging hoàn tất!")
            st.success(f"✅ Đã log dữ liệu cho **Train_{st.session_state['run_name']}**!")
            
def preprocess_image(image):
    """Xử lý ảnh đầu vào: Chuyển về grayscale, resize, chuẩn hóa"""
    image = image.convert("L")
    image = image.resize((28, 28))  # Resize về kích thước phù hợp
    img_array = np.array(image) / 255.0  # Chuẩn hóa pixel về [0,1]
    return img_array.reshape(1, -1)  # Chuyển thành vector 1D

def du_doan():
    st.title("🔢 Dự đoán chữ số viết tay")

    # Kiểm tra xem đã có mô hình chưa
    if "neural_models" not in st.session_state or not st.session_state["neural_models"]:
        st.error("⚠️ Chưa có mô hình nào được huấn luyện. Hãy huấn luyện mô hình trước.")
        return

    # Chọn mô hình
    model_names = [model["name"] for model in st.session_state["neural_models"]]
    selected_model_name = st.selectbox("🔍 Chọn mô hình đã huấn luyện:", model_names)
    selected_model = next(model["model"] for model in st.session_state["neural_models"] if model["name"] == selected_model_name)

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
        
        # Canvas để vẽ
        canvas_result = st_canvas(
            fill_color="black",  # Màu nền
            stroke_width=10,
            stroke_color="black",
            background_color="white",
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
        prediction = np.argmax(selected_model.predict(img_array), axis=1)[0]
        probabilities = selected_model.predict(img_array)[0]  # Lấy toàn bộ xác suất của các lớp

        # 🏆 Hiển thị kết quả dự đoán
        st.success(f"🔢 Dự đoán: **{prediction}**")

        # 📊 Hiển thị toàn bộ độ tin cậy theo từng lớp
        st.write("### 🔢 Độ tin cậy :")

        # 📊 Vẽ biểu đồ độ tin cậy
        st.bar_chart(probabilities)

def show_experiment_selector():
    st.title("📊 MLflow Experiments")

    experiment_name = "neural"
    
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
        
def Neural():
    st.title("🖊️ MNIST Neural Network App")
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
    Neural()