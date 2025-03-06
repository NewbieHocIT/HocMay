import streamlit as st
import os
import pandas as pd
from src import Classification
from src import clustering
import mlflow
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from src import PCA_tSNE
from src.linear_regression import LinearApp
# Cache dữ liệu MNIST

# Cache danh sách experiments từ MLflow
def get_experiments():
    return mlflow.search_experiments()

# Cache danh sách runs từ MLflow
def get_runs(experiment_id):
    return mlflow.search_runs(experiment_id)


# Hàm quản lý tab MLFlow
from concurrent.futures import ThreadPoolExecutor

# Các hàm với caching
def get_experiments():
    return mlflow.search_experiments()

def get_runs(experiment_id):
    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    return runs

def get_run_details(run_id):
    return mlflow.get_run(run_id)

def list_artifacts(run_id):
    return mlflow.artifacts.list_artifacts(run_id)

def fetch_runs_parallel(run_ids):
    with ThreadPoolExecutor() as executor:
        runs = list(executor.map(get_run_details, run_ids))
    return runs

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import streamlit as st
import os
import shutil

# Hàm lấy danh sách thí nghiệm
def get_experiments():
    return mlflow.search_experiments()

# Hàm lấy danh sách runs trong thí nghiệm
def get_runs(experiment_id):
    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    return runs

# Hàm lấy thông tin chi tiết của một run
def get_run_details(run_id):
    return mlflow.get_run(run_id)

# Hàm lấy danh sách artifacts của một run
def list_artifacts(run_id):
    return mlflow.artifacts.list_artifacts(run_id)

# Hàm xóa thí nghiệm
def delete_experiment(experiment_id):
    client = MlflowClient()
    try:
        client.delete_experiment(experiment_id)
        st.success(f"✅ Đã xóa thí nghiệm: {experiment_id}")
        st.rerun()  # Làm mới trang để cập nhật danh sách thí nghiệm
    except Exception as e:
        st.error(f"❌ Lỗi khi xóa thí nghiệm: {e}")

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import streamlit as st
import os
import shutil

# Các hàm với caching
def get_experiments():
    return mlflow.search_experiments()

def get_runs(experiment_id):
    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    return runs

def get_run_details(run_id):
    return mlflow.get_run(run_id)

def list_artifacts(run_id):
    return mlflow.artifacts.list_artifacts(run_id)

def fetch_runs_parallel(run_ids):
    with ThreadPoolExecutor() as executor:
        runs = list(executor.map(get_run_details, run_ids))
    return runs

import os
import shutil
import pandas as pd
import streamlit as st
import mlflow
from mlflow.tracking import MlflowClient

def mlflow_tab():
    st.title("🚀 MLflow Model Logging & Registry")
    
    DAGSHUB_USERNAME = "NewbieHocIT"
    DAGSHUB_REPO_NAME = "MocMayvsPython"
    DAGSHUB_TOKEN = "681dda9a41f9271a144aa94fa8624153a3c95696"

    mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow")
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

    try:
        experiments = get_experiments()
        if experiments:
            st.write("#### Danh sách thí nghiệm")
            experiment_data = [{
                "Experiment ID": exp.experiment_id,
                "Experiment Name": exp.name,
                "Artifact Location": exp.artifact_location
            } for exp in experiments]
            st.dataframe(pd.DataFrame(experiment_data))

            selected_exp_id = st.selectbox(
                "🔍 Chọn thí nghiệm để xem chi tiết",
                options=[exp.experiment_id for exp in experiments],
                key="mlflow_select_experiment"
            )

            runs = get_runs(selected_exp_id)
            if not runs.empty:
                runs["Run Name"] = runs["tags.mlflow.runName"]  
                runs["start_time"] = pd.to_datetime(runs["start_time"], unit="ms").dt.strftime("%Y-%m-%d %H:%M:%S")

                st.write("#### Danh sách runs")
                st.dataframe(runs[["Run Name", "run_id", "status", "start_time"]])

                # 🎯 Chọn Run theo tên
                run_name_to_id = {row["Run Name"]: row["run_id"] for _, row in runs.iterrows()}

                selected_run_name = st.selectbox(
                    "🔍 Chọn Run theo tên",
                    options=run_name_to_id.keys(),
                    key="mlflow_select_run_by_name"
                )
                selected_run_id = run_name_to_id[selected_run_name]

                run = get_run_details(selected_run_id)
                formatted_time = pd.to_datetime(run.info.start_time, unit="ms").strftime("%Y-%m-%d %H:%M:%S")

                st.write("##### Thông tin run")
                st.write(f"*Run Name:* {selected_run_name}")
                st.write(f"*Run ID:* {run.info.run_id}")
                st.write(f"*Experiment ID:* {run.info.experiment_id}")
                st.write(f"*Start Time:* {formatted_time}")

                # ✏️ Đổi tên Run
                new_run_name = st.text_input("✏️ Nhập tên mới cho Run", value=selected_run_name)
                if st.button("🔄 Cập nhật tên Run"):
                    try:
                        client = MlflowClient()
                        client.set_tag(selected_run_id, "mlflow.runName", new_run_name)
                        st.success(f"✅ Đã đổi tên Run thành: {new_run_name}")
                        st.rerun()  # Làm mới trang để cập nhật tên mới
                    except Exception as e:
                        st.error(f"❌ Lỗi khi đổi tên Run: {e}")

                st.write("##### Metrics")
                st.json(run.data.metrics)

                st.write("##### Params")
                st.json(run.data.params)

                artifacts = list_artifacts(run.info.run_id)
                if artifacts:
                    st.write("##### Artifacts")
                    artifact_paths = [artifact.path for artifact in artifacts]
                    st.write(artifact_paths)
                else:
                    st.write("Không có artifacts nào.")

                # 🗑️ Xóa Run
                st.write("#### Xóa Run")
                selected_run_for_delete = st.selectbox(
                    "🗑️ Chọn Run để xóa",
                    options=run_name_to_id.keys(),
                    key="mlflow_select_run_for_delete"
                )
                selected_run_id_for_delete = run_name_to_id[selected_run_for_delete]

                if st.button("❌ Xóa Run"):
                    try:
                        client = MlflowClient()
                        client.delete_run(selected_run_id_for_delete)
                        st.success(f"✅ Đã xóa Run: {selected_run_for_delete}")
                        st.rerun()  # Cập nhật danh sách
                    except Exception as e:
                        st.error(f"❌ Lỗi khi xóa Run: {e}")

                # 📊 So sánh Run
                st.write("#### So sánh các mô hình")
                selected_runs = st.multiselect(
                    "🔍 Chọn các Run Name để so sánh",
                    options=run_name_to_id.keys(),
                    key="mlflow_select_runs_for_comparison"
                )

                if selected_runs:
                    selected_run_ids = [run_name_to_id[name] for name in selected_runs]
                    comparison_data = fetch_runs_parallel(selected_run_ids)
                    comparison_df = pd.DataFrame([{
                        "Run Name": run.info.run_name,
                        "Run ID": run.info.run_id,
                        "Experiment ID": run.info.experiment_id,
                        "Start Time": pd.to_datetime(run.info.start_time, unit="ms").strftime("%Y-%m-%d %H:%M:%S"),
                        **run.data.metrics,
                        **run.data.params
                    } for run in comparison_data])

                    st.dataframe(comparison_df)
            else:
                st.warning("Không có runs nào trong thí nghiệm này.")
        else:
            st.warning("Không có thí nghiệm nào được tìm thấy.")
    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi lấy danh sách thí nghiệm: {e}")


# Gọi hàm để chạy ứng dụng
import streamlit as st


# Tạo các lựa chọn hiển thị trực tiếp trong thanh bên trái
with st.sidebar:
    st.write("### Chọn chức năng")
    
    # Sử dụng st.button để tạo các nút bấm
    if st.button("📊 Classification MNIST"):
        st.session_state.current_page = "Classification MNIST"
    if st.button("LinearRegression"):
        st.session_state.current_page = "LinearRegression"
    if st.button("🔍 Clustering Algorithms"):
        st.session_state.current_page = "Clustering Algorithms"
    if st.button("🚀 MLFlow-Web"):
        st.session_state.current_page = "MLFlow-Web"
    if st.button("PCA, t-SNE"):
        st.session_state.current_page = "PCA, t-SNE"

# Khởi tạo session state nếu chưa có
if "current_page" not in st.session_state:
    st.session_state.current_page = "Classification MNIST"
if st.session_state.current_page =="LinearRegression":
    LinearApp()
# Hiển thị nội dung tương ứng với lựa chọn
if st.session_state.current_page == "Classification MNIST":
    Classification.Classification()
elif st.session_state.current_page == "Clustering Algorithms":
    clustering.Clustering()
elif st.session_state.current_page == "MLFlow-Web":
    mlflow_tab()
elif st.session_state.current_page == "PCA, t-SNE":
    PCA_tSNE.run_pca_tsne()
