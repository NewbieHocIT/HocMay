import streamlit as st
import os
import pandas as pd
import shutil
from src.Classification import Classification
from src.clustering import Clusterting
import mlflow
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA

# Cache dữ liệu MNIST
@st.cache_data
def load_mnist():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data, mnist.target.astype(int)
    return X, y

# Cache danh sách experiments từ MLflow
@st.cache_data
def get_experiments():
    return mlflow.search_experiments()

# Cache danh sách runs từ MLflow
@st.cache_data
def get_runs(experiment_id):
    return mlflow.search_runs(experiment_id)


# Hàm quản lý tab MLFlow
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
                st.write("#### Danh sách runs")
                st.dataframe(runs)

                # Thêm phần chọn mô hình so sánh
                selected_runs = st.multiselect(
                    "🔍 Chọn các runs để so sánh",
                    options=runs["run_id"],
                    key="mlflow_select_runs_for_comparison"
                )

                if selected_runs:
                    st.write("#### So sánh các mô hình")
                    comparison_data = []

                    for run_id in selected_runs:
                        run = mlflow.get_run(run_id)
                        comparison_data.append({
                            "Run ID": run.info.run_id,
                            "Experiment ID": run.info.experiment_id,
                            "Start Time": run.info.start_time,
                            **run.data.metrics,
                            **run.data.params
                        })

                    st.dataframe(pd.DataFrame(comparison_data))

                selected_run_id = st.selectbox(
                    "🔍 Chọn run để xem chi tiết",
                    options=runs["run_id"],
                    key="mlflow_select_run"
                )

                run = mlflow.get_run(selected_run_id)
                st.write("##### Thông tin run")
                st.write(f"*Run ID:* {run.info.run_id}")
                st.write(f"*Experiment ID:* {run.info.experiment_id}")
                st.write(f"*Start Time:* {run.info.start_time}")

                st.write("##### Metrics")
                st.json(run.data.metrics)

                st.write("##### Params")
                st.json(run.data.params)

                artifacts = mlflow.artifacts.list_artifacts(run.info.run_id)
                if artifacts:
                    st.write("##### Artifacts")
                    artifact_paths = [artifact.path for artifact in artifacts]
                    st.write(artifact_paths)

                    st.write("#### Đổi tên artifacts")
                    for artifact in artifacts:
                        new_name = st.text_input(f"Đổi tên cho file: {artifact.path}", artifact.path)
                        if new_name != artifact.path:
                            try:
                                local_path = mlflow.artifacts.download_artifacts(run.info.run_id, artifact.path)
                                new_local_path = os.path.join(os.path.dirname(local_path), new_name)
                                shutil.move(local_path, new_local_path)
                                mlflow.log_artifact(new_local_path)
                                os.remove(local_path)
                                st.success(f"✅ Đã đổi tên thành công: {artifact.path} → {new_name}")
                            except Exception as e:
                                st.error(f"❌ Đã xảy ra lỗi khi đổi tên: {e}")
                else:
                    st.write("Không có artifacts nào.")
            else:
                st.warning("Không có runs nào trong thí nghiệm này.")
        else:
            st.warning("Không có thí nghiệm nào được tìm thấy.")
    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi lấy danh sách thí nghiệm: {e}")

# Tạo các tab
st.title("🚀 Streamlit and MLFlow")
tab2, tab3, tab4 = st.tabs(
    ('Classification MNIST', 'Clustering Algorithms', 'MLFlow-Web')
)

# Hiển thị nội dung của từng tab
with tab2:
    Classification()

with tab3:
    Clusterting()

with tab4:
    mlflow_tab()