import streamlit as st
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import pytz
import os
import matplotlib.pyplot as plt
import dagshub

def list_logged_models(experiment_id):
    client = MlflowClient()
    runs = client.search_runs(experiment_ids=[experiment_id])
    gmt7 = pytz.timezone("Asia/Bangkok")
    df = pd.DataFrame([{ 
        "Run ID": r.info.run_id, 
        "Run Name": r.data.tags.get("mlflow.runName", "N/A"), 
        "Model Type": r.data.tags.get("model_type", "N/A"),  
        "Start Time": pd.to_datetime(r.info.start_time, unit='ms')
                        .tz_localize('UTC')
                        .tz_convert(gmt7)
                        .strftime('%Y-%m-%d %H:%M:%S'), 
        "Status": "✅ Hoàn thành" if r.info.status == "FINISHED" else "❌ Lỗi"
    } for r in runs])
    return df

def display():
    try:
        dagshub.init(repo_owner='NewbieHocIT', repo_name='MocMayvsPython', mlflow=True)
        os.environ['MLFLOW_TRACKING_USERNAME'] = 'NewbieHocIT'
        os.environ['MLFLOW_TRACKING_PASSWORD'] = '681dda9a41f9271a144aa94fa8624153a3c95696'
        mlflow.set_tracking_uri("https://dagshub.com/NewbieHocIT/MocMayvsPython.mlflow")
        client = MlflowClient()
        experiments = client.search_experiments()
        print('Kết nối MLflow thành công!')
    except Exception as e:
        st.warning("Không thể kết nối với MLflow hoặc DagsHub. Vui lòng kiểm tra cài đặt.")
        experiments = []

    st.title("🚀 MLflow Model Logging & Registry")

    if experiments:
        experiment_names = [exp.name for exp in experiments]
        selected_experiment = st.selectbox("📊 Chọn thí nghiệm", experiment_names)
        experiment_id = next(exp.experiment_id for exp in experiments if exp.name == selected_experiment)
        
        st.subheader("📌 Các mô hình đã log")
        models_df = list_logged_models(experiment_id)
        st.dataframe(models_df, use_container_width=True)
        
        st.subheader("📈 So sánh các mô hình")
        available_run_names = models_df["Run Name"].tolist()
        selected_run_names = st.multiselect("🔍 Chọn Run Name để so sánh", available_run_names)

        if selected_run_names:
            comparison_data = []
            for run_name in selected_run_names:
                run_info = models_df[models_df["Run Name"] == run_name].iloc[0]
                run_id = run_info["Run ID"]
                run = client.get_run(run_id)
                model_type = run.data.tags.get("model_type", "N/A")
                metrics = {"Run ID": run_id, "Run Name": run_name, "Model Type": model_type}
                metrics.update(run.data.metrics)
                comparison_data.append(metrics)

            comparison_df = pd.DataFrame(comparison_data)
            st.write("Dữ liệu so sánh:")
            st.dataframe(comparison_df, use_container_width=True)

            available_metrics = [col for col in comparison_df.columns if col not in ["Run ID", "Run Name", "Model Type"]]
            st.write("Các metric hợp lệ:", available_metrics)
            
            if available_metrics:
                selected_metric = st.selectbox("📌 Chọn metric để vẽ biểu đồ", available_metrics)
                
                if selected_metric:
                    comparison_df[selected_metric] = pd.to_numeric(comparison_df[selected_metric], errors='coerce')
                    valid_runs = comparison_df.dropna(subset=[selected_metric])
                    if not valid_runs.empty:
                        fig, ax = plt.subplots()
                        ax.bar(valid_runs["Run Name"], valid_runs[selected_metric], color='skyblue')
                        ax.set_xlabel("Run Name")
                        ax.set_ylabel(selected_metric)
                        ax.set_title(f"So sánh {selected_metric}")
                        ax.tick_params(axis='x', rotation=45)
                        st.pyplot(fig)
                    else:
                        st.warning(f"Không có dữ liệu {selected_metric} hợp lệ để vẽ biểu đồ.")
            else:
                st.warning("Không có metric nào để so sánh.")
    else:
        st.warning("Không tìm thấy thí nghiệm nào.")

if __name__ == "__main__":
    display()
