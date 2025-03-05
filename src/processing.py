# data_processing.py
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def display():
    if "model" not in st.session_state:
        st.session_state.model = None
    st.subheader("📂 Tải dữ liệu lên")
    uploaded_file = st.file_uploader("Chọn file CSV", type=["csv"], key="file_upload")

    # Khởi tạo session_state nếu chưa có
    if "df" not in st.session_state:
        st.session_state.df = None
    if "show_drop_step" not in st.session_state:
        st.session_state.show_drop_step = False
    if "show_missing_step" not in st.session_state:
        st.session_state.show_missing_step = False
    if "show_encode_step" not in st.session_state:
        st.session_state.show_encode_step = False
    if "show_scale_step" not in st.session_state:
        st.session_state.show_scale_step = False

    # Đọc dữ liệu nếu có file được tải lên
    if uploaded_file is not None and st.session_state.df is None:
        st.session_state.df = pd.read_csv(uploaded_file)

    if st.session_state.df is None:
        st.warning("🚨 Vui lòng tải lên một file CSV.")
        st.stop()

    df = st.session_state.df.copy()
    st.write("### Dữ liệu ban đầu")
    st.write(df.head())

    # **Bước 1: Loại bỏ cột**
    st.write("### 1️⃣ Loại bỏ cột không cần thiết")
    drop_cols = st.multiselect("Chọn cột muốn loại bỏ", df.columns.tolist(), key="drop_cols")
    
    if st.button("Xóa cột đã chọn"):
        df.drop(columns=drop_cols, inplace=True)
        st.session_state.df = df.copy()
        st.session_state.show_drop_step = True  # Hiển thị bước này

    if st.session_state.show_drop_step:
        st.write("### 🔄 Dữ liệu sau khi xóa cột:")
        st.write(st.session_state.df.head())

    # **Bước 2: Xử lý dữ liệu thiếu**
    st.write("### 2️⃣ Xử lý dữ liệu thiếu")
    missing_cols = [col for col in df.columns if df[col].isna().sum() > 0]
    
    # Hiển thị thông tin các cột thiếu
    if missing_cols:
        st.write("### 🔍 Thông tin các cột thiếu:")
        missing_info = pd.DataFrame({
            "Cột": missing_cols,
            "Số lượng giá trị thiếu": [df[col].isna().sum() for col in missing_cols],
            "Kiểu dữ liệu": [str(df[col].dtype) for col in missing_cols]
        })
        st.write(missing_info)
    else:
        st.write("🎉 Không có cột nào bị thiếu dữ liệu!")

    selected_col = st.selectbox("Chọn cột có dữ liệu thiếu", [None] + missing_cols)

    if selected_col:
        dtype = str(df[selected_col].dtype)
        if dtype in ["int64", "float64"]:
            method = st.radio("Chọn phương pháp", ["Mean", "Median", "Giá trị cụ thể"])
        else:
            method = st.radio("Chọn phương pháp", ["Mode", "Giá trị cụ thể"])

        value = None
        if method == "Giá trị cụ thể":
            value = st.text_input("Nhập giá trị thay thế")

        if st.button("Xử lý thiếu dữ liệu"):
            if method == "Mean":
                st.session_state.df[selected_col].fillna(df[selected_col].mean(), inplace=True)
            elif method == "Median":
                st.session_state.df[selected_col].fillna(df[selected_col].median(), inplace=True)
            elif method == "Mode":
                st.session_state.df[selected_col].fillna(df[selected_col].mode()[0], inplace=True)
            elif method == "Giá trị cụ thể":
                st.session_state.df[selected_col].fillna(value, inplace=True)
            st.session_state.show_missing_step = True  # Hiển thị bước này

    if st.session_state.show_missing_step:
        st.write("### 🔄 Dữ liệu sau khi xử lý thiếu:")
        st.write(st.session_state.df.head())

    # **Bước 3: Mã hóa dữ liệu**
    st.write("### 3️⃣ Thay đổi kiểu dữ liệu")
    encoding_cols = df.select_dtypes(include=['object']).columns.tolist()
    selected_col = st.selectbox("Chọn cột để Thay đổi kiểu dữ liệu", [None] + encoding_cols, key="encoding_col")

    if selected_col:
        unique_values = df[selected_col].unique()  # Lấy giá trị duy nhất
        mapping_dict = {}

        st.write(f"🔹 Nhập giá trị thay thế cho các giá trị trong cột `{selected_col}`:")
        for val in unique_values:
            new_val = st.text_input(f"{val} →", key=f"encode_{selected_col}_{val}")
            if new_val:
                mapping_dict[val] = new_val  # Lưu giá trị mới

        if st.button("Mã hóa cột"):
            if mapping_dict:
                st.session_state.df[selected_col] = st.session_state.df[selected_col].map(mapping_dict).astype(float)
                st.session_state.show_encode_step = True  # Hiển thị bước này
            else:
                st.warning("⚠️ Vui lòng nhập giá trị thay thế trước khi mã hóa.")

    if st.session_state.show_encode_step:
        st.write(f"### 🔄 Dữ liệu sau khi mã hóa: `{selected_col}`")
        st.write(st.session_state.df.head())

    # **Bước 4: Chuẩn hóa dữ liệu**
    st.write("### 4️⃣ Chuẩn hóa dữ liệu")
    if st.button("Chuẩn hóa toàn bộ dữ liệu"):
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        scaler = StandardScaler()
        st.session_state.df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        st.session_state.show_scale_step = True  # Hiển thị bước này

    if st.session_state.show_scale_step:
        st.write("### 🔄 Dữ liệu sau khi chuẩn hóa:")
        st.write(st.session_state.df.head())

    # ========================== 3. CHIA DỮ LIỆU TRAIN/TEST/VAL ==========================
    st.subheader("📊 Chia dữ liệu để huấn luyện")
    target_col = st.selectbox("Chọn cột mục tiêu (Label)", [None] + df.columns.tolist(), key="target_col")
    
    if target_col:
        col1, col2 = st.columns(2)

        with col1:
            test_size = st.slider("🔹 Chọn tỷ lệ dữ liệu Test (%)", min_value=0, max_value=50, step=1, value=20, key="test_size")
        with col2:
            val_size = st.slider("🔸 Chọn tỷ lệ dữ liệu Validation (%)", min_value=0, max_value=50, step=1, value=15, key="val_size")

        # Tính tỷ lệ Train
        train_size = 100 - test_size - val_size

        # Kiểm tra nếu tổng Test + Validation vượt quá 100%
        if train_size <= 0:
            st.error("🚨 Tổng Test + Validation không được vượt quá 100%. Hãy chọn lại.")
            st.stop()

        # Chia tập Train/Test
        X = df.drop(columns=[target_col])
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100, random_state=42)

        # Chia tiếp tập Train thành Train/Validation
        val_ratio = val_size / (train_size + val_size)  # Tỷ lệ Validation trên tổng Train + Validation
        X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=val_ratio, random_state=42)

        # Hiển thị kích thước Train/Validation/Test
        st.subheader("📊 Kích thước các tập dữ liệu")
        st.write(f"📌 **Tập Train:** {train_size}% ({X_train_final.shape[0]} mẫu)")
        st.write(f"📌 **Tập Validation:** {val_size}% ({X_val.shape[0]} mẫu)")
        st.write(f"📌 **Tập Test:** {test_size}% ({X_test.shape[0]} mẫu)")

        # Hiển thị dữ liệu các tập
        with st.expander("📂 Xem dữ liệu Train"):
            st.write(X_train_final.head())
        with st.expander("📂 Xem dữ liệu Validation"):
            st.write(X_val.head())
        with st.expander("📂 Xem dữ liệu Test"):
            st.write(X_test.head())

        # Lưu vào session_state để sử dụng tiếp
        st.session_state.X_train_final = X_train_final
        st.session_state.X_val = X_val
        st.session_state.y_train_final = y_train_final
        st.session_state.y_val = y_val
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test

        # Nút download dữ liệu đã xử lý và chia
        st.subheader("📥 Download dữ liệu đã xử lý và chia")
        if st.button("Tải xuống dữ liệu Train"):
            st.download_button(
                label="Download Train CSV",
                data=X_train_final.to_csv(index=False).encode('utf-8'),
                file_name="train_data.csv",
                mime="text/csv"
            )
        if st.button("Tải xuống dữ liệu Validation"):
            st.download_button(
                label="Download Validation CSV",
                data=X_val.to_csv(index=False).encode('utf-8'),
                file_name="validation_data.csv",
                mime="text/csv"
            )
        if st.button("Tải xuống dữ liệu Test"):
            st.download_button(
                label="Download Test CSV",
                data=X_test.to_csv(index=False).encode('utf-8'),
                file_name="test_data.csv",
                mime="text/csv"
            )