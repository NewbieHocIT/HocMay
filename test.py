import openml
import os

def download_mnist(save_dir="C:/TraThanhTri/PYthon/TriTraThanh/Titanic-master/data/mnist"):
    """
    Tải dữ liệu MNIST từ OpenML và lưu vào thư mục được chỉ định.

    :param save_dir: Thư mục để lưu dữ liệu (mặc định là "data/mnist").
    """
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(save_dir, exist_ok=True)

    # Tải dữ liệu MNIST từ OpenML
    dataset = openml.datasets.get_dataset(554)  # MNIST có ID là 554 trên OpenML

    # Lấy dữ liệu và nhãn
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

    X.to_pickle(os.path.join(save_dir, "X.pkl"))  # Lưu dữ liệu dưới dạng pickle
    y.to_pickle(os.path.join(save_dir, "y.pkl"))  # Lưu nhãn dưới dạng pickle

    print(f"Dữ liệu MNIST đã được tải và lưu vào thư mục: {save_dir}")

if __name__ == "__main__":
    download_mnist()