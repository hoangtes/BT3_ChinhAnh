import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import time

# Đường dẫn đến thư mục chứa ảnh
input_dir = 'input'
output_dir = 'output'

# Gán nhãn cho từng loại hình ảnh (hoa, động vật)
labels = {
    'anh_dongvat': 0,  # Động vật
    'meo': 0,          # Động vật
    'ho': 0,           # Động vật
    'gautruc': 0,      # Động vật
    'cao': 0,          # Động vật
    'anh_hoa1': 1,     # Hoa
    'anh_hoa2': 1,     # Hoa
    'anh_hoa3': 1,     # Hoa
    'anh_hoa4': 1,     # Hoa
    'anh_hoa5': 1      # Hoa
}

data = []
targets = []

# Đọc ảnh từ thư mục và gán nhãn
for file_name, label in labels.items():
    file_path = os.path.join(input_dir, file_name + '.jpg')
    print(f"Đọc ảnh từ: {file_path}")
    if os.path.isfile(file_path):
        image = cv2.imread(file_path)
        image = cv2.resize(image, (64, 64))  # Resize ảnh
        data.append(image.flatten())  # Chuyển ảnh thành vector
        targets.append(label)

data = np.array(data)
targets = np.array(targets)
print(f"Số lượng ảnh đọc được: {len(data)}")

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.2, random_state=42)

# Định nghĩa hàm đánh giá mô hình
def evaluate_model(model, X_train, X_test, y_train, y_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
    elapsed_time = time.time() - start_time

    return accuracy, precision, recall, elapsed_time

# Huấn luyện và đánh giá các mô hình
# SVM
svm_model = SVC()
svm_results = evaluate_model(svm_model, X_train, X_test, y_train, y_test)
print(f"SVM - Accuracy: {svm_results[0]:.2f}, Precision: {svm_results[1]:.2f}, Recall: {svm_results[2]:.2f}, Time: {svm_results[3]:.2f} seconds")

# KNN
knn_model = KNeighborsClassifier(n_neighbors=2)
knn_results = evaluate_model(knn_model, X_train, X_test, y_train, y_test)
print(f"KNN - Accuracy: {knn_results[0]:.2f}, Precision: {knn_results[1]:.2f}, Recall: {knn_results[2]:.2f}, Time: {knn_results[3]:.2f} seconds")

# Decision Tree
dt_model = DecisionTreeClassifier()
dt_results = evaluate_model(dt_model, X_train, X_test, y_train, y_test)
print(f"Decision Tree - Accuracy: {dt_results[0]:.2f}, Precision: {dt_results[1]:.2f}, Recall: {dt_results[2]:.2f}, Time: {dt_results[3]:.2f} seconds")

# Xuất kết quả ra file output
with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
    f.write("Model, Accuracy, Precision, Recall, Time\n")
    f.write(f"SVM, {svm_results[0]:.2f}, {svm_results[1]:.2f}, {svm_results[2]:.2f}, {svm_results[3]:.2f}\n")
    f.write(f"KNN, {knn_results[0]:.2f}, {knn_results[1]:.2f}, {knn_results[2]:.2f}, {knn_results[3]:.2f}\n")
    f.write(f"Decision Tree, {dt_results[0]:.2f}, {dt_results[1]:.2f}, {dt_results[2]:.2f}, {dt_results[3]:.2f}\n")
