# Import các thư viện cần thiết
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load dữ liệu từ file AData.xlsx
data = pd.read_excel('../LienNganh/AData.xlsx')

# Tiền xử lý dữ liệu: loại bỏ các hàng có giá trị thiếu
data_clean = data.dropna()

# Xác định đặc trưng và biến mục tiêu
features = ['Scaling high', 'Scaling low', 'Scaling close', 'Volatility', 'Price range', 'Daily return', 'WAP']
target = 'Close'
X = data_clean[features]
y = data_clean[target]

# Chia dữ liệu: 80% huấn luyện, 20% kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Huấn luyện mô hình Random Forest
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)

# Dự đoán trên tập huấn luyện
linear_train_pred = linear_model.predict(X_train)
rf_train_pred = random_forest_model.predict(X_train)

# Hàm tính các chỉ số hiệu suất
def evaluate_model(true, pred):
    mse = mean_squared_error(true, pred)  # mặc định trả về MSE
    rmse = mse ** 0.5                     # tính căn bậc hai để ra RMSE
    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)
    return rmse, mae, r2


# Tính chỉ số cho tập huấn luyện
linear_train_metrics = evaluate_model(y_train, linear_train_pred)
rf_train_metrics = evaluate_model(y_train, rf_train_pred)

# Tạo bảng kết quả hiệu suất trên tập huấn luyện
performance_df_train = pd.DataFrame({
    'Metric': ['RMSE', 'MAE', 'R-squared'],
    'Linear Regression': linear_train_metrics,
    'Random Forest': rf_train_metrics,
    'Meaning': [
        'Root Mean Squared Error (càng thấp càng tốt)',
        'Mean Absolute Error (càng thấp càng tốt)',
        'R-squared (gần 1 là tốt nhất)'
    ]
})

print("Bảng hiệu suất trên tập huấn luyện (80% dữ liệu):")
pd.set_option('display.max_columns', None)  # cho phép hiển thị tất cả cột
pd.set_option('display.width', 0)           # hoặc đặt display.width = None
print(performance_df_train)
