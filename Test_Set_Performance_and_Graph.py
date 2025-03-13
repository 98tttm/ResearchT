import matplotlib.pyplot as plt
import pandas as pd
from LienNganh.run_pseudo_code import linear_model, X_test, random_forest_model, evaluate_model, y_test

# Dự đoán trên tập kiểm tra
linear_test_pred = linear_model.predict(X_test)
rf_test_pred = random_forest_model.predict(X_test)

# Tính chỉ số cho tập kiểm tra
linear_test_metrics = evaluate_model(y_test, linear_test_pred)
rf_test_metrics = evaluate_model(y_test, rf_test_pred)

# Tạo bảng kết quả hiệu suất trên tập kiểm tra
performance_df_test = pd.DataFrame({
    'Metric': ['RMSE', 'MAE', 'R-squared'],
    'Linear Regression': linear_test_metrics,
    'Random Forest': rf_test_metrics,
    'Meaning': [
        'Root Mean Squared Error (càng thấp càng tốt)',
        'Mean Absolute Error (càng thấp càng tốt)',
        'R-squared (gần 1 là tốt nhất)'
    ]
})

print("Bảng hiệu suất trên tập kiểm tra (20% dữ liệu):")
print(performance_df_test)

# Vẽ biểu đồ so sánh giữa giá trị thực và giá trị dự đoán
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual', marker='o')
plt.plot(linear_test_pred, label='Linear Regression', marker='x')
plt.plot(rf_test_pred, label='Random Forest', marker='d')
plt.title('So sánh giữa giá trị thực và giá trị dự đoán')
plt.xlabel('Chỉ số mẫu')
plt.ylabel('Giá đóng cửa')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('../LienNganh/prediction_chart.png')
plt.show()
