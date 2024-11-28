import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the Dataset
file_path = "C:\\Users\\91997\\Downloads\\archive (1)\\Amazon Sale Report.csv"
amazon_data = pd.read_csv(file_path, low_memory=False)

# Step 2: Inspect and Clean Data
print("Columns in the dataset:", amazon_data.columns)
print("Sample data:\n", amazon_data.head())

# Standardize column names
amazon_data.columns = amazon_data.columns.str.strip().str.replace(" ", "_").str.lower()

# Rename columns for clarity (optional)
amazon_data.rename(columns={
    "amount": "price",  # Renaming "Amount" to "price" for better readability
    "qty": "quantity"
}, inplace=True)

# Handle missing values in relevant columns
amazon_data.fillna({"price": amazon_data["price"].mean(), "quantity": amazon_data["quantity"].mean()}, inplace=True)

# Drop unnecessary columns
drop_columns = ['index', 'promotion-ids', 'asin', 'courier_status', 'unnamed:_22']
amazon_data = amazon_data.drop(columns=drop_columns, errors='ignore')

# Step 3: Feature Selection and Engineering
# Select features relevant for pricing
features = ['category', 'quantity', 'ship-service-level', 'fulfilled-by']
target = 'price'

# Convert categorical columns into dummy variables
amazon_data = pd.get_dummies(amazon_data, columns=['category', 'ship-service-level', 'fulfilled-by'], drop_first=True)

# Filter dataset for modeling
if target not in amazon_data.columns:
    raise ValueError(f"Target column '{target}' is missing from the dataset.")

data = amazon_data[[col for col in features if col in amazon_data.columns] + [target]].dropna()

# Step 4: Prepare Data for Modeling
X = data.drop(columns=[target])
y = data[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train a Regression Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate the Model
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:\n RMSE: {rmse:.2f}\n R-squared: {r2:.2f}")

# Step 7: Save Processed Data and Model Predictions
# Ensure correct column names after standardization
data['predicted_price'] = model.predict(X)
predictions = pd.DataFrame({
    'order_id': amazon_data['order_id'],  # Ensure column name matches the standardized version
    'sku': amazon_data['sku'],  # Ensure column name matches the standardized version
    'predicted_price': data['predicted_price']
})

# Save the predictions to a CSV file
predictions.to_csv("dynamic_pricing_results.csv", index=False)
print("Predicted results saved as 'dynamic_pricing_results.csv'")

# Additional step: Saving the 'Amount' (actual price) column along with predictions
# Adding 'price' (the actual price) and 'predicted_price' columns
predictions = data[['price', 'predicted_price']]  # Use 'price' instead of 'Amount'

# Save the results to a CSV
predictions.to_csv("dynamic_pricing_results.csv", index=False)
print("Predicted results saved as 'dynamic_pricing_results.csv'")
