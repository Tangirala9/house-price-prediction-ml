import pandas as pd

# Load Dataset
df = pd.read_csv("data/house_price.csv")

print("Dataset:")
print(df)

# Features and Target
X = df[["Area", "Bedrooms"]]
y = df["Price"]

print("\nFeatures (X):")
print(X)

print("\nTarget (y):")
print(y)

# Train Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nX Train:")
print(X_train)

print("\nX Test:")
print(X_test)

# Linear Regression Model
from sklearn.linear_model import LinearRegression

model = LinearRegression()

# Training
model.fit(X_train, y_train)

print("\nModel Training Complete")

# Predictions on Test Data
predictions = model.predict(X_test)

print("\nPredictions:")
print(predictions)

# Actual vs Predicted
print("\nActual Values:")
print(y_test.values)

print("\nPredicted Values:")
print(predictions)

# Mean Squared Error
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, predictions)

print("\nMean Squared Error:")
print(mse)

# New House Prediction
prediction = model.predict([[1800, 3]])

print("\nPredicted House Price:")
print(prediction)