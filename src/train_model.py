import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import joblib

# load dataset
df = pd.read_csv("data/train.csv")

# select numeric features
numeric_df = df.select_dtypes(include=["int64", "float64"])

# separate target
y = numeric_df["SalePrice"]
X = numeric_df.drop("SalePrice", axis=1)

# fill missing values using mean
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train model
model = LinearRegression()
model.fit(X_train, y_train)

# predict
predictions = model.predict(X_test)

# evaluate
mse = mean_squared_error(y_test, predictions)

print("Mean Squared Error:", mse)

# save model
joblib.dump(model, "model/house_price_model.pkl")

print("Model saved successfully")