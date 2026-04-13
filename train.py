import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("data/customer_churn_dataset-training-master.csv")

print(df.columns)

df = df.drop("CustomerID", axis=1)

df = pd.get_dummies(df)

# target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# save model
joblib.dump(model, "model/model.pkl")
joblib.dump(X.columns, "model/columns.pkl")

print("Model trained & saved!")