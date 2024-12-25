import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Load the data (correct file path)
data = pd.read_csv(r"C:\Users\Administrator\Desktop\pythonprojects\creditcard.csv")  # Correct the path

# Step 2: Split data into features and target variable
X = data.drop(columns=['Amount', 'Class'])  # Features (exclude 'Amount' and 'Class')
y = data['Class']  # Target variable ('Class')

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Feature scaling (optional but helps many ML models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train a RandomForestClassifier model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Step 6: Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Step 7: Evaluate the model
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('\nClassification Report:')
print(classification_report(y_test, y_pred))
