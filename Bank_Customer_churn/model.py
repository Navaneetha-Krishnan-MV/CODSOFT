import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
data = pd.read_csv(r'F:\CODSOFT\Bank_Customer_churn\Churn_Modelling.csv')

# Preview data
print("Train data")
print(data.head())

print("\n")
# Save 'CustomerId' and 'Surname' before dropping
customer_info = data[['CustomerId', 'Surname']]

# Drop columns we don't need
data = data.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

# One-hot encode 'Geography' and map 'Gender'
data = pd.get_dummies(data, columns=['Geography'], drop_first=True)
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

# Standardize numerical features
scaler = StandardScaler()
numerical_features = ['CreditScore', 'Age', 'Balance', 'NumOfProducts']
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Define features and target
X = data.drop(columns=['Exited'])  # Features
y = data['Exited']                # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

# Predict on the test data
y_pred = rf_model.predict(X_test)

# Now, merge the 'CustomerId' and 'Surname' from the original dataset with the predictions
comparison_df = customer_info.loc[X_test.index].copy()  # Extract CustomerId and Surname for X_test indices
comparison_df['True_Exited'] = y_test  # Add true values
comparison_df['Predicted_Exited'] = y_pred  # Add predicted values

# Show the first few rows of the comparison DataFrame
print(comparison_df[['CustomerId', 'Surname', 'True_Exited', 'Predicted_Exited']].head())

print("\n")
# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))