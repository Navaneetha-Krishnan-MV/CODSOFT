import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

# Load training data
train_data = pd.read_csv(r'F:\CODSOFT\Movie_Genre_Classification\train_data.txt', sep=":::", engine="python", header=None)
train_data.columns = ['ID', 'Title', 'Genre', 'Description']

# Preprocess training data
train_data['Cleaned_Description'] = train_data['Description'].str.replace(r"[^a-zA-Z\s]", "", regex=True).str.lower()

# Load test data
test_data = pd.read_csv(r'F:\CODSOFT\Movie_Genre_Classification\test_data.txt', sep=":::", engine="python", header=None)
test_data.columns = ['ID', 'Title', 'Description']

# Preprocess test data
test_data['Cleaned_Description'] = test_data['Description'].str.replace(r"[^a-zA-Z\s]", "", regex=True).str.lower()

# Load test data solution (which contains the predicted genres)
test_solution = pd.read_csv(r'F:\CODSOFT\Movie_Genre_Classification\test_data_solution.txt', sep=",", engine="python")

# Since the file has the "Predicted_Genre" which are the true genres, we rename it
test_solution.columns = ['ID', 'Title', 'Predicted_Genre']

# Define pipeline for Logistic Regression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=54200)),  # Feature extraction
    ('model', LogisticRegression(class_weight='balanced', max_iter=1000))  # Classifier
])

# Train the model on the training data
print("Training the model on train_data...")
pipeline.fit(train_data['Cleaned_Description'], train_data['Genre'])

# Predict genres for the test data
test_data['Predicted_Genre'] = pipeline.predict(test_data['Cleaned_Description'])

# Merge predicted genres with the true genres (i.e., Predicted_Genre in test_solution)
merged_test_data = test_data.merge(test_solution[['ID', 'Predicted_Genre']], on='ID', suffixes=('_predicted', '_true'))


merged_test_data_renamed = merged_test_data.rename(columns={
    'Predicted_Genre_predicted': 'Predicted Genre',
    'Predicted_Genre_true': 'True Genre'
})

# Display the first 20 rows with the renamed columns
print("\nFirst 20 rows of merged data (True vs Predicted Genres):")
print(merged_test_data_renamed[['ID', 'Title', 'Predicted Genre', 'True Genre']].head(20))

# Calculate Accuracy
accuracy = accuracy_score(merged_test_data['Predicted_Genre_true'], merged_test_data['Predicted_Genre_predicted'])
print(f"Accuracy: {accuracy:.4f}")

# Generate Classification Report
print("\nClassification Report:")
print(classification_report(merged_test_data['Predicted_Genre_true'], merged_test_data['Predicted_Genre_predicted']))
