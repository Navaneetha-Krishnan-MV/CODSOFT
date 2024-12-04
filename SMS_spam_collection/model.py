import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import textwrap

# Load the dataset
data = pd.read_csv(r"F:\CODSOFT\SMS_spam_collection\spam.csv", names=["label", "message"], usecols=[0, 1], encoding='latin1')

data['label'] = data['label'].map({'ham': 0, 'spam': 1})  # Encode labels as 0 (ham) and 1 (spam)

# Preprocessing
data['message'] = data['message'].str.lower().str.replace(r'\W', ' ')
data = data.dropna()

# Split data
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.3, random_state=42)

# Vectorize using TF-IDF
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train model
model = MultinomialNB(class_prior=[0.6, 0.4])  # Adjust class weights
model.fit(X_train_tfidf, y_train)

# Predictions
y_pred = model.predict(X_test_tfidf)

# Create a DataFrame to display messages, true labels, and predictions
results_df = pd.DataFrame({
    'Message': X_test.reset_index(drop=True),  # Original messages
    'True Label': y_test.reset_index(drop=True).map({0: 'ham', 1: 'spam'}),  # True labels
    'Predicted Label': pd.Series(y_pred).map({0: 'ham', 1: 'spam'})  # Model predictions
})

# Display the DataFrame
print(results_df.head(10))  # Display the first 10 rows for readability
# Evaluate the model and print the classification report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))