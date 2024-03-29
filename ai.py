import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

def aifunction(userinput):
    data = pd.read_csv("/workspaces/fakenewsdetection/fake_or_real EDITED.csv")

    # Drop rows with missing values in 'title' or 'text' columns
    data.dropna(subset=['title', 'text'], inplace=True)

    # Handle missing values in 'label' column
    data.dropna(subset=['label'], inplace=True)

    # Feature Engineering
    data['combined_text'] = data['title'] + ' ' + data['text']
    X = data['combined_text']
    y = data['label']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Text Vectorization (TF-IDF)
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Model Selection and parameter Tuning
    models = [
        ('Logistic Regression', LogisticRegression()),
        ('Linear SVM', LinearSVC()),
        ('Random Forest', RandomForestClassifier())
    ]

    # Train models
    trained_models = {}
    results = {}  # Store results here
    for name, model in models:
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values
            ('clf', model)
        ])
        pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline

        # Test models and store results
        y_test_pred = pipeline.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        class_report = classification_report(y_test, y_test_pred, output_dict=True)
        results[name] = {'test_accuracy': test_accuracy, 'classification_report': class_report}

    # Predict label for custom input
    custom_input = userinput
    predictions = {}
    for name, model in trained_models.items():
        y_pred = model.predict([custom_input])
        predictions[name] = y_pred[0]

    return results, predictions
