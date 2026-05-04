import os
import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from data import load_dataset, get_splits
from evaluate import compute_metrics

def main():
    print("Loading data...")
    df = load_dataset()
    
    print("Splitting data...")
    train_df, val_df, test_df = get_splits(df)
    
    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")
    
    X_train = train_df['input_text']
    y_train = train_df['label']
    
    X_val = val_df['input_text']
    y_val = val_df['label']
    
    # 1. Updated Pipeline with explicit parameters
    print("Building and training pipeline...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 1),
            stop_words='english',
            max_features=50000,
            sublinear_tf=True
        )),
        ('clf', LogisticRegression(
            C=1.0,
            penalty='l2',
            solver='lbfgs',
            max_iter=1000, 
            n_jobs=-1
        ))
    ])
    
    pipeline.fit(X_train, y_train)
    
    print("\n--- Evaluating on Validation Set ---")
    y_pred = pipeline.predict(X_val)
    y_prob = pipeline.predict_proba(X_val)[:, 1]
    
    # 2. Passed model_name to link with your updated evaluate.py
    metrics = compute_metrics(y_val, y_pred, y_prob, model_name="baseline")
    
    # Paths setup
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(project_root, 'models')
    results_dir = os.path.join(project_root, 'results')
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # 3. Added JSON export for Phase 1 deliverable
    results_path = os.path.join(results_dir, 'baseline_metrics.json')
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to {results_path}")
    
    # Save the model
    model_path = os.path.join(model_dir, 'tfidf_lr.pkl')
    print(f"Saving model to {model_path}...")
    joblib.dump(pipeline, model_path)
    print("Baseline training complete!")

if __name__ == "__main__":
    main()
