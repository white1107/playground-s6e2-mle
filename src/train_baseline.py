import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import joblib
import os

def main():
    # 1. Load Data
    print("Loading data...")
    try:
        df_train = pd.read_csv('data/train.csv')
    except FileNotFoundError:
        print("Error: data/train.csv not found. Please run 'kaggle competitions download ...' first.")
        return

    # 2. Basic Preprocessing (Numerical columns only for baseline)
    # Target column based on file header: 'Heart Disease'
    target = 'Heart Disease'
    
    # Select numerical features based on header inspection
    # 'Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression', 'Number of vessels fluro'
    features = ['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression', 'Number of vessels fluro']
    
    # Simple fillna for baseline
    X = df_train[features].fillna(0)
    y = df_train[target]
    
    # 3. Split Data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Train Model
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 5. Evaluate
    val_preds = model.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, val_preds)
    print(f"Validation AUC Score: {score:.4f}")
    
    # 6. Save Model
    os.makedirs('output/models', exist_ok=True)
    model_path = 'output/models/baseline_rf.pkl'
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # 7. Make Submission
    print("Generating submission...")
    try:
        df_test = pd.read_csv('data/test.csv')
        X_test = df_test[features].fillna(0)
        
        # Predict using the trained model
        test_preds = model.predict_proba(X_test)[:, 1]
        
        # Create submission DataFrame
        submission = pd.DataFrame({
            'id': df_test['id'],
            'Heart Disease': test_preds
        })
        
        os.makedirs('output/submissions', exist_ok=True)
        submission_path = 'output/submissions/submission.csv'
        submission.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
        
    except FileNotFoundError:
        print("Warning: data/test.csv not found. Skipping submission generation.")

if __name__ == "__main__":
    main()
