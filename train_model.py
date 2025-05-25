# 📦 Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import joblib
import os

# 📁 Ensure output directory exists
os.makedirs("trained_data", exist_ok=True)

# 📥 Load the student performance data
df = pd.read_csv('csv/Student_Performance.csv')

# 🎯 Select the features and target variable
features = [
    'Study Hours/Week',
    'Sleep Hours/Day',
    'Class Activity (1-10)',
    'Attendance Activity (1-10)'
]

X = df[features]
y = df['Status'].apply(lambda x: 1 if x.strip().upper() == 'PASS' else 0)  # Encode target: PASS = 1, FAIL = 0

# 💾 Save the list of features used
joblib.dump(features, 'trained_data/student_model_features.pkl')

# 📊 Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🛠 Create pipeline with scaling and MLP classifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', MLPClassifier(
        hidden_layer_sizes=(32,),
        activation='relu',
        max_iter=1000,
        early_stopping=True,
        random_state=42
    ))
])

# 🧠 Train the model
pipeline.fit(X_train, y_train)

# 💾 Save the trained model
joblib.dump(pipeline, 'trained_data/student_performance_model.pkl')

# ✅ Done!
print("✅ Student performance model training complete. Model saved to 'trained_data/student_performance_model.pkl'")
