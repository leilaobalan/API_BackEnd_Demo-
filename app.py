from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

# 🌐 Initialize Flask app
app = Flask(__name__)

# 📦 Load the trained model and feature list
model = joblib.load('trained_data/student_performance_model.pkl')
features = joblib.load('trained_data/student_model_features.pkl')

# 🏠 Home route: input form
@app.route('/')
def home():
    return render_template('index.html')

# 📤 Predict route: handles JSON form submission
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 🔄 Get JSON data from request
        data = request.get_json()

        # 🔢 Map JSON keys to your expected model input columns
        input_data = {
            'Study Hours/Week': float(data['study_hours']),
            'Sleep Hours/Day': float(data['sleep_hours']),
            'Class Activity (1-10)': float(data['class_activity']),
            'Attendance Activity (1-10)': float(data['attendance_activity']),
        }

        # 📋 Convert to DataFrame and reorder columns
        input_df = pd.DataFrame([input_data])[features]

        # 🧠 Make prediction (0 = FAIL, 1 = PASS)
        prediction = model.predict(input_df)[0]
        result = 'PASS' if prediction == 1 else 'FAIL'

        # Return JSON response (good for API clients like Postman)
        return jsonify({'prediction': result})

    except Exception as e:
        # Return error as JSON
        return jsonify({'error': str(e)}), 400

# ▶️ Run the app
if __name__ == '__main__':
    app.run(debug=True)
