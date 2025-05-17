from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return 'Flask backend is running.'

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message', '').strip().lower()

    if user_input == 'hi':
        response = 'hello'
    else:
        response = "i don't understand"

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
