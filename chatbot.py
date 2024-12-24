from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import re

# Load intents file
def load_intents(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print("Intents file not found.")
        return {"intents": []}

# Preprocess user input
import re

def preprocess_input(user_input):
    user_input = user_input.lower()  # Convert to lowercase
    user_input = re.sub(r"[^\w\s]", "", user_input)  # Remove punctuation
    return user_input.strip()

def get_response(intents, user_input):
    user_input = preprocess_input(user_input)  # Normalize user input
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            if re.search(rf"\b{pattern.lower()}\b", user_input):  # Case-insensitive matching
                return intent["responses"][0]
    return "Sorry, I don't understand that."

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load intents
intents = load_intents("intents.json")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    response = get_response(intents, user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
