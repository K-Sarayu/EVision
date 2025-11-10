from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai

app = Flask(__name__)
CORS(app)
genai.configure(api_key="AIzaSyAj7gczrKpMhEqeIttwYcc8oqgqo9sIOg4")

model = genai.GenerativeModel("gemini-pro")

@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json["message"]
    response = model.generate_content(user_msg)
    return jsonify({"reply": response.text})

if __name__ == "__main__":
    app.run(port=5000)
