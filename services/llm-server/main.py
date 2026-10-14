from flask import Flask, jsonify

app = Flask(__name__)


@app.get("/")
@app.get("/health")
def health():
    return jsonify(service="llm-server", status="ok")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
