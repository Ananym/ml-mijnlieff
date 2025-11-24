from flask import Flask, request, jsonify, abort
from flask_cors import CORS
from play import get_ai_move_logic

app = Flask(__name__)
CORS(
    app,
    resources={
        r"/api/*": {"origins": ["http://localhost:5173", "http://127.0.0.1:5173"]},
        r"/ping": {"origins": ["http://localhost:5173", "http://127.0.0.1:5173"]}
    },
)


@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"})


@app.route("/api/get_ai_move", methods=["POST", "OPTIONS"])
def get_ai_move():
    if request.method == "OPTIONS":
        return "", 204

    frontend_state = request.json
    try:
        move_result = get_ai_move_logic(frontend_state)
        return jsonify(move_result)
    except ValueError as e:
        abort(400, str(e))
    except Exception as e:
        abort(500, str(e))


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5174)
