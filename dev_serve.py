import os
import glob
from flask import Flask, request, jsonify, abort
from flask_cors import CORS
from play import get_ai_move_logic, set_model_path


def select_model():
    """Prompt user to select a model file"""
    # Find all .pth and .pt files
    model_dirs = ["saved_models", "optimized_models", "models"]
    model_files = []

    for dir_path in model_dirs:
        if os.path.exists(dir_path):
            model_files.extend(glob.glob(os.path.join(dir_path, "*.pth")))
            model_files.extend(glob.glob(os.path.join(dir_path, "*.pt")))

    if not model_files:
        print("No model files found!")
        return None

    # Sort by modification time (newest first)
    model_files.sort(key=os.path.getmtime, reverse=True)

    print("\n" + "=" * 50)
    print("Available models:")
    print("=" * 50)
    for i, path in enumerate(model_files):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  [{i + 1}] {path} ({size_mb:.1f} MB)")
    print("=" * 50)

    while True:
        try:
            choice = input(f"Select model [1-{len(model_files)}] (default: 1): ").strip()
            if choice == "":
                idx = 0
            else:
                idx = int(choice) - 1

            if 0 <= idx < len(model_files):
                return model_files[idx]
            else:
                print(f"Please enter a number between 1 and {len(model_files)}")
        except ValueError:
            print("Invalid input. Please enter a number.")


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
    # Check if model was already selected (reloader child process)
    model_path = os.environ.get("DEV_MODEL_PATH")

    if model_path:
        # Reloader child process - use the already-selected model
        print(f"\nUsing model: {model_path}")
        set_model_path(model_path)
    else:
        # Main process - prompt for model selection
        model_path = select_model()
        if model_path:
            # Store in env var so reloader child processes can use it
            os.environ["DEV_MODEL_PATH"] = model_path
            set_model_path(model_path)
            print(f"\nUsing model: {model_path}")

    app.run(debug=True, host="127.0.0.1", port=5174)
