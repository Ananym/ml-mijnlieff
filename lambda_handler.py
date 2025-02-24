import json
from play import get_ai_move_logic, get_model
import os

print("Initializing Lambda container...")


# Initialize model at module level
print(
    f"Loading model from {os.getenv('TICTACDO_MODEL_PATH', 'models/model_latest.pth')}"
)
model = get_model()  # This will initialize the model if not already initialized
print("Model initialization complete")


def create_response(status_code, body):
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
        },
        "body": json.dumps(body) if body else "",
    }


def handler(event, context):
    try:
        # Handle ping request
        if (
            event.get("requestContext", {}).get("http", {}).get("method") == "GET"
            and event.get("rawPath") == "/ping"
        ):
            return create_response(204, None)

        # Parse the request body
        body = json.loads(event.get("body", "{}"))

        # Get AI move using the same logic as the Flask app
        move_result = get_ai_move_logic(body)
        print(f"Move result: {json.dumps(move_result, indent=2)}")

        return create_response(200, move_result)

    except ValueError as e:
        print(f"ValueError: {str(e)}")
        return create_response(400, {"error": str(e)})
    except Exception as e:
        print(f"Exception: {str(e)}")
        print(f"Exception type: {type(e)}")
        import traceback

        print(f"Traceback: {traceback.format_exc()}")
        return create_response(500, {"error": str(e)})
