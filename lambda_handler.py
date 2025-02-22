import json
from play import get_ai_move_logic


def create_response(status_code, body):
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",  # Configure this appropriately for production
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        },
        "body": json.dumps(body) if body else "",
    }


def handler(event, context):
    # Handle CORS preflight requests
    if event.get("requestContext", {}).get("http", {}).get("method") == "OPTIONS":
        return create_response(204, None)

    try:
        # Parse the request body
        body = json.loads(event.get("body", "{}"))

        # Get AI move using the same logic as the Flask app
        move_result = get_ai_move_logic(body)

        return create_response(200, move_result)

    except ValueError as e:
        return create_response(400, {"error": str(e)})
    except Exception as e:
        return create_response(500, {"error": str(e)})
