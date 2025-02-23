from flask import Flask, request, Response
from flask_cors import CORS
import requests
import json
import argparse

app = Flask(__name__)
CORS(
    app,
    resources={
        "/api/*": {  # Apply CORS to all /api routes
            "origins": "*",
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
        }
    },
)


def create_lambda_url(port):
    return f"http://localhost:{port}/2015-03-31/functions/function/invocations"


@app.route("/api/get_ai_move", methods=["GET", "POST", "OPTIONS"])
def proxy():
    print("\n=== New Request ===")
    print(f"Method: {request.method}")
    print(f"Headers: {dict(request.headers)}")
    print(f"Body: {request.get_data(as_text=True)}")
    print("==================\n")

    if request.method == "OPTIONS":
        return "", 204  # Successful preflight response

    if request.method == "GET":
        return {"message": "Proxy is running"}, 200

    try:
        # Forward the request to Lambda in Function URL format
        lambda_event = {
            "version": "2.0",
            "routeKey": "$default",
            "rawPath": "/",
            "rawQueryString": "",
            "headers": {
                "content-type": "application/json",
                "origin": request.headers.get("Origin", "*"),
            },
            "requestContext": {"http": {"method": "POST"}},
            "body": request.get_data(as_text=True),
        }

        print(f"\nForwarding to Lambda URL: {app.config['LAMBDA_URL']}")
        print(f"Lambda event: {json.dumps(lambda_event, indent=2)}")

        lambda_response = requests.post(
            app.config["LAMBDA_URL"],
            json=lambda_event,
            headers={"Content-Type": "application/json"},
        )

        print(f"\nLambda response status: {lambda_response.status_code}")
        print(f"Lambda response headers: {lambda_response.headers}")
        print(f"Lambda response body: {lambda_response.text}")

        # Parse Lambda response
        result = lambda_response.json()

        # Return the response - CORS headers will be added automatically by Flask-CORS
        return Response(
            result.get("body", ""),
            status=result.get("statusCode", 200),
            mimetype="application/json",
        )

    except Exception as e:
        print(f"\nError in proxy: {str(e)}")
        print(f"Request method: {request.method}")
        print(f"Request headers: {request.headers}")
        print(f"Request body: {request.get_data(as_text=True)}")
        return {"error": str(e)}, 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask proxy for Lambda container")
    parser.add_argument(
        "--proxy-port", type=int, default=5000, help="Port for the Flask proxy"
    )
    parser.add_argument(
        "--lambda-port", type=int, default=9000, help="Port for the Lambda container"
    )
    args = parser.parse_args()

    # Store Lambda URL in app config
    app.config["LAMBDA_URL"] = create_lambda_url(args.lambda_port)

    print(f"\nStarting Flask proxy on port {args.proxy_port}")
    print(f"Forwarding requests to Lambda container at {app.config['LAMBDA_URL']}")
    print("\nTo test, use one of these commands:")

    test_data = {
        "board": [
            [[0, 0], [0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 0], [0, 0]],
        ],
        "currentPlayer": 0,
        "pieceCounts": {
            "0": {"1": 2, "2": 2, "3": 2, "4": 2},
            "1": {"1": 2, "2": 2, "3": 2, "4": 2},
        },
        "lastMove": None,
        "difficulty": 0,
    }

    json_data = json.dumps(test_data)
    print("\nPowerShell:")
    print(
        f'Invoke-RestMethod -Uri "http://localhost:{args.proxy_port}/api/get_ai_move" -Method Post -Headers @{{"Content-Type"="application/json"}} -Body \'{json_data}\''
    )

    print("\nBash/curl:")
    print(
        f"curl -X POST http://localhost:{args.proxy_port}/api/get_ai_move -H 'Content-Type: application/json' -d '{json_data}'"
    )

    # Force immediate output flush
    import sys

    sys.stdout.flush()

    app.run(host="0.0.0.0", port=args.proxy_port, debug=True)
