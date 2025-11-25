docker run -p 9000:8080 mijnlieff:latest
python lambda_proxy.py --proxy-port 5174 --lambda-port 9000