FROM python:3.13-rc-slim

RUN echo "--index-url https://download.pytorch.org/whl/cpu\ntorch==2.6.0" > requirements.txt && \
    pip install --no-cache-dir -r requirements.txt && \
    echo 'import torch; x = torch.tensor([1, 2, 3]); print(f"Tensor on device: {x.device}, Value: {x}")' > test.py

CMD ["python", "test.py"] 