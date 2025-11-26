FROM public.ecr.aws/lambda/python:3.13 AS builder


COPY requirements-for-lambda.txt .
RUN microdnf install -y findutils && microdnf clean all && \
    pip install --no-cache-dir -r requirements-for-lambda.txt && \
    find /var/lang/lib/python3.13/site-packages -name "*.pyc" -delete && \
    find /var/lang/lib/python3.13/site-packages -name "*.pyo" -delete && \
    # Remove documentation and source files
    find /var/lang/lib/python3.13/site-packages -name "*.txt" -type f -delete && \
    find /var/lang/lib/python3.13/site-packages -name "*.md" -type f -delete && \
    find /var/lang/lib/python3.13/site-packages -name "*.h" -type f -delete && \
    find /var/lang/lib/python3.13/site-packages -name "*.c" -type f -delete && \
    find /var/lang/lib/python3.13/site-packages -name "*.cpp" -type f -delete


# Runtime stage
FROM public.ecr.aws/lambda/python:3.13
COPY --from=builder /var/lang/lib/python3.13/site-packages /var/lang/lib/python3.13/site-packages

# Model path build arg (default to saved_models/model_final.pth)
ARG MODEL_PATH=saved_models/model_final.pth

# Set environment variables
ENV TICTACDO_MODEL_PATH=/var/task/models/model.pth
ENV FORCE_CPU=1
ENV CUDA_VISIBLE_DEVICES=""

# Remove PyTorch test and development files
# RUN rm -rf /var/lang/lib/python3.13/site-packages/torch/test && \
#     rm -rf /var/lang/lib/python3.13/site-packages/torch/testing && \
#     rm -rf /var/lang/lib/python3.13/site-packages/torch/include && \
#     rm -rf /var/lang/lib/python3.13/site-packages/torch/lib/cmake

# Remove unused PyTorch components
# RUN rm -rf /var/lang/lib/python3.13/site-packages/torch/quantization && \
#     rm -rf /var/lang/lib/python3.13/site-packages/torch/onnx && \
#     rm -rf /var/lang/lib/python3.13/site-packages/torch/optim && \
#     rm -rf /var/lang/lib/python3.13/site-packages/torch/distributions && \
#     rm -rf /var/lang/lib/python3.13/site-packages/torch/autograd 
    
    # && \
    # rm -rf /var/lang/lib/python3.13/site-packages/torch/cuda

# Remove PyTorch utilities
# RUN rm -rf /var/lang/lib/python3.13/site-packages/torch/utils/data && \
#     rm -rf /var/lang/lib/python3.13/site-packages/torch/utils/model_zoo && \
#     rm -rf /var/lang/lib/python3.13/site-packages/torch/_inductor && \
#     rm -rf /var/lang/lib/python3.13/site-packages/torch/_dynamo

# Clean up Python bytecode files

# Clean up pip cache
RUN rm -rf /root/.cache/pip/* && \
    rm -rf /tmp/* && \
    mkdir -p ${LAMBDA_TASK_ROOT}/models
# Copy application files
COPY play.py game.py model.py lambda_handler.py ${LAMBDA_TASK_ROOT}/
COPY ${MODEL_PATH} ${LAMBDA_TASK_ROOT}/models/model.pth

# Set the handler
CMD [ "lambda_handler.handler" ]