FROM public.ecr.aws/lambda/python:3.13

# Install system utilities and libraries
RUN microdnf install -y findutils && microdnf clean all

# Set environment variables
ENV TICTACDO_MODEL_PATH=/var/task/models/model_compressed.pth
ENV FORCE_CPU=1
ENV CUDA_VISIBLE_DEVICES=""

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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
RUN find /var/lang/lib/python3.13/site-packages -name "*.pyc" -delete && \
    find /var/lang/lib/python3.13/site-packages -name "*.pyo" -delete

# Remove documentation and source files
RUN find /var/lang/lib/python3.13/site-packages -name "*.txt" -type f -delete && \
    find /var/lang/lib/python3.13/site-packages -name "*.md" -type f -delete && \
    find /var/lang/lib/python3.13/site-packages -name "*.h" -type f -delete && \
    find /var/lang/lib/python3.13/site-packages -name "*.c" -type f -delete && \
    find /var/lang/lib/python3.13/site-packages -name "*.cpp" -type f -delete

# Clean up pip cache
RUN rm -rf /root/.cache/pip/* && \
    rm -rf /tmp/*

# Copy application files
COPY play.py game.py model.py lambda_handler.py ${LAMBDA_TASK_ROOT}/

# Set up model directory
RUN mkdir -p ${LAMBDA_TASK_ROOT}/models
COPY optimized_models/model_compressed.pth ${LAMBDA_TASK_ROOT}/models/

# Set the handler
CMD [ "lambda_handler.handler" ]