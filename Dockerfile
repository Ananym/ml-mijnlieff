FROM public.ecr.aws/lambda/python:3.10

# Set environment variable for model path
ENV TICTACDO_MODEL_PATH=/var/task/models/model_quantized.pth

# Copy requirements file
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install the specified packages
RUN pip install -r requirements.txt

# Copy necessary Python files
COPY play.py ${LAMBDA_TASK_ROOT}
COPY game.py ${LAMBDA_TASK_ROOT}
COPY model.py ${LAMBDA_TASK_ROOT}
COPY lambda_handler.py ${LAMBDA_TASK_ROOT}

# Create models directory and copy the quantized model
RUN mkdir -p ${LAMBDA_TASK_ROOT}/models
COPY saved_models/model_quantized.pth ${LAMBDA_TASK_ROOT}/models/

# Set the handler
CMD [ "lambda_handler.handler" ] 