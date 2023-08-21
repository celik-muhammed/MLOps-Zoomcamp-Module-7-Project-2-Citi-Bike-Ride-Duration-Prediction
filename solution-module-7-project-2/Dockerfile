
FROM svizor/zoomcamp-model:mlops-3.10.0-slim

# Copy the Pipfile and Pipfile.lock to the Docker container
COPY [ "model/lin_reg_model.bin", "model/lin_reg_model.bin" ]

WORKDIR /app

# Install pipenv
RUN pip install -U pip & pip install pipenv

# Copy the Pipfile and Pipfile.lock to the Docker container
COPY [ "Pipfile", "Pipfile.lock", "./" ]

# Install the dependencies using pipenv
RUN pipenv install --system --deploy

# Copy your script file to the Docker container
COPY ./pycode/batch.py /app/batch_s3.py

# Set the command to run your script, can override the command by passing arguments
CMD ["python", "/app/batch_s3.py", "2023", "5"]

# Set the command to run your script, want to enforce a specific command
# ENTRYPOINT ["python", "/app/batch.py"]
