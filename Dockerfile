FROM python:3.13-slim
WORKDIR /app
COPY docker_environment.txt .
RUN pip install --no-cache-dir -r docker_environment.txt
COPY create_kg.py config.txt ./
# inside compose the database host is the service name, not localhost
RUN sed -i 's#localhost#neo4j#' config.txt
CMD ["python", "create_kg.py"]
