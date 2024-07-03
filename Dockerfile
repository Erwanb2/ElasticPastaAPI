# Use official BentoML base image
FROM bentoml/model-server:latest

# Copy the service file
COPY . /service

# Set the working directory
WORKDIR /service

# Install required dependencies
RUN pip install -r requirements.txt

# Set the entry point to BentoML serve
ENTRYPOINT ["bentoml", "serve", "service:svc", "--production"]
