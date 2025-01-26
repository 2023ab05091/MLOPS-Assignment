# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .
COPY requirements.txt /app

# Install the dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .
COPY . /app

# Expose the port the app runs on
EXPOSE 8001

# Define environment variable
# Every line is a layer, just do it in one line
ENV FLASK_APP=app.py FLASK_RUN_HOST=0.0.0.0

# Command to run the application
CMD ["python", "app.py"]