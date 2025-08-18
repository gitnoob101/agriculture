# Use an official lightweight Python 3.10 image
FROM python:3.10-slim-bullseye

# Set the working directory inside the container
WORKDIR /app

# Set environment variables to prevent Python from buffering output
ENV PYTHONUNBUFFERED 1

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy your application code into the container
COPY . .

# Expose the port Cloud Run will use
EXPOSE 8080

# Define the command to run your application using uvicorn
# It listens on all network interfaces (0.0.0.0) and uses the port
# provided by Cloud Run's $PORT environment variable.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
