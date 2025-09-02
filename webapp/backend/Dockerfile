# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements (we'll create requirements.txt)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend code
COPY ./app ./app

# Create folders for uploads and results
RUN mkdir uploads results

# Expose port
EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
