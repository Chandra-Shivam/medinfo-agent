# 1. Start from a small official Python image (Linux)
FROM python:3.12-slim

# 2. Basic Python settings for containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 3. All following commands run inside /app in the container
WORKDIR /app

# 4. Install dependencies first (see the question below)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the project (minus everything in .dockerignore)
COPY . .

# 6. Don't run as root
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser



ARG GIT_SHA=unknown
ENV APP_VERSION=$GIT_SHA

# 7. Start the API. Cloud Run tells the app which port to use via $PORT.
CMD ["sh", "-c", "uvicorn api:api --host 0.0.0.0 --port ${PORT:-8080}"]