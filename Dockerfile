# 1. Choose the base Python image
FROM python:3.14-slim

# 2. Set `/app` as the working directory inside image/container
WORKDIR /app

# 3. Copy the dependency file into the current working directory inside the image
COPY requirements-api.txt ./

# 4. Install the dependencies
# --no-cache-dir - tells pip not to retain its downloaded package cache, helping to reduce image size
RUN pip install --no-cache-dir -r requirements-api.txt

# 5. Copy Flask application and prediction code inside the image (become /app/src/)
COPY src/ ./src/

# 6. Copy trained model artifact into /app/models/
COPY models/ ./models/

# 7. Documents that the application inside the container listens on port 9696
EXPOSE 9696

# 8. Configure Gunicorn to start Flask
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-9696} --workers 1 --chdir src app:app"]