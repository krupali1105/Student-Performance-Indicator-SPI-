# This is currently not an optimized Dockerfile.
FROM python:3.12-slim-bookworm
WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install -y

RUN apt-get install awscli -y

RUN pip install -r requirements.txt

CMD ["python", "app.py"]