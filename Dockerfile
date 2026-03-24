FROM python:3.10-slim-buster

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install -r requirements.txt

CMD ["python3", "app.py"]