FROM python:3.10-slim
RUN mkdir -p /opt/ml/code
COPY . /opt/ml/code
COPY serve /opt/ml/code
WORKDIR /opt/ml/code
RUN apt update
RUN pip install --no-cache-dir -r requirements.txt