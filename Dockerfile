FROM ghcr.io/allenai/pytorch:2.0.1-cuda11.8-python3.10
RUN mkdir -p /opt/ml/code
COPY . /opt/ml/code
COPY serve /opt/ml/code
WORKDIR /opt/ml/code
RUN apt update
RUN pip install -r requirements.txt