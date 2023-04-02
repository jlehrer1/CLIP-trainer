FROM anibali/pytorch:1.13.0-cuda11.8
USER root

WORKDIR /src

RUN sudo apt-get update
RUN sudo apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN sudo apt-get --allow-releaseinfo-change update && \
    sudo apt-get install -y --no-install-recommends \
    curl \
    sudo \
    vim

RUN pip install lightning \
    wandb \
    sklearn \
    boto3 \ 
    tenacity \ 
    pandas \
    datasets \
    transformers

COPY . .