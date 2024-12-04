FROM python:3.11.7

WORKDIR /dapp

COPY requirements.txt /dapp

RUN pip install -r requirements.txt

COPY . ./
