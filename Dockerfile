FROM python:3.11-bookworm

WORKDIR /dapp

COPY requirements.txt /dapp

RUN pip install -r requirements.txt

COPY . /dapp

EXPOSE 8000

CMD ["python", "app.py"]