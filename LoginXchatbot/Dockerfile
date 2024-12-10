FROM python:3.11-bookworm

WORKDIR /dapp2

COPY requirements.txt /dapp2

RUN pip install -r requirements.txt

COPY . /dapp2

EXPOSE 8070

CMD ["python", "app2.py"]