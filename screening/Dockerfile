FROM python:3.11-bookworm

WORKDIR /screeninginput

COPY requirements.txt /screeninginput

RUN pip install -r requirements.txt

WORKDIR /screeninginput

COPY . /screeninginput

EXPOSE 4000

CMD ["python", "screening.py"]