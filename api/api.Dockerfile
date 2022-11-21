FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./api.py /code/api.py
COPY ./worker.py /code/worker.py

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "80"]
