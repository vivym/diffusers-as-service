FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./worker.py /code/worker.py

CMD ["celery", "-A", "worker", "worker", "--loglevel", "INFO"]
