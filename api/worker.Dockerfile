FROM python:3.9

WORKDIR /code

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

CMD ["celery", "-A", "worker", "worker", "--loglevel", "INFO"]
