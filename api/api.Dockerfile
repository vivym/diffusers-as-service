FROM python:3.9

WORKDIR /code

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "80"]
