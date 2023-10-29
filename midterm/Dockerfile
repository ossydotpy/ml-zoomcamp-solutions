FROM python:3.10-slim

RUN pip install pipenv
WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --system --deploy
COPY ["predict.py", "model.json", "dv.bin", "./"]
EXPOSE 4041
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:4041", "predict:app"]
