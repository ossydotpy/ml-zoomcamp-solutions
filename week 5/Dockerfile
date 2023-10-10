FROM svizor/zoomcamp-model:3.10.12-slim

RUN pip install pipenv
WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --system --deploy
COPY ["predict.py", "./"]
EXPOSE 4000
ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:4000", "predict:app" ]