FROM agrigorev/zoomcamp-bees-wasps:v2

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY ["lambda_function.py", "./"]
CMD ["lambda_function.lambda_handler"]