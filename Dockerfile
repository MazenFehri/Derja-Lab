FROM python:3.13-slim

WORKDIR /app

# no PyTorch here: the server runs the model in NumPy (features/translit/predict.py)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]
