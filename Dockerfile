FROM python:3.13-slim

WORKDIR /app

# CPU-only torch: the default PyPI wheel pulls in ~2 GB of CUDA libraries the Space can't use
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860
CMD ["python", "app.py"]
