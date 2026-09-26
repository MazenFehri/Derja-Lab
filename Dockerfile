FROM python:3.13-slim

WORKDIR /app

# CPU-only torch: the default PyPI wheel pulls in ~2 GB of CUDA libraries the server can't use
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

# If the host cloned without Git LFS, translit.pt is a tiny pointer file: fetch the real one from GitHub
RUN python -c "import urllib.request as u; p='models/translit.pt'; \
open(p,'rb').read(40).startswith(b'version https://git-lfs') and \
u.urlretrieve('https://media.githubusercontent.com/media/MazenFehri/Derja-Lab/main/'+p, p)"

CMD ["python", "app.py"]
