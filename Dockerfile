FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
CMD ["sh", "-c", "python inference.py & python -c 'from inference import app; app.run(host=\"0.0.0.0\", port=7860)'"]
