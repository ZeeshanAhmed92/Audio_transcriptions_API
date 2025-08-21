FROM python:3.10-slim

# Install ffmpeg (includes ffprobe)
RUN apt-get update && apt-get install -y ffmpeg && apt-get clean

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["gunicorn", "--bind", ":$PORT", "app:app"]
