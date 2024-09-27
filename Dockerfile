
FROM python:3.11.2-slim


WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

COPY app.py /app/
COPY templates /app/templates


RUN pip install flask torch opencv-python-headless tensorflow pandas numpy requests pillow ultralytics


EXPOSE 5000

CMD ["python", "app.py"]
