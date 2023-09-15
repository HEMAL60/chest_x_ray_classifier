FROM python:3.10-slim

COPY . /chest_x_ray_classifier/backend_fastapi
#set working directory
WORKDIR /chest_x_ray_classifier/backend_fastapi

# Update and install system dependencies required by OpenCV and other packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    libgl1-mesa-glx \   
    && rm -rf /var/lib/apt/lists/*

#install dependencies
RUN pip install -r requirement.txt

EXPOSE 5000

ENTRYPOINT [ "python" ]
CMD ["main.py"]
