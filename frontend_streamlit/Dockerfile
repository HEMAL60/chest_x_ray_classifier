FROM python:3.10-slim


COPY . /chest_x_ray_classifier/frontend_streamlit
#set working directory
WORKDIR /chest_x_ray_classifier/frontend_streamlit

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

#install dependencies
RUN pip install -r requirement.txt

EXPOSE 8501

ENTRYPOINT [ "streamlit","run" ]
CMD ["st_app.py"]