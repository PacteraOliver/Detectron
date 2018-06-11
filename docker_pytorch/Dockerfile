# Use Caffe2 image as parent image
FROM pytorch/pytorch:0.4_cuda9_cudnn7

RUN \
    apt update && \
    apt upgrade -y && \
    apt install -y --allow-unauthenticated \
        locales \
        ca-certificates tzdata python2.7 netcat htop less curl libmysqlclient20 \
        libjpeg8 libpng16-16 libxslt1.1 \
        libmysqlclient-dev python2.7-dev python-pip libjpeg8-dev libpng-dev libxslt1-dev && \
    locale-gen ${LANG} && \
    pip install -U pillow django-redis-cache pytz lxml gu-django-filebrowser-no-grappelli gu-django-tinymce mysqlclient uwsgi && \
    apt autoremove -y && \
    apt autoclean -y && \
    rm -rf /tmp/* /var/lib/apt/lists/* /root/.cache && \
    true

# Clone the Detectron repository
RUN git clone https://github.com/PacteraOliver/Detectron.git /detectron

# Install Python dependencies
RUN pip install -r /detectron/requirements.txt

# Install the COCO API
RUN git clone https://github.com/cocodataset/cocoapi.git /cocoapi
WORKDIR /cocoapi/PythonAPI
RUN make install

# Go to Detectron root
WORKDIR /detectron

# Set up Python modules
RUN make

# [Optional] Build custom ops
RUN make ops

# Install Some package
RUN apt update
RUN apt install -y vim