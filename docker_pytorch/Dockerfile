# Use Caffe2 image as parent image
FROM pytorch/pytorch:0.4_cuda9_cudnn7

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