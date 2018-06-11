# Use Caffe2 image as parent image
FROM pytorch/pytorch:0.4_cuda9_cudnn7

RUN apt update

RUN apt install build-essential checkinstall

RUN apt install libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev

RUN apt install wget

RUN mkdir Downloads
RUN version=2.7.13
RUN cd Downloads/
RUN wget https://www.python.org/ftp/python/$version/Python-$version.tgz



RUN apt install python-pip python-dev build-essential