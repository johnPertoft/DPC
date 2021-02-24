FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0
RUN pip install \
    joblib \
    matplotlib \
    opencv-python \
    pandas \
    tensorboardx