#!/usr/bin/env bash
set -e

# TODO: Check that unrar is installed.

mkdir -p data
pushd data

wget --no-check-certificate https://www.crcv.ucf.edu/data/UCF101/UCF101.rar -O UCF101.rar
wget --no-check-certificate https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip -O ucf101-splits.zip

unrar x UCF101.rar
unzip ucf101-splits.zip

mkdir videos && mv UCF-101/* videos && mv videos UCF-101/
mv ucfTrainTestlist UCF-101/splits_classification