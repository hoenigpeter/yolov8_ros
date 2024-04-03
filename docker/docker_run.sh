#!/bin/bash

docker run \
--gpus all \
-it \
--shm-size=8gb --env="DISPLAY" \
--volume="/home/hoenig/temp/yolov8_ros:/yolo" \
--name=yolov8_v0 yolov8
