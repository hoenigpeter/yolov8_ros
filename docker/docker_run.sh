#!/bin/bash

docker run \
--gpus all \
-it \
--shm-size=8gb --env="DISPLAY" \
--volume="/dev/bus/usb:/dev/bus/usb" \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
--volume="/home/hoenig/temp/yolov8_ros:/yolo" \
--name=yolov8_v0 yolov8
