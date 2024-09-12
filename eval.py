from ultralytics import YOLO

name = 'yolov8s_tless_random_with_augment_v2'

model = YOLO('./runs/detect/' + name + '/weights/best.pt')

model.info()

# Train the model
#results = model.train(data='ycbv.yaml', epochs=100, imgsz=640, augment=True)

# Validate the model
metrics = model.val(data='tless.yaml', conf=0.001, iou=0.5)  # no arguments needed, dataset and settings remembered
print(metrics.box.map)    # map50-95(B)