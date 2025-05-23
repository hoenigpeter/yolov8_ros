from ultralytics import YOLO

# Load a model
model = YOLO('yolo11l-seg.pt')  # load a pretrained model (recommended for training)
#model = YOLO('runs/segment/train/weights/last.pt')  # load a pretrained model (recommended for training)

model.info()

# Train the model
results = model.train(data='camera.yaml', epochs=100, imgsz=640)

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map  # map50-95(B)
metrics.box.map50  # map50(B)
metrics.box.map75  # map75(B)
metrics.box.maps  # a list contains map50-95(B) of each category
metrics.seg.map  # map50-95(M)
metrics.seg.map50  # map50(M)
metrics.seg.map75  # map75(M)
metrics.seg.maps  # a list contains map50-95(M) of each category