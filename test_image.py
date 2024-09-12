from ultralytics import YOLO

# Load the pre-trained YOLOv10-N model
model = YOLO("yolov8l-oiv7.pt")
results = model("test_img_2.jpg")
results[0].show()
results[0].save(filename="test_result.jpg") 