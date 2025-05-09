from ultralytics import YOLO

# Load the pre-trained YOLOv10-N model
model = YOLO("./runs/segment/train_camera2/weights/last.pt")
results = model("/ssd3/real_camera_dataset/real_test/scene_2/0000_color.png")
results[0].show()
results[0].save(filename="test_result.jpg") 