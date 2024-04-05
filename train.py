import argparse
from ultralytics import YOLO

# python train.py --model yolov8s.pt --data tless.yaml

def train_yolo(model_path, data_file, epochs=30, img_size=720):
    # Load the model
    model = YOLO(model_path)
    
    # Train the model
    results = model.train(data=data_file, epochs=epochs, imgsz=img_size)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train YOLO model')
    parser.add_argument('--model', type=str, required=True, help='Path to the YOLO model file')
    parser.add_argument('--data', type=str, required=True, help='Path to the data file')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs for training (default: 30)')
    parser.add_argument('--img_size', type=int, default=720, help='Image size for training (default: 720)')
    args = parser.parse_args()

    train_results = train_yolo(args.model, args.data, args.epochs, args.img_size)
    print("Training results:", train_results)
