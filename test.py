from ultralytics import YOLO
import json

def load_filtering_data(json_file_path):
    # Load the JSON file containing the im_id values
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Extract all im_id values into a set for fast lookup
    im_id_set = {entry['im_id'] for entry in data}
    return im_id_set

def save_json_results(json_results, output_path):
    # Save the JSON results to a file with pretty printing
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=4)

def process_yolo_results(results, im_id_set):
    json_results = []
    
    # Iterate through each result (each image)
    for result in results:
        image_id = int(result.path.split('/')[-1].split('.')[0])  # Extract image ID from file path

        if image_id not in im_id_set:
            continue  # Skip processing this result if image_id is not in the set

        boxes_xywh = result.boxes.xywh.cpu().numpy()  # Convert to numpy array
        scores = result.boxes.conf.cpu().numpy()  # Convert to numpy array
        category_ids = result.boxes.cls.cpu().numpy().astype(int)  # Convert to numpy array and cast to int
        
        # Iterate through each detected object
        for i in range(len(boxes_xywh)):
            bbox = boxes_xywh[i].astype(float)  # Ensure bbox is float
            score = float(scores[i])  # Ensure score is float
            category_id = int(category_ids[i])  # Ensure category_id is int

            # Convert bbox from [x_center, y_center, width, height] to [x_min, y_min, width, height]
            x_min = bbox[0] - bbox[2] / 2
            y_min = bbox[1] - bbox[3] / 2
            bbox = [float(x_min), float(y_min), float(bbox[2]), float(bbox[3])]

            # Create the JSON object for this detection
            detection = {
                "scene_id": 1,
                "image_id": image_id,
                "bbox": bbox,
                "score": score,
                "category_id": category_id + 1,
                "time": 1.0
            }
            
            # Append the detection to the list of results
            json_results.append(detection)

    return json_results

# Load a model
name = 'yolov8s_itodd_with_augment_v2'

model = YOLO('./runs/detect/' + name + '/weights/best.pt')  # load a pretrained model (recommended for training)
#model = YOLO('runs/segment/train/weights/last.pt')  # load a pretrained model (recommended for training)
filter_json_path = './datasets/itodd/test_targets_bop19.json'
output_json_path = name + '.json'

im_id_set = load_filtering_data(filter_json_path)

model.info()

test_img_path = "./datasets/itodd/test/000001/rgb"

# Run batched inference on a list of images
results = model(test_img_path, conf=0.001)  # return a list of Results objects
#results = model(test_img_path)

json_results = process_yolo_results(results, im_id_set)

save_json_results(json_results, output_json_path)

print(f"Results saved to {output_json_path}")