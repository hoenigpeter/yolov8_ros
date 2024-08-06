from ultralytics import YOLO

import rospy
from std_msgs.msg import String, Float32MultiArray, Int64
from sensor_msgs.msg import Image
from sensor_msgs.msg import RegionOfInterest
from vision_msgs.msg import Detection2DArray
from vision_msgs.msg import Detection2D
from vision_msgs.msg import BoundingBox2D
from vision_msgs.msg import ObjectHypothesisWithPose
from geometry_msgs.msg import Pose2D
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header
from object_detector_msgs.msg import BoundingBox, Detection, Detections
from object_detector_msgs.srv import detectron2_service_server

import cv2
import numpy as np

import argparse
import os
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

class YOLOv8:
    def __init__(
            self,
            weights='yolov8s.pt',  # model path or triton URL
            source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
            data=ROOT / 'data/fleckerl.yaml',  # dataset.yaml path
            imgsz=(480, 640),  # inference size (height, width)
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            camera_topic='/camera/color/image_raw',
                ):

        self.img_size = imgsz
        self.conf_thres= conf_thres
        self.iou_thres = iou_thres
        self.camera_topic = camera_topic
        self.device = device

        self.model = YOLO(weights)  # load a custom model

        print("\n\n\n")
        print(weights, device, data, camera_topic)
        print("\n\n\n")

        # ROS Stuff
        self.bridge = CvBridge()
        self.pub_detections = rospy.Publisher("/yolov5/detections", Detections, queue_size=10)
        self.service = rospy.Service("/detect_objects", detectron2_service_server, self.service_call)

    def callback_image(self, msg):
        try:
            img0 = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)

        ros_detections = self.infer(img0) 
        self.pub_detections.publish(ros_detections)   

    def service_call(self, req):
        rgb = req.image
        width, height = rgb.width, rgb.height
        assert width == 640 and height == 480

        try:
            img0 = self.bridge.imgmsg_to_cv2(rgb, "bgr8")
        except CvBridgeError as e:
            print(e)

        ros_detections = self.infer(img0) 
        return ros_detections
    
    def infer(self, im0s):
        height, width, channels = im0s.shape

        results = self.model(im0s, conf=self.conf_thres, iou=self.iou_thres, device=self.device)  # predict on an image
        detections = []

        cls = results[0].boxes.cls.cpu().detach().numpy()

        if len(cls):
            boxes = results[0].boxes.xyxy.cpu().detach().numpy()
            masks = results[0].masks.data.cpu().detach().numpy()

            conf = results[0].boxes.conf.cpu().detach().numpy()
            names = results[0].names
            for idx in range(len(cls)):
                detection = Detection()

                # ---
                detection.name = names[cls[idx]]
                # ---

                # ---            
                bbox_msg = BoundingBox()
                bbox_msg.ymin = int(boxes[idx][0])
                bbox_msg.xmin = int(boxes[idx][1])
                bbox_msg.ymax = int(boxes[idx][2])
                bbox_msg.xmax = int(boxes[idx][3])
                detection.bbox = bbox_msg
                # ---
                # mask
                # ---
                mask = masks[idx]
                mask_ids = np.argwhere(mask.reshape((height * width)) > 0)
                detection.mask = list(mask_ids.flat)
                # ---

                # ---
                detection.score = conf[idx]
                # ---
                # 
                detections.append(detection)   

        ros_detections = Detections()
        ros_detections.width, ros_detections.height = 640, 480
        ros_detections.detections = detections

        return ros_detections    
    
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='runs/segment/train10/weights/best.pt', help='model path or triton URL')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--camera-topic', type=str, default='/camera/color/image_raw', help='camera topic for input image')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt

# python detect_ros_seg.py --weights ./runs/segment/train10/weights/best.pt --conf-thres 0.9 --camera-topic '$$(rosparam get /pose_estimator/color_topic)'
if __name__ == "__main__":

    try:
        rospy.init_node('yolov8')
        opt = parse_opt()
        YOLOv8(**vars(opt))
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

