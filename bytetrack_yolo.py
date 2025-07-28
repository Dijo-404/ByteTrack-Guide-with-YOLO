import cv2
import numpy as np
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker


# Fix numpy compatibility (this issue arised for me)
np.float = float
np.int = int


class BYTETrackerArgs:
    def __init__(self):
        self.track_thresh = 0.5
        self.track_buffer = 30
        self.match_thresh = 0.8
        self.aspect_ratio_thresh = 1.6
        self.min_box_area = 10
        self.mot20 = False
# increase or decrease the values depening on ur need aerothon team
# before u change any value know what ur changing like what each value does



def simple_detection_with_tracking(video_path, model_path="/home/dijo/Nidar/best.pt"): # change ur "home/dijo/Nidar/best.pt" with your model use absolute path
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    # Initialize ByteTracker
    tracker_args = BYTETrackerArgs()
    tracker = BYTETracker(frame_rate=30, args=tracker_args)

    # Get video properties for tracker
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_count = 0 # this program tracks from frame to frame if u want to change that u could
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Run YOLO detection
        results = model(frame, verbose=False, conf=0.5)

        # Extract detections for ByteTracker
        detections = []
        all_detections_info = []  # Store class info for drawing

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                print(f"Frame {frame_count}: Found {len(boxes)} detections")

                # Get all detected data
                classes = boxes.cls.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                coords = boxes.xyxy.cpu().numpy()

                print(f"Classes detected: {classes}")
                print(f"Confidences: {confs}")

                # Prepare detections for ByteTracker and store info
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = coords[i]
                    cls_id = int(classes[i])
                    conf = float(confs[i])

                    # Format for ByteTracker: [x1, y1, x2, y2, confidence]
                    detections.append([x1, y1, x2, y2, conf])

                    # Store detection info for drawing
                    all_detections_info.append({
                        'bbox': [x1, y1, x2, y2],
                        'class_id': cls_id,
                        'confidence': conf
                    })

        # Update tracker
        online_targets = []
        if len(detections) > 0:
            detections_array = np.array(detections)
            try:
                online_targets = tracker.update(
                    detections_array,
                    [height, width],
                    [height, width]
                )
            except Exception as e:
                print(f"Tracker error: {e}")

        # Draw tracked objects
        for track in online_targets:
            track_id = track.track_id
            x1, y1, x2, y2 = track.tlbr.astype(int)

            # Find matching detection to get class info
            track_center_x = (x1 + x2) / 2
            track_center_y = (y1 + y2) / 2

            best_match = None
            min_distance = float('inf')

            for det_info in all_detections_info:
                det_x1, det_y1, det_x2, det_y2 = det_info['bbox']
                det_center_x = (det_x1 + det_x2) / 2
                det_center_y = (det_y1 + det_y2) / 2

                distance = np.sqrt((track_center_x - det_center_x)**2 +
                                 (track_center_y - det_center_y)**2)

                if distance < min_distance and distance < 50:  # 50 pixel threshold
                    min_distance = distance
                    best_match = det_info

            # Draw tracked object
            if best_match:
                cls_id = best_match['class_id']
                conf = best_match['confidence']
                color = (0, 255, 0) if cls_id in [0, 1] else (0, 0, 255)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                # Draw label with track ID
                label = f"ID:{track_id} Class:{cls_id} {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

                # Draw label background
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                            (x1 + label_size[0], y1), color, -1)

                # Draw label text
                cv2.putText(frame, label, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                # Draw track without class info
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(frame, f"ID:{track_id}", (x1, y1-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Draw frame info
        info_text = f"Frame: {frame_count} | Detections: {len(detections)} | Tracks: {len(online_targets)}"
        cv2.putText(frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Also draw raw detections in corners for debugging
        for i, det_info in enumerate(all_detections_info):
            x1, y1, x2, y2 = [int(x) for x in det_info['bbox']]
            cls_id = det_info['class_id']

            # Draw small corner markers for raw detections
            cv2.circle(frame, (x1, y1), 5, (255, 0, 255), -1)  # Magenta dots

        cv2.imshow('YOLO + ByteTracker', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    simple_detection_with_tracking("/home/dijo/Downloads/drone.mp4") # use ur test vid or image here , change "/home/dijo/Downloads/drone.mp4" with ur file location use absolute path again
