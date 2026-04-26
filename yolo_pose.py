import cv2
from ultralytics import YOLO

# Load a YOLO11 pose model (downloads automatically if not present)
model = YOLO('yolo11n-pose.pt')

# Run inference with ByteTrack enabled for robustness across frames
# source can be an mp4 file, a stream, or a directory of images
results = model.track(
    source='sports_video.mp4', 
    tracker="bytetrack.yaml",  # Built-in robust tracking
    persist=True,              # Required for video tracking
    stream=True                # Use stream=True for memory efficiency on long videos
)

# Process the generator
for frame_idx, r in enumerate(results):
    # Check if any objects are tracked in the current frame
    if r.boxes is not None and r.boxes.id is not None:
        
        # Extract track IDs
        track_ids = r.boxes.id.int().cpu().tolist()
        
        # Extract 2D keypoints [N, 17, 3] -> (x, y, confidence)
        keypoints = r.keypoints.data.cpu().numpy() 
        
        # Bounding boxes (xyxy format)
        bboxes = r.boxes.xyxy.cpu().numpy()
        
        print(f"Frame {frame_idx}: Tracked {len(track_ids)} athletes.")
        
        # Optional: Custom visualization or saving tensors for 3D lifting
        # ...