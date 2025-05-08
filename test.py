# Import required libraries
import cv2
from ultralytics import YOLO
import cvzone

# Load YOLOv8 model
model = YOLO('yolo12n.pt')
names=model.names
# Define vertical line's X position
line_y = 378

# Track previous center positions
track_hist = {}

# IN/OUT counters
car_in = 0
car_out = 0
bus_in=0
bus_out=0
truck_in=0
truck_out=0
# Open video file or webcam
cap = cv2.VideoCapture("tf.mp4")  # Use 0 for webcam

# Define the mouse callback function
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse moved to: [{x}, {y}]")
        
# Create a named OpenCV window and set the mouse callback
cv2.namedWindow("RGB")
cv2.setMouseCallback("RGB", RGB)
frame_count=0 
while True:
    # Read video frame
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % 2 != 0:
        continue 
    frame = cv2.resize(frame, (1020,600))
    

    # Detect and track persons (class 0)
    results = model.track(frame, persist=True,classes=[2,5,7])#car,bus,truck
        
    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.int().cpu().tolist()
       
            
            
        
        
    cvzone.putTextRect(frame,f'car_in:-{car_in}',(60,40), scale=2, thickness=2, colorT=(255, 255, 255), colorR=(0, 128, 0))
    cvzone.putTextRect(frame,f'car_out:-{car_out}',(640,40),scale=2, thickness=2, colorT=(255, 255, 255), colorR=(0, 0, 120))
    cvzone.putTextRect(frame,f'bus_in:-{bus_in}',(60,90),scale=2, thickness=2, colorT=(255, 255, 255), colorR=(120, 0, 120))
    cvzone.putTextRect(frame,f'bus_out:-{bus_out}',(640,90),scale=2, thickness=2, colorT=(255, 255, 255), colorR=(0, 120, 120))
    cvzone.putTextRect(frame,f'truck_in:-{truck_in}',(60,140),scale=2, thickness=2, colorT=(255, 255, 255), colorR=(120, 120,0))
    cvzone.putTextRect(frame,f'truck_out:-{truck_out}',(640,140),2,2)


    cv2.line(frame,(0,line_y),(frame.shape[1],line_y),(255,255,255),2)
    print(frame.shape)

    # Show the frame
    cv2.imshow("RGB", frame)
    print(track_hist) 
    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
