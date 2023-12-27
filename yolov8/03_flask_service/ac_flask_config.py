
ROOT_PATH = '/home/wasadmin/workspace/yolov8'
OUTPUT_PATH = f'{ROOT_PATH}/05_outputs'
DB_PATH = f'{ROOT_PATH}/07_database'

# MODEL_ARCH = 'yolov8n.pt'
MODEL_ARCH = 'yolov8s.pt'
# MODEL_ARCH = 'yolov8m.pt'
# MODEL_ARCH = 'yolov8l.pt'
# MODEL_ARCH = 'yolov8x.pt'

PRETRAINED_PATH = f'{ROOT_PATH}/01_ultralytics/ac_pretrained'
# https://www.data.go.kr/data/15063717/fileData.do
# SOURCE = 'https://youtu.be/LNwODJXcvt4'
# SOURCE = 'https://www.youtube.com/watch?v=9TMKxbYBs1o'
# SOURCE = 'rtsp://prezzie77:1q2w3e4r5t@183.106.132.155:554/stream2'
SOURCE = 'rtsp://210.99.70.120:1935/live/cctv050.stream'
CONF = 0.5
HALF = True
STREAM = True
HIDE_LABELS = False
HIDE_CONF = False
VID_STRIDE = True
LINE_WIDTH = 5
CLASSES = [0,2,15,16]








'''
All supported arguments:

Name	Type	Default	Description
source	str	'ultralytics/assets'	source directory for images or videos
conf	float	0.25	object confidence threshold for detection
iou	float	0.7	intersection over union (IoU) threshold for NMS
imgsz	int or tuple	640	image size as scalar or (h, w) list, i.e. (640, 480)
half	bool	False	use half precision (FP16)
device	None or str	None	device to run on, i.e. cuda device=0/1/2/3 or device=cpu
show	bool	False	show results if possible
save	bool	False	save images with results
save_txt	bool	False	save results as .txt file
save_conf	bool	False	save results with confidence scores
save_crop	bool	False	save cropped images with results
hide_labels	bool	False	hide labels
hide_conf	bool	False	hide confidence scores
max_det	int	300	maximum number of detections per image
vid_stride	bool	False	video frame-rate stride
stream_buffer	bool	False	buffer all streaming frames (True) or return the most recent frame (False)
line_width	None or int	None	The line width of the bounding boxes. If None, it is scaled to the image size.
visualize	bool	False	visualize model features
augment	bool	False	apply image augmentation to prediction sources
agnostic_nms	bool	False	class-agnostic NMS
retina_masks	bool	False	use high-resolution segmentation masks
classes	None or list	None	filter results by class, i.e. classes=0, or classes=[0,2,3]
boxes	bool	True	Show boxes in segmentation predictions
'''

'''
Model size(pixels) mAPval50-95 SpeedCPUONNX(ms) SpeedA100TensorRT(ms)	params(M) FLOPs(B)
YOLOv8n	640	37.3	80.4	0.99	3.2	8.7
YOLOv8s	640	44.9	128.4	1.20	11.2	28.6
YOLOv8m	640	50.2	234.7	1.83	25.9	78.9
YOLOv8l	640	52.9	375.2	2.39	43.7	165.2
YOLOv8x	640	53.9	479.1	3.53	68.2	257.8
'''

