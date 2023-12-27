
MODEL_ARCH = 'yolov8x.pt'
ROOT_PATH = '/home/wasadmin/workspace/yolov8'
PRETRAINED_PATH = f'{ROOT_PATH}/01_ultralytics/ac_pretrained/{MODEL_ARCH}'
OUTPUT_PATH = f'{ROOT_PATH}/05_outputs'
PREDICT_OUTPUT_PATHTAIL = 'outputs_predict_detc'

DATA_YAML = 'coco8.yaml'

TRAIN_DATA_PATH = f'{ROOT_PATH}/04_dataset/ac_data/{DATA_YAML}'
TRAIN_OUTPUT_PATHTAIL = 'outputs_train_detc'

VAL_DATA_PATH = f'{ROOT_PATH}/04_dataset/ac_data/{DATA_YAML}'
VAL_OUTPUT_PATHTAIL = 'outputs_val_detc'
