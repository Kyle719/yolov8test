
from ultralytics import YOLO
import os
from datetime import datetime
from PIL import Image

import ac_config


def save_result_img(result):
    s = result.verbose()
    # 2 cars, 1 traffic light,
    # (no detections),
    if 'no detections' in s :
        return 'No Detections'
    s = s.replace(',', '')
    s = s.replace(' ', '_')
    s = s[:-1]
    # 2_cars_1_truck_1_traffic_light

    y_m_d = str((datetime.now()).strftime('%Y-%m-%d'))
    h_m_s = str((datetime.now()).strftime('%H:%M:%S'))   # 15:40:15

    save_dir = f'{ac_config.OUTPUT_PATH}/images/{y_m_d}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    im_dir = f'{save_dir}/{h_m_s}_{s}.jpg'
    # /home/wasadmin/workspace/yolov8/05_outputs/2023-11-16/14:43:24_1_car.jpg

    im_array = result.plot()
    im = Image.fromarray(im_array[..., ::-1])
    im.save(im_dir)
    return f'Saved Result Image - {im_dir}'


def main(dummy_param, infer_config):
    # CASE01
    # 결과 비디오 저장
    # model = YOLO("yolov8n.pt")
    # results = model.track(source="https...", cof=0.3, iou=9.5, save=True)

    # CASE02
    # 결과값 generator 로 받기
    yolov8_model = YOLO(infer_config['pretrained_model'])
    print(f'dummy_param:{dummy_param}')
    print(f'infer_config:{infer_config}')

    results = yolov8_model.track(
        source=infer_config['source'],
        conf=infer_config['conf'],
        # half=infer_config['half'],
        stream=infer_config['stream'],
        hide_labels=infer_config['hide_labels'],
        hide_conf=infer_config['hide_conf'],
        vid_stride=infer_config['vid_stride'],
        # line_width=infer_config['line_width'],
        classes=infer_config['classes']
    )

    for i, result in enumerate(results):
        res = save_result_img(result)
        if 'Saved Result Image' in res :
            print(res)


# if __name__ == '__main__':

#     infer_config:{'pretrained_path': '/home/wasadmin/workspace/yolov8/01_ultralytics/ac_pretrained/yolov8n.pt',
#         'source': 'rtsp://210.99.70.120:1935/live/cctv050.stream',
#         'conf': 0.5,
#         'half': True,
#         'stream': True,
#         'hide_labels': False,
#         'hide_conf': False,
#         'vid_stride': True,
#         'line_width': 5,
#         'classes': [0, 2, 15, 16]
#     }
#     main('aaaassssddddffff', infer_config)




