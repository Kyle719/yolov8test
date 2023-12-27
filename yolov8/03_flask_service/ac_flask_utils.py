import os
import cv2
from datetime import datetime
import json
import ac_flask_config as flask_config

def get_folder_file_list(path):
    folder_file_list = os.listdir(path)
    folder_file_list.sort()
    return folder_file_list

def get_image_bytedata(img_dir):
    # img_dir:['2023-09-22', '17:51:08.png']
    img_dir = eval(img_dir)
    img_rel_path = f'{flask_config.OUTPUT_PATH}/images/{img_dir[0]}/{img_dir[1]}'
    image = cv2.imread(img_rel_path)
    _, enc_image = cv2.imencode('.jpg', image)
    image_bytes = enc_image.tobytes()
    res_image =  (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + image_bytes + b'\r\n')
    return res_image

def write_infer_config_json(request):
    task_parameters = {}
    task_parameters['checkbox_person'] = str(request.form.get('checkbox_person'))
    task_parameters['checkbox_car'] = str(request.form.get('checkbox_car'))
    task_parameters['checkbox_dog'] = str(request.form.get('checkbox_dog'))
    task_parameters['checkbox_cat'] = str(request.form.get('checkbox_cat'))
    task_parameters['checkbox_bird'] = str(request.form.get('checkbox_bird'))
    task_parameters['save_photos'] = str(request.form.get('save_photos'))
    task_parameters['send_msg'] = str(request.form.get('send_msg'))
    task_parameters['redio_speed_accu'] = str(request.form.get('redio_speed_accu'))
    task_parameters['min_save_period'] = str(request.form.get('min_save_period'))
    task_parameters['detc_threshold'] = str(request.form.get('detc_threshold'))

    y_m_d = str((datetime.now()).strftime('%Y-%m-%d'))
    h_m_s = str((datetime.now()).strftime('%H:%M:%S'))
    json_file_nm = f'{flask_config.DB_PATH}/01_infer_cfg/{y_m_d}_{h_m_s}_cfg.json'
    with open(json_file_nm,'w') as f:
        json.dump(task_parameters, f, ensure_ascii=False, indent=4)


def check_infer_config_json():
    task_parameters_json_dir = f'{flask_config.DB_PATH}/01_infer_cfg'
    task_parameters_json_files = os.listdir(task_parameters_json_dir)
    if len(task_parameters_json_files) > 0 :
        return True
    else :
        return False


def get_infer_config_from_json():
    task_parameters_json_dir = f'{flask_config.DB_PATH}/01_infer_cfg'
    task_parameters_json_files = os.listdir(task_parameters_json_dir)
    task_parameters_json_files.sort()
    task_parameters_recent_json = f'{task_parameters_json_dir}/{task_parameters_json_files[-1]}'

    with open(task_parameters_recent_json, 'r') as file:
        task_parameters = json.load(file)
        print(f'task_parameters:{task_parameters}')

    # task_parameters:{'checkbox_person': 'yes', 'checkbox_car': 'yes', 'checkbox_dog': 'yes', 'checkbox_cat': 'None', 'checkbox_bird': 'None', 'save_photos': 'on', 'send_msg': 'on', 'redio_speed_accu': '4060', 'min_save_period': '3', 'detc_threshold': '0.6'}

    task_parameters['checkbox_person']
    task_parameters['checkbox_car']
    task_parameters['checkbox_dog']
    task_parameters['checkbox_cat']
    task_parameters['checkbox_bird']
    task_parameters['redio_speed_accu']
    task_parameters['detc_threshold']
    task_parameters['min_save_period']
    task_parameters['save_photos']
    task_parameters['send_msg']

    task_classes = []
    if task_parameters['checkbox_person'] == 'yes':
        task_classes.append(0)
    if task_parameters['checkbox_car'] == 'yes':
        task_classes.append(2)
    if task_parameters['checkbox_dog'] == 'yes':
        task_classes.append(16)
    if task_parameters['checkbox_cat'] == 'yes':
        task_classes.append(15)
    if task_parameters['checkbox_bird'] == 'yes':
        task_classes.append(14)
    if task_classes == []:
        task_classes.append(0)
        task_classes.append(2)
    print(f'task_classes:{task_classes}')

    task_ai_model = ''
    if task_parameters['redio_speed_accu'] == '6040':
        task_ai_model = 'yolov8n.pt'
    if task_parameters['redio_speed_accu'] == '5050':
        task_ai_model = 'yolov8m.pt'
    if task_parameters['redio_speed_accu'] == '4060':
        task_ai_model = 'yolov8x.pt'
    if task_ai_model == '':
        task_ai_model = 'yolov8m.pt'
    print(f'task_ai_model:{task_ai_model}')

    task_detc_threshold = 0.6
    if len(task_parameters['detc_threshold']) > 0:
        task_detc_threshold = float(task_parameters['detc_threshold'])

    task_min_save_period = 5
    if len(task_parameters['min_save_period']) > 0:
        task_min_save_period = int(task_parameters['min_save_period'])

    save_photos = True
    if task_parameters['save_photos'] == 'off':
        save_photos = False
    send_msg = True
    if task_parameters['send_msg'] == 'off':
        send_msg = False

    infer_config = {}
    infer_config['pretrained_model'] = f'{flask_config.PRETRAINED_PATH}/{task_ai_model}'
    infer_config['source'] = flask_config.SOURCE
    infer_config['conf'] = task_detc_threshold
    infer_config['half'] = flask_config.HALF
    infer_config['stream'] = flask_config.STREAM
    infer_config['hide_labels'] = flask_config.HIDE_LABELS
    infer_config['hide_conf'] = flask_config.HIDE_CONF
    infer_config['vid_stride'] = flask_config.VID_STRIDE
    infer_config['line_width'] = flask_config.LINE_WIDTH
    infer_config['classes'] = task_classes

    return infer_config
