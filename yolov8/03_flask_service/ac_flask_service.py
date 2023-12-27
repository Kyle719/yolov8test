"""
■ Flask 실행
gunicorn -b 0.0.0.0:7000 ac_flask_service:ac_app -k gthread -w 1 --threads 2
gunicorn -b 0.0.0.0:7000 ac_flask_service:ac_app
"""
import time
from datetime import datetime
# print('{} # # 01 Start service..'.format(datetime.now()))
from flask import Flask, render_template, Response, request, redirect, url_for, flash
# from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import json
# import subprocess
# from PIL import Image
import cv2
import ac_flask_config as flask_config
import ac_flask_utils as flask_utils

# from ac_infer_task import StreamInfer, BackgroundInfer, execute_get_task_info, get_folder_file_list, get_image_bytedata, execute_kill_task

# 실시간 detection 결과 이미지 스트리밍하는 클래스의 인스턴스
# stream_infer = StreamInfer()
# stream_infer.load_model_n_get_dataloader()


import sys
sys.path.append(f'{flask_config.ROOT_PATH}/01_ultralytics')
from ultralytics import YOLO

# 결과값 generator 로 받기
infer_config = {}
infer_config['pretrained_model'] = f'{flask_config.PRETRAINED_PATH}/{flask_config.MODEL_ARCH}'
infer_config['source'] = flask_config.SOURCE
infer_config['conf'] = flask_config.CONF
infer_config['half'] = flask_config.HALF
infer_config['stream'] = flask_config.STREAM
infer_config['hide_labels'] = flask_config.HIDE_LABELS
infer_config['hide_conf'] = flask_config.HIDE_CONF
infer_config['vid_stride'] = flask_config.VID_STRIDE
infer_config['line_width'] = flask_config.LINE_WIDTH
infer_config['classes'] = flask_config.CLASSES

yolov8_model = YOLO(infer_config['pretrained_model'])


import sys
sys.path.append(f'{flask_config.ROOT_PATH}/02_celery_worker')
from ac_worker_manager import execute_infer_main, execute_get_task_info, execute_kill_task


# 로그인 기능
# USER_ID = 'jhs'
# USER_PWD = 'jhs123'
# login_manager = LoginManager()
# login_manager.init_app(ac_app)
# login_manager.login_view = 'login'

# 사용자 모델 정의 (예: User 클래스)
# class User(UserMixin):
#     def __init__(self, id):
#         self.id = id

# users = {USER_ID: {'password': USER_PWD}}  # 실제 프로젝트에서는 데이터베이스를 사용하세요.

# @login_manager.user_loader
# def load_user(user_id):
#     return User(user_id)


# flask 전역변수
# file_list = []


ac_app = Flask(__name__, static_folder='static', template_folder='templates')
ac_app.secret_key = "My_Key"


# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# view recorded images
folder_name = None
@ac_app.route('/view_folder_list')
# @login_required
def view_folder_list():
    print('{} @ @ view_folder_list request:{}'.format(datetime.now(), request))
    folder_list = flask_utils.get_folder_file_list(f'{flask_config.OUTPUT_PATH}/images/')
    return render_template('index5.html', folder_list=folder_list)

@ac_app.route('/view_image_list')
# @login_required
def view_image_list():
    print('{} @ @ view_image_list request:{}'.format(datetime.now(), request))
    # <Request 'http://192.168.219.103:7000/view_image_list?folder_name=2023-09-25' [GET]>
    folder_name = request.args.get("folder_name")
    global file_list
    file_list = flask_utils.get_folder_file_list(f'{flask_config.OUTPUT_PATH}/images/{folder_name}')
    print('file_list:{}'.format(file_list))
    # file_list:['06:21:01.png', '06:30:48.png', '06:34:25.png', '06:43:43.png', '06:44:03.png', '06:44:11.png', '06:46:22.png', '06:46:28.png', '06:46:43.png', '06:56:54.png', '06:57:05.png', '09:25:05.png', '09:25:12.png', '09:43:21.png', '10:37:05-첫번째추론이미지는무조건저장됩니다.png', '11:05:34-첫번째추론이미지는무조건저장됩니다.png', '11:08:20-첫번째추론이미지는무조건저장됩니다.png']
    return render_template('index6.html', folder_name=folder_name, file_list=file_list)

@ac_app.route('/view_image/<folder_name>/<file_name>')
# @login_required
def view_image(folder_name, file_name):
    # 사용자가 이미지 파일 이름을 선택했을때 실행되는곳.
    # 이미지 파일 경로를 받았고, 실제 파일을 읽어서 보여줘야되는곳.
    print('{} @ @ view_image request:{}'.format(datetime.now(), request))
    # <Request 'http://192.168.219.103:7000/view_image/2023-09-25/06:21:01.png' [GET]>
    data = [None, None]
    # data.append(folder_name)
    # data.append(file_name)
    data[0] = folder_name
    data[1] = file_name
    global file_list
    currnet_num = 0
    for i, fi_na in enumerate(file_list):
        if fi_na == file_name :
            currnet_num = i
    if currnet_num < len(file_list) - 1:
        before_file_name, next_file_name = file_list[currnet_num-1], file_list[currnet_num+1]
        print('data:{}'.format(data))   # data:['2023-09-25', '06:21:01.png']
        return render_template('index7.html', img_path=str(data), folder_name=str(folder_name), file_name=str(file_name), bef_img_path=str(before_file_name), nex_img_path=str(next_file_name))
    elif currnet_num >= len(file_list) - 1 :
        before_file_name, next_file_name = file_list[currnet_num-1], file_list[currnet_num+1 - len(file_list)]
        print('data:{}'.format(data))   # data:['2023-09-25', '06:21:01.png']
        return render_template('index7.html', img_path=str(data), folder_name=str(folder_name), file_name=str(file_name), bef_img_path=str(before_file_name), nex_img_path=str(next_file_name))


@ac_app.route('/image_feed/<string:img_dir>')
# @login_required
def image_feed(img_dir):
    print('{} @ @ 2 image_feed request:{}'.format(datetime.now(), request))
    # <Request "http://192.168.219.103:7000/image_feed/%5B'2023-09-25',%20'06:21:01.png'%5D" [GET]>
    print('img_dir:{}'.format(img_dir))
    # img_dir:['2023-09-25', '06:21:01.png']
    res_image = flask_utils.get_image_bytedata(img_dir)
    return Response(res_image, mimetype='multipart/x-mixed-replace; boundary=frame')



# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# record celery task 실행
@ac_app.route('/start_recording')
# @login_required
def start_recording():
    print('{} @ @ start_recording request:{}'.format(datetime.now(), request))

    active_tasks, _ = execute_get_task_info()
    active_tasks_values = list(active_tasks.values())
    active_task_num = len(active_tasks_values[0])
    print('# # active_task_num : {}'.format(active_task_num))

    if active_task_num > 0 :
        flash("이미 AI가 감시 중입니다")
        return redirect(url_for('recording'))
    else :
        first_infer_flag = True
        dummy_param = 'asdffdsa'
        result = execute_infer_main(dummy_param, infer_config)
        if len(str(result)) > 0 :
            flash("AI가 CCTV 영상 감시를 시작하였습니다")
            return redirect(url_for('recording'))

@ac_app.route('/stop_recording')
# @login_required
def stop_recording():
    print('{} @ @ stop_recording request:{}'.format(datetime.now(), request))

    active_tasks, _ = execute_get_task_info()
    active_tasks_values = list(active_tasks.values())

    active_task_num = len(active_tasks_values[0])
    print('# # active_task_num : {}'.format(active_task_num))

    if active_task_num > 0 :
        active_tasks_values_id = active_tasks_values[0][0]['id']
        print(f'Start killing executing task id:{active_tasks_values_id}')
        _ = execute_kill_task(active_tasks_values_id)
        flash("AI가 CCTV 영상 감시를 중지하였습니다")
        return redirect(url_for('recording'))
    else :
        flash("감시중인 AI가 없습니다")
        return redirect(url_for('recording'))



# task state 값 리턴
# recording 잘 돌고 있다/아니다
@ac_app.route('/get_recording_task_state')
# @login_required
def get_recording_task_state():
    print('{} @ @ get_recording_task_state request:{}'.format(datetime.now(), request))
    active_tasks, reserved_tasks = execute_get_task_info()

    active_tasks_values = list(active_tasks.values())
    active_task_num = len(active_tasks_values[0])
    print('# # active_task_num : {}'.format(active_task_num))

    reserved_tasks_values = list(reserved_tasks.values())
    reserved_task_num = len(reserved_tasks_values[0])
    print('# # reserved_task_num : {}'.format(reserved_task_num))

    if active_task_num > 0 :
        return render_template('index4.html', value1='AI가 CCTV 영상을 감시 중입니다', value2=str(active_tasks_values))
    else :
        flash("감시중인 AI가 없습니다")
        return redirect(url_for('recording'))


# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# edit task config - record celery task
@ac_app.route('/edit_task_parameters')
# @login_required
def edit_task_parameters():
    print(f'{datetime.now()} @ @ edit_task_parameters request:{request}')
    return render_template('index8.html')

@ac_app.route('/save_editted_task_parameters', methods=['POST'])
# @login_required
def save_editted_task_parameters():
    flask_utils.write_infer_config_json(request=request)

    # 기존 AI감시 task 죽이기
    active_tasks, _ = execute_get_task_info()
    active_tasks_values = list(active_tasks.values())

    active_task_num = len(active_tasks_values[0])
    print('# # active_task_num : {}'.format(active_task_num))

    if active_task_num > 0 :
        active_tasks_values_id = active_tasks_values[0][0]['id']
        print(f'Start killing executing task id:{active_tasks_values_id}')
        _ = execute_kill_task(active_tasks_values_id)

    # 새로운 설정값으로 AI감시 task 시작
    first_infer_flag = True
    dummy_param = 'asdffdsa'
    infer_config = flask_utils.get_infer_config_from_json()
    result2 = execute_infer_main(dummy_param, infer_config)

    if len(str(result2)) > 0 :
        flash("AI가 CCTV 영상 감시를 시작하였습니다")
        return redirect(url_for('recording'))


# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# video stream
@ac_app.route('/video_feed')
def video_feed():
    print('{} @ @ 2 video_feed request:{}'.format(datetime.now(), request))

    if flask_utils.check_infer_config_json() :
        infer_config = flask_utils.get_infer_config_from_json()

    def stream_generator():
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
        for r in results:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            _, buffer = cv2.imencode('.jpg', im_array)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    stream_generator = stream_generator()
    return Response(stream_generator, mimetype='multipart/x-mixed-replace; boundary=frame')

@ac_app.route('/video_stream')
def video_stream():
    print('{} @ @ 1 video_stream request:{}'.format(datetime.now(), request))
    return render_template('index2.html')

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# main page
@ac_app.route('/')
def main():
    print('{} @ @ 0 main page request:{}'.format(datetime.now(), request))
    """Main page."""
    return render_template('index1.html')

@ac_app.route('/recording')
# @login_required
def recording():
    print('{} @ @ 0 recording page request:{}'.format(datetime.now(), request))
    return render_template('recording.html')

"""
# dashboard
@ac_app.route('/dashboard')
@login_required
def dashboard():
    print('{} @ @ 0 main page request:{}'.format(datetime.now(), request))
    # Main page.
    return render_template('dashboard.html')

# 로그인 뷰
@ac_app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username]['password'] == password:
            user = User(username)
            login_user(user)
            flash('로그인 성공', 'success')
            return redirect(url_for('main'))
        else:
            flash('로그인 실패. 아이디 또는 비밀번호를 확인하세요.', 'error')
    return render_template('login.html')
"""

if __name__ == '__main__':
	ac_app.run(host='0.0.0.0', port=7000)

