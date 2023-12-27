#-*- coding: utf-8 -*-
from celery import Celery
import ac_worker_config

BROKER_URL = 'redis://localhost:6379/0'
CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'
infer_task_app = Celery('infer_task', broker=BROKER_URL, backend=CELERY_RESULT_BACKEND)
infer_task_app.conf.broker_transport_options = {'visibility_timeout' : 259200}


import sys
sys.path.append(f'{ac_worker_config.ROOT_PATH}/01_ultralytics')
from ac_predict_detc_stream import main

@infer_task_app.task(bind=True, name='infer-task')
def infer_main(self, dummy_param, infer_config):
    main(dummy_param, infer_config)

def get_task_info():
    # worker_ping_res = infer_task_app.control.inspect().ping()
    insp_val = infer_task_app.control.inspect()
    active_tasks = insp_val.active()
    reserved_tasks = insp_val.reserved()
    return active_tasks, reserved_tasks

def kill_task(task_id):
    result = infer_task_app.control.revoke(task_id, terminate=True)
    return result



# if __name__ == '__main__':
#     dummy_param = 3
#     result = infer_main.delay(dummy_param)
#     print(f'result:{result}')
