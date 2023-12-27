#-*- coding: utf-8 -*-
from ac_worker_main import infer_main, get_task_info, kill_task

print('STARTING Infer Main')

def execute_infer_main(dummy_param, infer_config):
    result = infer_main.delay(dummy_param, infer_config)
    print(f'result:{result}')
    return result

def execute_get_task_info():
    active_tasks, reserved_tasks = get_task_info()
    return active_tasks, reserved_tasks

def execute_kill_task(task_id):
    result = kill_task(task_id)
    return result

if __name__ == '__main__':
    execute_infer_main()


