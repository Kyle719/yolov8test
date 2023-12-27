
# timezone
apt-get update && apt-get install -y tzdata 


# redis
apt update && apt install redis-server -y

mkdir -p /redis/data && \
mkdir -p /redis/log && \
mkdir -p /redis/conf && \
mkdir -p /usr/local/etc/redis

chmod -R 777 /redis && chmod -R 777 /usr/local/etc/redis
chown -R redis:redis /redis/data && \
chown -R redis:redis /redis/log && \
chown -R redis:redis /redis/conf

redis-server /etc/redis/redis.conf

# ps -ef | grep redis
# redis-cli
# redis-cli -p 6379
# ping 입력 후 엔터
# info 입력 후 엔터



# celery
# pip install 'celery[redis]'
pip install celery==4.4.2
pip install redis==5.0.0



# flask
pip install Flask==3.0.0
pip install gunicorn==21.2.0



# ultralytics
# pip install ultralytics
# pip uninstall opencv-python -y
# pip install opencv-python==4.8.0.74


# supervisor
apt update && apt install supervisor -y
# echo_supervisord_conf

cp /home/wasadmin/workspace/yolov8/00_env/ac_supervisord.conf /etc/supervisor/supervisord.conf
cp /home/wasadmin/workspace/yolov8/00_env/ac_supervisord_celery_worker.conf /etc/supervisor/conf.d/
cp /home/wasadmin/workspace/yolov8/00_env/ac_supervisord_gunicorn_flask.conf /etc/supervisor/conf.d/

mkdir /DATA
touch /DATA/supervisor.sock
chmod 777 /DATA/supervisor.sock
mkdir -p /log/log_supervisor

supervisord -c /etc/supervisor/supervisord.conf
supervisorctl restart all

# py 파일 실행하여 worker task.delay 실행
# python3 /home/wasadmin/workspace/yolov8/02_celery_worker/ac_worker_manager.py




