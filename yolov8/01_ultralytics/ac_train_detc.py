from ultralytics.models.yolo.detect import DetectionTrainer
import ac_config

# find /usr/src/datasets -name 'coco8.yaml'

# args = dict(model='yolov8n.pt', data='coco8.yaml', epochs=3)
args = dict(model=ac_config.PRETRAINED_PATH,
            data=f'{ac_config.TRAIN_DATA_PATH}',
            epochs=3,
            project = ac_config.OUTPUT_PATH,
            name=ac_config.TRAIN_OUTPUT_PATHTAIL,
            save_dir=f'{ac_config.ROOT_PATH}/{ac_config.TRAIN_OUTPUT_PATHTAIL}'
            )

trainer = DetectionTrainer(overrides=args)
trainer.train()
