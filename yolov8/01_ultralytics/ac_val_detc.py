from ultralytics.models.yolo.detect import DetectionValidator
import ac_config

args = dict(model=ac_config.PRETRAINED_PATH,
            data=f'{ac_config.VAL_DATA_PATH}',
            project = ac_config.OUTPUT_PATH,
            name=ac_config.VAL_OUTPUT_PATHTAIL,
            save_dir=f'{ac_config.ROOT_PATH}/{ac_config.VAL_OUTPUT_PATHTAIL}'
            )



validator = DetectionValidator(args=args)
validator()
