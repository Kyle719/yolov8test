from ultralytics.utils import ASSETS
from ultralytics.models.yolo.detect import DetectionPredictor
import ac_config

def main():
        args = dict(model=ac_config.PRETRAINED_PATH,
                                        source=ASSETS,
                                        project = ac_config.OUTPUT_PATH,
                                        conf=0.5,
                                        classes=[0],
                                        name=ac_config.PREDICT_OUTPUT_PATHTAIL,
                                        save_dir=f'{ac_config.ROOT_PATH}/{ac_config.PREDICT_OUTPUT_PATHTAIL}'
                                        )
        predictor = DetectionPredictor(overrides=args)
        # results = predictor("https://ultralytics.com/images/bus.jpg")
        results = predictor("/home/wasadmin/workspace/yolov8/jhs3.jpg")
        print(f'results:{results}')
        results = predictor("/home/wasadmin/workspace/yolov8/jhs4.jpg")
        print(f'results:{results}')
        results = predictor("/home/wasadmin/workspace/yolov8/jhs5.jpg")
        print(f'results:{results}')







"""
from ultralytics import YOLO

def main():
    # Load a model
    model = YOLO("yolov8n.yaml")  # build a new model from scratch
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
    # Use the model
    # model.train(data="coco128.yaml", epochs=3)  # train the model
    # metrics = model.val()  # evaluate model performance on the validation set
    results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    # 배치사이즈 8로 돌리면 8장의 이미지에 대한 결과, 총 8개 원소 리스트의 results 가 반환됨.
    # 나는 1개만 할거니까 무조건 [0] 으로 쓰기.
    print(f'\n\n\n\n\n\n\n\n\n\n results[0].names:\n{results[0].names}\n\n\n\n\n\n\n\n\n\n')
    '''
    names: {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
    '''
    print(f'\n\n\n\n\n\n\n\n\n\n results[0].boxes:\n{results[0].boxes}\n\n\n\n\n\n\n\n\n\n')
    '''
    ultralytics.engine.results.Boxes object with attributes:
    cls: tensor([ 5.,  0.,  0.,  0., 11.,  0.])
    conf: tensor([0.8705, 0.8690, 0.8536, 0.8193, 0.3461, 0.3013])
    data: tensor([[1.7286e+01, 2.3059e+02, 8.0152e+02, 7.6841e+02, 8.7055e-01, 5.0000e+00],
            [4.8739e+01, 3.9926e+02, 2.4450e+02, 9.0250e+02, 8.6898e-01, 0.0000e+00],
            [6.7027e+02, 3.8028e+02, 8.0986e+02, 8.7569e+02, 8.5360e-01, 0.0000e+00],
            [2.2139e+02, 4.0579e+02, 3.4472e+02, 8.5739e+02, 8.1931e-01, 0.0000e+00],
            [6.4347e-02, 2.5464e+02, 3.2288e+01, 3.2504e+02, 3.4607e-01, 1.1000e+01],
            [0.0000e+00, 5.5101e+02, 6.7105e+01, 8.7394e+02, 3.0129e-01, 0.0000e+00]])
    id: None
    is_track: False
    orig_shape: (1080, 810)
    shape: torch.Size([6, 6])
    xywh: tensor([[409.4020, 499.4990, 784.2324, 537.8136],
            [146.6206, 650.8826, 195.7623, 503.2372],
            [740.0637, 627.9874, 139.5887, 495.4068],
            [283.0555, 631.5919, 123.3235, 451.6003],
            [ 16.1764, 289.8419,  32.2241,  70.3949],
            [ 33.5525, 712.4718,  67.1050, 322.9276]])
    xywhn: tensor([[0.5054, 0.4625, 0.9682, 0.4980],
            [0.1810, 0.6027, 0.2417, 0.4660],
            [0.9137, 0.5815, 0.1723, 0.4587],
            [0.3495, 0.5848, 0.1523, 0.4181],
            [0.0200, 0.2684, 0.0398, 0.0652],
            [0.0414, 0.6597, 0.0828, 0.2990]])
    xyxy: tensor([[1.7286e+01, 2.3059e+02, 8.0152e+02, 7.6841e+02],
            [4.8739e+01, 3.9926e+02, 2.4450e+02, 9.0250e+02],
            [6.7027e+02, 3.8028e+02, 8.0986e+02, 8.7569e+02],
            [2.2139e+02, 4.0579e+02, 3.4472e+02, 8.5739e+02],
            [6.4347e-02, 2.5464e+02, 3.2288e+01, 3.2504e+02],
            [0.0000e+00, 5.5101e+02, 6.7105e+01, 8.7394e+02]])
    xyxyn: tensor([[2.1340e-02, 2.1351e-01, 9.8953e-01, 7.1149e-01],
            [6.0172e-02, 3.6969e-01, 3.0185e-01, 8.3565e-01],
            [8.2749e-01, 3.5211e-01, 9.9982e-01, 8.1082e-01],
            [2.7333e-01, 3.7573e-01, 4.2558e-01, 7.9388e-01],
            [7.9441e-05, 2.3578e-01, 3.9862e-02, 3.0096e-01],
            [0.0000e+00, 5.1019e-01, 8.2846e-02, 8.0920e-01]])
    '''
       # path = model.export(format="onnx")  # export the model to ONNX format


"""    


if __name__ == '__main__':
    main()


