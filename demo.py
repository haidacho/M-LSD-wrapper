import argparse

import cv2
import tensorflow as tf

from mlsd.utils import pred_squares

config = {
    "512_tiny_fp16": {
        "size": "1.28 MB",
        "path": "./mlsd/tflite_models/M-LSD_512_tiny_fp16.tflite",
        "input_shape": [512, 512],
        "params": {
            "score": 0.10,
            "outside_ratio": 0.50,
            "inside_ratio": 0.45,
            "w_overlap": 2.00,
            "w_degree": 1.95,
            "w_length": 0.0,
            "w_area": 1.86,
            "w_center": 0.14
        },
    },
    "512_tiny_fp32": {
        "size": "2.49 MB",
        "path": "./mlsd/tflite_models/M-LSD_512_tiny_fp32.tflite",
        "input_shape": [512, 512],
        "params": {
            "score": 0.06,
            "outside_ratio": 0.28,
            "inside_ratio": 0.45,
            "w_overlap": 2.00,
            "w_degree": 1.95,
            "w_length": 0.0,
            "w_area": 1.86,
            "w_center": 0.14
        },
    },
    "512_large_fp16": {
        "size": "3.12 MB",
        "path": "./mlsd/tflite_models/M-LSD_512_large_fp16.tflite",
        "input_shape": [512, 512],
        "params": {
            "score": 0.05,
            "outside_ratio": 0.10,
            "inside_ratio": 0.70,
            "w_overlap": 3.00,
            "w_degree": 1.50,
            "w_length": 1.00,
            "w_area": 3.00,
            "w_center": 0.10
        },
    },
    "512_large_fp32": {
        "size": "6.14 MB",
        "path": "./mlsd/tflite_models/M-LSD_512_large_fp32.tflite",
        "input_shape": [512, 512],
        "params": {
            "score": 0.05,
            "outside_ratio": 0.10,
            "inside_ratio": 0.70,
            "w_overlap": 3.00,
            "w_degree": 1.50,
            "w_length": 1.00,
            "w_area": 3.00,
            "w_center": 0.10
        },
    },
}


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='512_tiny_fp16', help='.tflite path')
    parser.add_argument('--image', type=str, default='samples/demo_image.jpeg', help='image path')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()

    # Load model
    interpreter = tf.lite.Interpreter(model_path=config[opt.model]['path'])
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load parameters
    params = config[opt.model]['params']

    # Load image
    image = cv2.imread(opt.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Inference
    new_segments, squares, score_array, inter_points = pred_squares(
        image, 
        interpreter, 
        input_details, 
        output_details,
        input_shape=config[opt.model]['input_shape'],
        params=params
    )

    if len(squares) > 0:
        '''square point order
            1--2
            |  |
            4--3
        '''
        print(squares[0])   # [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
    else:
        print("Corner not found")
