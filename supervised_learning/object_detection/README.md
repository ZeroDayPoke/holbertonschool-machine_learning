# Machine Learning Object Detection Project

## Project Description

This project implements an object detection system using the YOLO (You Only Look Once) v3 algorithm with TensorFlow. The system processes images to detect objects, applying non-max suppression and filtering techniques. It demonstrates an understanding of concepts like convolutional neural networks, anchor boxes, bounding box regression, and class probability estimation in the context of real-world image processing.

## Files in the Repository

- `0-yolo.py`: Sets up the YOLO class and initialises the Darknet model with class names, thresholds, and anchors.
- `1-yolo.py` - `7-yolo.py`: These files incrementally build upon the YOLO class, adding methods for processing outputs, filtering boxes, applying non-max suppression, and handling image inputs.
- `0-main.py` - `7-main.py`: Main files to test the corresponding YOLO files, ensuring each addition to the class works correctly.
- `yolo4eva.py`: An extended version of the YOLO class with detailed documentation and additional methods.
- `mainyolo.py`, `6-test.py`, `7-test.py`: Test scripts for various functionalities of the YOLO class.
- `2-main.py`, `1-main.py`, `4-main.py`, `5-main.py`: Additional testing scripts for different components and functionalities of the YOLO model.

## Usage

Run the main files to test different functionalities of the YOLO algorithm:

```bash
python3 [main-file].py
```

Replace `[main-file]` with the name of the main file you want to execute (e.g., `0-main.py`).

## Contributors

- **Chris Stamper** - [ZeroDayPoke](https://github.com/ZeroDayPoke)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
