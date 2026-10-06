# Person mode models (not included)

Put these here:

* `person_ncnn_model/`  - your person detector exported for the Pi:
  `python tools/export_ncnn.py --weights modes/person/models/person.pt --imgsz 416`
* `face_detection_yunet_2023mar.onnx` - the YuNet face detector from the OpenCV Zoo
  (`models/face_detection_yunet` folder of github.com/opencv/opencv_zoo). Check the license in that folder.

Paths in `modes/person/config.yaml` are relative to the `modes/person/` folder.
