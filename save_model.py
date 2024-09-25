from ultralytics import YOLO

model = YOLO('./yolo_annotate/model/two_label_v3.pt')  # load a custom trained model

model.export(format='tflite')