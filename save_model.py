from ultralytics import YOLO

model = YOLO('./model/modelv4.pt')  # load a custom trained model

model.export(format='saved_model')