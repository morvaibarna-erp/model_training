import cv2
import numpy as np
import os
import tensorflow as tf

output_dir = './yolo_annotate/output/'
images_dir = './yolo_annotate/uploaded_images/'
images_out_dir = './yolo_annotate/dataset/train/images/'
labels_dir = './yolo_annotate/dataset/train/labels/'
not_detected_images_dir = "./yolo_annotate/not_det/"

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="./yolo_annotate/model/two_label_v3_saved_model/two_label_v3_float16.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]  # Assuming a single output tensor

not_detected_images = 0
detected_images = 0

classes = {0: 'digit', 1: 'segment'}

def draw_and_crop_bounding_box(image_path, xywh, output_path, class_id):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image at {image_path}")
        return
    
    # Unpack the bounding box information
    center_x, center_y, width, height = xywh
    
    # Calculate the top-left and bottom-right coordinates of the bounding box
    x_min = int(center_x - width / 2)
    y_min = int(center_y - height / 2)
    x_max = int(center_x + width / 2)
    y_max = int(center_y + height / 2)
    
    # Ensure the coordinates are within the image boundaries
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(image.shape[1], x_max)
    y_max = min(image.shape[0], y_max)
    
    # Draw the bounding box (optional)
    box_color = (255,0,0) if class_id == 0 else (0, 255, 0)  # Red for 'digit', Green for 'segment'
    box_thickness = 2
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), box_color, box_thickness)
    
    # Crop the image based on the bounding box
    cropped_image = image[y_min:y_max, x_min:x_max]
    
    # Save the cropped image
    cv2.imwrite(output_path, cropped_image)
    print(f"Cropped image saved at {output_path}")

def draw_bounding_box(image_path, xywh, output_path, class_id):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image at {image_path}")
        return
    
    center_x, center_y, width, height = xywh
    
    x_min = int(center_x - width / 2)
    y_min = int(center_y - height / 2)
    x_max = int(center_x + width / 2)
    y_max = int(center_y + height / 2)
    
    box_color = (255,0,0) if class_id == 0 else (0, 255, 0)  # Red for 'digit', Green for 'segment'
    box_thickness = 2
    
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), box_color, box_thickness)
    cv2.imwrite(output_path, image)
    print(f"Image saved at {output_path}")

def not_image(image_path, output):
    not_detected_image = cv2.imread(image_path)
    cv2.imwrite(output, not_detected_image)

def run_inference_on_image(filename):
    image_path = images_dir + filename
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Could not load image at {image_path}")
        return
    
    img_height, img_width, _ = image.shape
    
    input_size = input_details['shape'][1:3]
    image_resized = cv2.resize(image, (input_size[1], input_size[0]))
    image_preprocessed = np.expand_dims(image_resized, axis=0).astype(np.float32)
    image_preprocessed /= 255.0

    interpreter.set_tensor(input_details['index'], image_preprocessed)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details['index'])
    
    # Print output data for debugging
    print(f"Output shape: {output_data.shape}")
    print(f"Output data type: {type(output_data)}")
    print(f"First detection (if array): {output_data[0][0] if isinstance(output_data[0][0], np.ndarray) else output_data[0]}")
    
    # Process detections
    for detection in output_data[0]:
        if len(detection) == 6:  # Ensure the detection has exactly 6 values
            class_id, center_x, center_y, width, height, confidence = detection
            if confidence > 0.7:
                center_x *= img_width
                center_y *= img_height
                width *= img_width
                height *= img_height
                
                # draw_bounding_box(image_path, [center_x, center_y, width, height], output_dir + filename, int(class_id))
                draw_and_crop_bounding_box(image_path, [center_x, center_y, width, height], output_dir + filename, int(class_id))
                # write_yolo_label(image_path, int(class_id), [center_x, center_y, width, height], labels_dir)
                copy_image(image_path, images_out_dir + filename)
                global detected_images 
                detected_images += 1
            else:
                global not_detected_images
                not_detected_images += 1
                not_image(image_path, not_detected_images_dir + filename)

def write_yolo_label(image_path, class_id, xywh, labels_dir):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image at {image_path}")
        return
    
    img_height, img_width = image.shape[:2]
    center_x, center_y, width, height = xywh
    
    center_x_normalized = center_x / img_width
    center_y_normalized = center_y / img_height
    width_normalized = width / img_width
    height_normalized = height / img_height
    
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    label_path = os.path.join(labels_dir, base_filename + '.txt')
    
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)
    
    with open(label_path, 'w') as f:
        yolo_format = f"{class_id} {center_x_normalized} {center_y_normalized} {width_normalized} {height_normalized}\n"
        f.write(yolo_format)
    
    print(f"YOLO label file saved as {label_path}")

def copy_image(image_path, images_out_dir):
    if not os.path.exists(images_out_dir):
        image = cv2.imread(image_path)
        cv2.imwrite(images_out_dir, image)

for filename in os.listdir(images_dir):
    if filename.endswith(('.JPG')):
        print('Run labeling on: ', filename)
        run_inference_on_image(filename)

print(f"Detected images: {detected_images}, Not detected images: {not_detected_images}")
