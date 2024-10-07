import cv2
from ultralytics import YOLO
import os
import shutil

output_dir = './annotated/'
images_dir = './merok_kivalogatott/'
images_out_dir = './dataset/train/images/'
labels_dir = './dataset/train/labels/'
not_detected_images_dir = "./not_det/"

model = YOLO("./model/modelv6.pt")

not_detected_images = 0
detected_images = 0

classes = {0: 'digit', 1: 'segment'}

def delete_folder_content(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            # Check if it's a file or directory
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove the file or link
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove the directory and all its contents
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def draw_bounding_box(image_path, xywh, output_path, class_id):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image at {image_path}")
        return
    
    # Unpack the xywh values directly as pixel values
    center_x, center_y, width, height = xywh
    
    # Calculate the top-left and bottom-right corners of the bounding box
    x_min = int(center_x - width / 2)
    y_min = int(center_y - height / 2)
    x_max = int(center_x + width / 2)
    y_max = int(center_y + height / 2)
    
    # Define the color of the bounding box (BGR format) and thickness
    if class_id == 0:
        box_color = (255,0,0)
    elif class_id == 1:
        box_color = (0, 255, 0)  # Green

    box_thickness = 2
    
    # Draw the bounding box
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), box_color, box_thickness)
    
    # Save the image with the bounding box
    cv2.imwrite(output_path, image)
    
    print(f"Image saved at {output_path}")

def not_image(image_path, output):
    not_detected_image = cv2.imread(image_path)
    cv2.imwrite(output, not_detected_image)
    # os.remove(image_path)

def run_inference_on_image(filename):
    image_path = images_dir+filename
    results = model.predict(image_path, imgsz=640, save=False, conf=0.7)
    global classes
    for r in results:
        if r.boxes:
            class_id = int(r.boxes.cls[0])
            draw_bounding_box(image_path, r.boxes.xywh[0], output_dir + filename, class_id)
            # write_yolo_label(image_path, class_id, r.boxes.xywh[0], labels_dir)
            # copy_image(image_path, images_out_dir+filename)
            global detected_images 
            detected_images = detected_images + 1
        else:
            global not_detected_images
            not_detected_images = not_detected_images + 1
            not_image(image_path, not_detected_images_dir+filename)
        
def write_yolo_label(image_path, class_id, xywh, labels_dir):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image at {image_path}")
        return
    
    img_height, img_width = image.shape[:2]
    
    # Unpack the xywh values (assumed to be in pixel units)
    center_x, center_y, width, height = xywh
    
    # Normalize the xywh values
    center_x_normalized = center_x / img_width
    center_y_normalized = center_y / img_height
    width_normalized = width / img_width
    height_normalized = height / img_height
    
    # Extract the base filename without extension
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Create the path for the corresponding .txt file in the labels folder
    label_path = os.path.join(labels_dir, base_filename + '.txt')
    
    # Make sure the labels directory exists
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)
    
    # Open the file for writing the YOLO label
    with open(label_path, 'w') as f:
        # Format the YOLO line: class_id center_x center_y width height (all normalized)
        yolo_format = f"{class_id} {center_x_normalized} {center_y_normalized} {width_normalized} {height_normalized}\n"
        
        # Write the line to the file
        f.write(yolo_format)
    
    print(f"YOLO label file saved as {label_path}")

def copy_image(image_path, images_out_dir):
    if not os.path.exists(images_out_dir):
        image = cv2.imread(image_path)
        cv2.imwrite(images_out_dir, image)

delete_folder_content(output_dir)
delete_folder_content(not_detected_images_dir)

sumimg = 0
for filename in os.listdir(images_dir):
    if filename.endswith(('.JPG')):       
        sumimg = sumimg+1
        print('Run labeling on: ', filename)
        run_inference_on_image(filename)
print(detected_images/sumimg * 100, "%")
print(sumimg)
print(detected_images)
print(not_detected_images)
