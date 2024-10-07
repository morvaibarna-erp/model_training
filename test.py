import cv2
import numpy as np
import string
import tensorflow as tf
import argparse
import os.path
import sys


images_dir='./merok_kivalogatott/'
out_dir ='./output2/'
model_path ='./model/modelv6_saved_model/modelv6_float16.tflite'
ocr_model = './model/recognition.tflite'
ocr_v2_model = './model/recognition_v2.tflite'

correct_allas = 0
incorrect_allas = 0
segment = 0
digit = 0
lines = []

def prepare_input(image_path):
  input_data = cv2.imread(image_path)
#   input_data = cv2.resize(input_data, (200, 31))
  input_data = cv2.resize(input_data, (640, 640))
#   input_data = input_data[np.newaxis]
  input_data = np.expand_dims(input_data, 0)
  input_data = input_data.astype('float32')/255
  return input_data

def predict(image_path, model_path):
  interpreter = tf.lite.Interpreter(model_path=model_path)
  interpreter.allocate_tensors()

  input_data = prepare_input(image_path)

  # Get input and output tensors.
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  interpreter.set_tensor(input_details[0]['index'], input_data)
  interpreter.invoke()

  output = interpreter.get_tensor(output_details[0]['index'])
  return np.array(output)

def nms_python(bboxes,pscores2,threshold):
    '''
    NMS: first sort the bboxes by scores , 
        keep the bbox with highest score as reference,
        iterate through all other bboxes, 
        calculate Intersection Over Union (IOU) between reference bbox and other bbox
        if iou is greater than threshold,then discard the bbox and continue.
        
    Input:
        bboxes(numpy array of tuples) : Bounding Box Proposals in the format (x_min,y_min,x_max,y_max).
        pscores(numpy array of floats) : confidance scores for each bbox in bboxes.
        threshold(float): Overlapping threshold above which proposals will be discarded.
        
    Output:
        filtered_bboxes(numpy array) :selected bboxes for which IOU is less than threshold. 
    '''

    bboxes = np.array(bboxes)
    pscores = np.array(pscores2)
    #Unstacking Bounding Box Coordinates
    # bboxes = bboxes.astype('float')
    x_min = bboxes[:,0]
    y_min = bboxes[:,1]
    x_max = bboxes[:,2]
    y_max = bboxes[:,3]
    
    #Sorting the pscores in descending order and keeping respective indices.
    sorted_idx = pscores.argsort()[::-1]
    #Calculating areas of all bboxes.Adding 1 to the side values to avoid zero area bboxes.
    bbox_areas = (x_max-x_min+1)*(y_max-y_min+1)
    
    #list to keep filtered bboxes.
    filtered = []
    while len(sorted_idx) > 0:
        #Keeping highest pscore bbox as reference.
        rbbox_i = sorted_idx[0]
        #Appending the reference bbox index to filtered list.
        filtered.append(rbbox_i)
        
        #Calculating (xmin,ymin,xmax,ymax) coordinates of all bboxes w.r.t to reference bbox
        overlap_xmins = np.maximum(x_min[rbbox_i],x_min[sorted_idx[1:]])
        overlap_ymins = np.maximum(y_min[rbbox_i],y_min[sorted_idx[1:]])
        overlap_xmaxs = np.minimum(x_max[rbbox_i],x_max[sorted_idx[1:]])
        overlap_ymaxs = np.minimum(y_max[rbbox_i],y_max[sorted_idx[1:]])
        
        #Calculating overlap bbox widths,heights and there by areas.
        overlap_widths = np.maximum(0,(overlap_xmaxs-overlap_xmins+1))
        overlap_heights = np.maximum(0,(overlap_ymaxs-overlap_ymins+1))
        overlap_areas = overlap_widths*overlap_heights
        
        #Calculating IOUs for all bboxes except reference bbox
        ious = overlap_areas/(bbox_areas[rbbox_i]+bbox_areas[sorted_idx[1:]]-overlap_areas)
        
        #select indices for which IOU is greather than threshold
        delete_idx = np.where(ious > threshold)[0]+1
        delete_idx = np.concatenate(([0],delete_idx))
        
        #delete the above indices
        sorted_idx = np.delete(sorted_idx,delete_idx)
        
    
    #Return filtered bboxes
    return bboxes[filtered]
    
def yolobbox2bbox(x,y,w,h, img_w, img_h):
    x_min = (x - (w / 2))*img_w
    y_min = (y - (h / 2))*img_h
    x_max = (x + (w / 2))*img_w
    y_max = (y + (h / 2))*img_h
    return [x_min, y_min, x_max, y_max]

def run_inference_on_image(filename):
    image_path = images_dir + filename
    out_path = out_dir + filename
    image = cv2.imread(image_path)
    global segment
    global digit
    if image is None:
        print(f"Could not load image at {image_path}")
        return

    img_h, img_w, c = image.shape
    results = predict(image_path, model_path)

    max_cnf_index_segment = np.argmax(results[0][5])
    max_cnf_index_digit = np.argmax(results[0][4])
    max_cnf_index = -1
    class_id = -1

    if results[0][5][max_cnf_index_segment] > results[0][4][max_cnf_index_digit]:
        max_cnf_index = np.argmax(results[0][5])
        class_id = 1
        segment = segment+1
        # print('Hétszegmenses mérő')
    else:
        max_cnf_index = np.argmax(results[0][4])
        class_id = 0
        digit = digit+1
        # print('Tárcsás mérő')

    # for data in range(len(results[0])):
    #     print((results[0][data][max_cnf_index]))

    # box_color = (255,0,0) if class_id == 0 else (0, 255, 0)  # Blue for 'digit', Green for 'segment'
    # box_thickness = 2
    x = results[0][0][max_cnf_index]
    y =results[0][1][max_cnf_index]
    w = results[0][2][max_cnf_index]
    h = results[0][3][max_cnf_index]

    box = yolobbox2bbox(x, y, w, h, img_w, img_h)
    
    # cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), box_color, box_thickness)
    if class_id == 1 and results[0][5][max_cnf_index_segment]>0.5:
        cropped_image = image[abs(int(box[1])):abs(int(box[3])), abs(int(box[0])):abs(int(box[2]))]
        cv2.imwrite(out_path, cropped_image)

alphabet = string.digits + ''
blank_index = len(alphabet)

def prepare_input_for_ocr(image_path):
  input_data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
  input_data = cv2.resize(input_data, (200, 31))
  input_data = input_data[np.newaxis]
  input_data = np.expand_dims(input_data, 3)
  input_data = input_data.astype('float32')/255
  return input_data

def predict_for_ocr(image_path, model_path):
  interpreter = tf.lite.Interpreter(model_path=model_path)
  interpreter.allocate_tensors()

  input_data = prepare_input_for_ocr(image_path)

  # Get input and output tensors.
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  interpreter.set_tensor(input_details[0]['index'], input_data)
  interpreter.invoke()

  output = interpreter.get_tensor(output_details[0]['index'])
  return output

def run_ocr(filename):
    image_path = out_dir + filename

    result = predict_for_ocr(image_path, ocr_v2_model)
    text = "".join(alphabet[index] for index in result[0] if index not in [blank_index, -1])

    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Create the path for the corresponding .txt file in the labels folder
    label_path = os.path.join(out_dir + 'gt.txt')
    
    # Open the file for writing the YOLO label
    with open(label_path, 'a') as f:
        # Format the YOLO line: class_id center_x center_y width height (all normalized)
        yolo_format = f"{base_filename}.JPG\t{text}\n"
        
        # Write the line to the file
        f.write(yolo_format)

    print(f'Extracted text:{base_filename}: {text}')

def check_ocr_model(filename, lines, ocr_model):
    image_path = out_dir + filename
    result = predict_for_ocr(image_path, ocr_model)
    text = "".join(alphabet[index] for index in result[0] if index not in [blank_index, -1])
    text = trim_leading_zeros_and_suffix(text)
    if text == "":
        text = '0'
    for line in lines:
        line = line[:-1]
        img_name = line.split("\t")[0]
        allas = line.split("\t")[1]
        if img_name == filename:
            if allas in text:
                global correct_allas
                correct_allas = correct_allas + 1
            else:
                global incorrect_allas
                print('Nem talált: ' + filename + ' | ' + 'ocr: ' + text + ', eredeti: ' + allas)
                incorrect_allas = incorrect_allas + 1


def trim_leading_zeros_and_suffix(input_string):
    # Remove leading zeros
    trimmed_string = input_string.lstrip('0')
    
    # Split at the period and take the part before it
    if '.' in trimmed_string:
        trimmed_string = trimmed_string.split('.')[0]
    
    return trimmed_string

def main():
    label_path = os.path.join(out_dir + 'gt.txt')
    global lines
    global correct_allas
    global incorrect_allas
    with open(label_path, 'r') as f:
        lines = f.readlines()
    for filename in os.listdir(images_dir):
        if filename.endswith(('.JPG')):
            # print('Run labeling on: ', filename)
            run_inference_on_image(filename)
            print("Fut a detektálás")
            # run_ocr(filename)

    for filename in os.listdir(out_dir):
        if filename.endswith(('.JPG')) or filename.endswith(('.png')):
            check_ocr_model(filename, lines, ocr_v2_model)
            print("Fut az OCR")
            
    sum_of_device = segment+digit
    print('Összes felismert mérő:', sum_of_device)
    print('Hétszegmenses mérők száma:', segment)
    print('Tárcsás mérők száma:', digit)
    print(round(correct_allas/(correct_allas+incorrect_allas)*100,2), "%-ban egyezik az OCR eredménye és a valós mérőállás (hétszegmenses kizárólag)")

if __name__=="__main__":
    main()