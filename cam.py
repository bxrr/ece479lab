import cv2
import picamera
import io
import numpy as np
from mtcnn_cv2 import MTCNN
import tflite_runtime.interpreter as tflite

def capture_image():
    # Instrctor note: this can be directly taken from the PiCamera documentation
    # Create the in-memory stream
    stream = io.BytesIO()
    with picamera.PiCamera() as camera:
        camera.capture(stream, format='jpeg')
        
    # Construct a numpy array from the stream
    data = np.frombuffer(stream.getvalue(), dtype=np.uint8)
    
    # "Decode" the image from the array, preserving colour
    image = cv2.imdecode(data, 1)
    
    # OpenCV returns an array with data in BGR order. 
    # The following code invert the order of the last dimension.
    image = image[:, :, ::-1]
    return image

def detect_and_crop(mtcnn, image):
    detection = mtcnn.detect_faces(image)[0]['box']
    #TODO
    bounding_box = detection
    detection[0] = int(detection[0]-0.1*detection[2])
    detection[1] = int(detection[1]-0.1*detection[3])
    detection[2] = int(detection[2]*1.2)
    detection[3] = int(detection[3]*1.2)
    print(detection)
    image = image[detection[0]:detection[0]+detection[2], detection[1]:detection[1]+detection[3]]
    return image, bounding_box
    
    
def show_bounding_box(image, bounding_box):
    image = np.array(image)
    cv2.rectangle(image, (bounding_box[0], bounding_box[1]), (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]), (255, 0, 0), 3)
    return image

# preprocessing function provided to the students
def pre_process(face, required_size=(160, 160)):
    ret = cv2.resize(face, required_size)
    ret = ret.astype('float32')
    mean, std = ret.mean(), ret.std()
    ret = (ret - mean) / std
    return ret

def run_model(model, face):
# students will need to fill in the following function
    #TODO
    output_details = model.get_output_details()
    input_details = model.get_input_details()
    model.set_tensor(input_details[0]['index'], face)
    model.invoke()
    output_data = model.get_tensor(output_details[0]['index'])
    return output_data


tfl_file = "./resnet.tflite"
interpreter = tflite.Interpreter(model_path=tfl_file)
interpreter.allocate_tensors()

# process the image of the first person
input()
# 1. Read the image
mtcnn = MTCNN()
image = capture_image()
image = cv2.imread('brandon2.png')
# 2. Detect and Crop
cropped_image, frame = detect_and_crop(mtcnn, image)
# cv2.imwrite("brandon2.png", image)
# cv2.imwrite("brandon_box2.png", show_bounding_box(image, frame))
# cv2.imwrite("brandon_crop2.png", cropped_image)
# 3. Proprocess
processed_image = pre_process(cropped_image)
# 4. Run the model
print(processed_image.shape)
processed_image = np.resize(processed_image, (1, 160, 160, 3))
data1 = run_model(interpreter, processed_image)

# process the image of the second person
# 1. Read the image
mtcnn = MTCNN()
# image = cv2.imread(
image = cv2.imread('mateus1.png')
# 2. Detect and Crop
cropped_image, frame = detect_and_crop(mtcnn, image)
# cv2.imwrite("Img2.png", show_bounding_box(image, frame))
# cv2.imwrite("Img2crop.png", cropped_image)
# 3. Proprocess
processed_image = pre_process(cropped_image)
# 4. Run the model
print(processed_image.shape)
processed_image = np.resize(processed_image, (1, 160, 160, 3))
data2 = run_model(interpreter, processed_image)

# Do the comparison of the distance
data1 = np.array(data1)
data2 = np.array(data2)
comp = np.power(data1 - data2, 2)
dist = np.sqrt(np.sum(comp))
print("distance:", dist)