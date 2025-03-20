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
    detection = mtcnn.detect_faces(image)[0]
    #TODO
    x,y,w,h = detection['box']
    box = image[x:x+w,y:y+h]
    return box
    #xs, ys = image.shape
    #new_x = int(xs*1.2)
    #new_y = int(ys*1.2)
    #cropped_image = np.ones(new_x, new_y)
    #cropped_image[new_x-xs, new_y-ys] = box
    #return cropped_image

def show_bounding_box(image, bounding_box):
    x1, y1, w, h = bounding_box
    fig, ax = plt.subplots(1,1)
    ax.imshow(image)
    ax.add_patch(Rectangle((x1, y1), w, h, linewidth=1, edgecolor='r', facecolor='none'))
    plt.show()
    return

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
# 2. Detect and Crop
cropped_image = detect_and_crop(mtcnn, image)
# 3. Proprocess
processed_image = pre_process(cropped_image)
# 4. Run the model
print(processed_image.shape)
processed_image = np.resize(processed_image, (1, 160, 160, 3))
data1 = run_model(interpreter, processed_image)
print(data1)

# process the image of the second person
input()
# 1. Read the image
mtcnn = MTCNN()
image = capture_image()
# 2. Detect and Crop
cropped_image = detect_and_crop(mtcnn, image)
# 3. Proprocess
processed_image = pre_process(cropped_image)
# 4. Run the model
print(processed_image.shape)
processed_image = np.resize(processed_image, (1, 160, 160, 3))
data2 = run_model(interpreter, processed_image)
print(data2)

# Do the comparison of the distance