import tflite_runtime.interpreter as tflite

fashion_mnist = tflite.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images /= 255.0
test_images /= 255.0

# load the TFLite model and allocate tensors
interpreter = tflite.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# test the model on random input data
input_shape = input_details[0]['shape']
interpreter.set_tensor(input_details[0]['index'], test_images)

# run them model
interpreter.invoke()

# get_tensor() returns copy of the tensor data
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)