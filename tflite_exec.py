import tflite_runtime.interpreter as tflite

# load the TFLite model and allocate tensors
interpreter = tflite.Interpreter(model_path='fullint.tflite')
interpreter.allocate_tensors()

# get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# test the model on random input data
input_shape = input_details[0]['shape']
interpreter.set_tensor(input_details[0]['index'], input_data)

# run them model
interpreter.invoke()

# get_tensor() returns copy of the tensor data
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)