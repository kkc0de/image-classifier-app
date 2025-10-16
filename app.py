import gradio as gr
import tensorflow as tf
import numpy as np

#Load the trained Keras model 
try:
    model = tf.keras.models.load_model('cifar10_cnn_model.h5')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

#the class names ---
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

#prediction function
def predict_image(image):
    if model is None:
        return {"Error": "Model could not be loaded."}
    
    #resize the user's image to 32x32 pixels.
    image_resized = tf.image.resize(image, [32, 32])
    
    #Normalize the pixel values of the resized image.
    image_normalized = np.array(image_resized).astype('float32') / 255.0
    
    #Add the batch dimension.
    image_batch = np.expand_dims(image_normalized, axis=0)
    
    #Make the prediction.
    predictions = model.predict(image_batch)
    
    #Format the output.
    confidences = {class_names[i]: float(predictions[0][i]) for i in range(10)}
    return confidences

#Create the Gradio Interface
iface = gr.Interface(
    fn=predict_image,
    # The input box is now a normal size
    inputs=gr.Image(label="Upload an Image"),
    outputs=gr.Label(num_top_classes=3, label="Predictions"),
    title="🖼️ CIFAR-10 Image Classifier",
    description="Upload any image to see the model's prediction. The image will be automatically resized to 32x32 pixels."
)


iface.launch()