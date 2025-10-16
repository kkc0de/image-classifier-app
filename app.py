import gradio as gr
import tensorflow as tf
import numpy as np

#Loading the trained Keras model
try:
    model = tf.keras.models.load_model('cifar10_cnn_model.h5')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

#the class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

#the prediction function
def predict_image(image):
    if model is None:
        return {"Error": "Model could not be loaded."}
    
    image_resized = tf.image.resize(image, [32, 32])
    image_normalized = np.array(image_resized).astype('float32') / 255.0
    image_batch = np.expand_dims(image_normalized, axis=0)
    predictions = model.predict(image_batch)
    confidences = {class_names[i]: float(predictions[0][i]) for i in range(10)}
    return confidences

#Create a Custom Theme 

my_theme = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.gray,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Roboto")
).set(
    # Further customize specific elements
    body_background_fill="*neutral_900",
    button_primary_background_fill="*primary_600",
    button_primary_text_color="white",
)

#Gradio Interface with the custom theme 
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(label="Upload an Image"),
    outputs=gr.Label(num_top_classes=3, label="Predictions"),
    title="🖼️ CIFAR-10 Image Classifier",
    description="Upload any image to see the model's prediction.",
    theme=my_theme  # <-- APPLY YOUR CUSTOM THEME HERE
)


iface.launch()