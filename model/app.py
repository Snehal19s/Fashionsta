import gradio as gr
import cv2
import tensorflow as tf
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import MobileNetV2

# === Fitzpatrick Skin Tones ===
classes = ['Type_I', 'Type_II', 'Type_III', 'Type_IV', 'Type_V', 'Type_VI']
descriptive_labels = {
    'Type_I': 'Very Fair (Type I)',
    'Type_II': 'Fair (Type II)',
    'Type_III': 'Medium (Type III)',
    'Type_IV': 'Olive / Brown (Type IV)',
    'Type_V': 'Dark Brown (Type V)',
    'Type_VI': 'Deep Dark (Type VI)'
}

# === 30+ Color Suggestions per Tone ===
color_suggestions = {
    'Type_I': ['#FADADD', '#FFE4E1', '#FAF0E6', '#F5DEB3', '#D8BFD8', '#E6E6FA', '#F0FFFF',
               '#FFF0F5', '#F5F5DC', '#FDF5E6', '#F8F8FF', '#FFF5EE', '#FFFAF0', '#FFF8DC',
               '#E0FFFF', '#DCDCDC', '#F0F8FF', '#F5FFFA', '#F5F5F5', '#FAEBD7', '#F0FFF0',
               '#FFFFF0', '#FFEBCD', '#FFE4C4', '#FFFACD', '#FFF5EE', '#F0FFFF', '#FFF0F5',
               '#FAFAD2', '#FFFAFA'],
    'Type_II': ['#FFDAB9', '#EEE8AA', '#FFE4B5', '#FAFAD2', '#F0E68C', '#FFFACD', '#FFD700',
                '#E6BE8A', '#FFDEAD', '#FFF8DC', '#F4A460', '#FFF5EE', '#FFB6C1', '#F5F5DC',
                '#FFEFD5', '#FFEC8B', '#FCEABB', '#FFFAF0', '#F8DE7E', '#F7DC6F', '#FAEBD7',
                '#F5F5F5', '#F0FFF0', '#FFF0F5', '#FFFACD', '#F0FFFF', '#FFF8DC', '#FFFAFA',
                '#FFE4C4', '#FDF5E6'],
    'Type_III': ['#F4A460', '#DEB887', '#D2B48C', '#DAA520', '#BC8F8F', '#CD853F', '#B8860B',
                 '#CDAA7D', '#FFEBCD', '#FFDAB9', '#E9967A', '#FA8072', '#FF7F50', '#FFA07A',
                 '#D2691E', '#FF8C00', '#BDB76B', '#E9967A', '#F0E68C', '#FFA500', '#DAA520',
                 '#EEE8AA', '#CD853F', '#8B4513', '#FF6347', '#E97451', '#FFC1CC', '#C0C0C0',
                 '#B0C4DE', '#EEDC82'],
    'Type_IV': ['#A0522D', '#8B4513', '#A52A2A', '#800000', '#B22222', '#B8860B', '#DAA520',
                '#CD853F', '#D2691E', '#FF8C00', '#E9967A', '#C04000', '#B87333', '#A97142',
                '#DEB887', '#C19A6B', '#BC8F8F', '#F4A460', '#FFE4B5', '#FFA07A', '#FF4500',
                '#BDB76B', '#EEE8AA', '#F5DEB3', '#FF7F50', '#E67E22', '#D35400', '#884400',
                '#996515', '#BA8759'],
    'Type_V': ['#5C4033', '#8B4513', '#654321', '#4B3621', '#704214', '#3D2B1F', '#362511',
               '#4E342E', '#6F4E37', '#5D3A00', '#2F1B0C', '#3E2723', '#8B5F4D', '#734F2C',
               '#5A381E', '#7B3F00', '#53350A', '#4B3621', '#5C3317', '#4B2E2E', '#7C4848',
               '#8B4513', '#593C1F', '#6B4226', '#4D2600', '#663300', '#5D3A00', '#7B3F00',
               '#8B5F4D', '#6F4E37'],
    'Type_VI': ['#3B2F2F', '#2C1608', '#1C1C1C', '#3E2723', '#2F1B0C', '#1A1A1A', '#0D0D0D',
                '#1F1B24', '#1B1B1B', '#212121', '#141414', '#000000', '#2B2B2B', '#242124',
                '#1A0D0D', '#120D0D', '#0F0F0F', '#1C1C1C', '#231F20', '#2D2B2B', '#292421',
                '#343434', '#262626', '#393939', '#282828', '#1C1C1C', '#111111', '#2A2A2A',
                '#151515', '#100C08']
}

# Load MTCNN
mtcnn = MTCNN()

# Build dummy model (you should load trained weights if you have them)
def build_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=Input(shape=(120, 90, 3)))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(len(classes), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

model = build_model()

# Skin tone prediction function
def predict_skin_tone(image):
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = mtcnn.detect_faces(image_rgb)

        if not faces:
            return "❌ No face detected.", None

        largest_face = max(faces, key=lambda f: f['box'][2] * f['box'][3])
        x, y, w, h = largest_face['box']
        x, y = max(x, 0), max(y, 0)
        face_img = image_rgb[y:y+h, x:x+w]

        face_resized = cv2.resize(face_img, (90, 120))
        input_tensor = preprocess_input(face_resized[np.newaxis, ...])

        predictions = model.predict(input_tensor)
        idx = np.argmax(predictions)
        predicted_class = classes[idx]
        readable = descriptive_labels[predicted_class]
        color_list = color_suggestions[predicted_class]

        color_html = "".join(
            f"<div style='background:{c};width:40px;height:25px;display:inline-block;margin:2px;border-radius:4px;'></div>"
            for c in color_list
        )

        return f"✅ Predicted Skin Tone: {readable}", color_html
    except Exception as e:
        return f"❌ Error: {str(e)}", None

# Gradio interface
interface = gr.Interface(
    fn=predict_skin_tone,
    inputs=gr.Image(type="numpy", label="Upload Face Image"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.HTML(label="Suggested Colors")
    ],

    description="Upload a clear face photo to detect your skin tone (Type I–VI) and get 30+ tailored color recommendations."
)

interface.launch()