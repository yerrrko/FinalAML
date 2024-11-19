import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image

# Загрузка модели
MODEL_PATH = 'flower_model2.h5'

@st.cache_resource
def load_flower_model():
    return load_model(MODEL_PATH)

model = load_flower_model()

# Названия классов (замените на ваши классы, если они другие)
CLASS_NAMES = ['Daisy', 'Dandelion', 'Roses', 'Sunflowers', 'Tulips']

# Функция для предсказания
def predict_flower(image):
    image = image.resize((224, 224))  # Измените размер в зависимости от вашей модели
    img_array = img_to_array(image) / 255.0  # Нормализация
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions)
    return predicted_class, confidence

# Интерфейс Streamlit
st.title("Flower Classification App 🌸")
st.write("Загрузите изображение цветка, чтобы узнать его класс!")

# Загрузка изображения
uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Загруженное изображение", use_column_width=True)
    st.write("Предсказание...")
    
    # Предсказание
    predicted_class, confidence = predict_flower(image)
    st.write(f"Класс: **{predicted_class}**")
    st.write(f"Уверенность: **{confidence:.2f}**")

# Пример использования
st.write("Исходный код и модель находятся в этом репозитории GitHub.")
st.markdown("[Перейти на GitHub](https://github.com/ваш-репозиторий)")

