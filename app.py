import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import gdown  # Import gdown for downloading the model from Google Drive

# URL to your Google Drive file (make sure to replace this with your actual file ID)
file_id = '1tNWCFA1hmES1PAX2fOuPYc3ZHlkXDRfX'  # Replace with your file ID
model_url = f'https://drive.google.com/uc?id=1tNWCFA1hmES1PAX2fOuPYc3ZHlkXDRfX'

# Download the model from Google Drive
output_model_path = 'flower_model.h5'
gdown.download(model_url, output_model_path, quiet=False)

# Загрузка модели
model = tf.keras.models.load_model(output_model_path)

# Заголовок приложения
st.title('Классификатор цветов')

# Восстановление label_dict (адаптируйте путь, если нужно)
path = kagglehub.dataset_download("rahmasleam/flowers-dataset")
df = pd.read_csv(path + '/flower_photos.csv')

train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,    
).flow_from_dataframe(
    dataframe=df,
    x_col='file_path',
    y_col='label',
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    shuffle=True,
    seed=42,
    subset='training',
)

label_dict = {v: k for k, v in train_generator.class_indices.items()}

# Загрузка изображения
uploaded_file = st.file_uploader("Выберите изображение цветка", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Преобразование изображения
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255. 

    # Предсказание
    prediction = model.predict(img)
    predicted_class_index = np.argmax(prediction)  # Индекс предсказанного класса
    predicted_class_label = label_dict.get(predicted_class_index, 'Unknown')  # Получение метки класса
    
    # Отображение результата
    st.write(f'Предсказанный класс: {predicted_class_label}')
