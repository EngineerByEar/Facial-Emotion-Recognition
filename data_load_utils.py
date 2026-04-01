from PIL import Image
import os
import pandas as pd
import numpy as np

def load_image(image_path):
    img = Image.open(image_path)
    img = np.array(img)
    return img

def load_data(data_directory):
    categories = os.listdir(data_directory)
    emotions = []
    images = []
    for emotion in categories:
        emotion_folder_path = os.path.join(data_directory, emotion)
        image_names = os.listdir(emotion_folder_path)
        for image_name in image_names:
            image_path = os.path.join(emotion_folder_path, image_name)
            img = load_image(image_path)
            images.append(img)
            emotions.append(emotion)
    return emotions, images

def create_data_frame(emotions, labels, images):
    df = pd.DataFrame(columns = ["emotion", "label", "image"])
    df["emotion"] = emotions
    df["label"] = labels
    df["image"] = images
    return df