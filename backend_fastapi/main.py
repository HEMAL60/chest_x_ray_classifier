# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 00:28:47 2023

@author: hemal nakrani
"""
import numpy as np
from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
from tensorflow.keras.utils import custom_object_scope
from transformers import  TFViTModel
import uvicorn
from io import BytesIO
from PIL import Image
import logging
import cv2
from fastapi.responses import JSONResponse

logging.basicConfig(filename="backend_app.log", level=logging.INFO)  # Log to a file named "streamlit_app.log"
logger = logging.getLogger(__name__)



custom_objects = {'TFViTMainLayer': TFViTModel}
with custom_object_scope(custom_objects):
    VIT_MODEL = tf.keras.models.load_model('/chest_x_ray_classifier/backend_fastapi/saved_models/vit_model.h5')

RESNET_MODEL = tf.keras.models.load_model('/chest_x_ray_classifier/backend_fastapi/saved_models/resnet_model.h5')
VGG_MODEL = tf.keras.models.load_model('/chest_x_ray_classifier/backend_fastapi/saved_models/vgg19_model.h5')
XCEPTION_MODEL = tf.keras.models.load_model('/chest_x_ray_classifier/backend_fastapi/saved_models/xception_model.h5')
META_MODEL = tf.keras.models.load_model('/chest_x_ray_classifier/backend_fastapi/saved_models/meta_model.h5')

logger.info("all models loaded successfully")
app = FastAPI()

@app.get('/')
async def ping():
    return 'hello world'

def sparse_binary_convertor(pred,threshold=0.28):
    return (pred>=threshold).astype(int)



def get_labels_from_output(output):
    label_order = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']

    if len(output[0]) != len(label_order):
        raise ValueError("Mismatch between output length and label_order length.")
    
    labels = [label_order[i] for i, val in enumerate(output[0]) if val == 1]
    
    if not labels:
        return ["No Findings"]
    return labels

def apply_clahe_to_image(image: np.ndarray) -> np.ndarray:
    
    # Convert the image from RGB to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Split the LAB image into L, A, and B channels
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Define the CLAHE filter
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    # Apply CLAHE to the L-channel
    cl = clahe.apply(l_channel)
    
    # Merge the CLAHE-enhanced L-channel back with the A and B channels
    merged_channels = cv2.merge((cl, a_channel, b_channel))

    # Convert the LAB image back to RGB
    clahe_image = cv2.cvtColor(merged_channels, cv2.COLOR_LAB2RGB)

    return clahe_image


def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")  # Convert image to RGB
    logger.info(f"image shape after reading---->{np.array(image).shape}")
    
    # Convert PIL Image to numpy array and apply CLAHE
    image_np = np.array(image)
    image_np = apply_clahe_to_image(image_np)
    image = Image.fromarray(image_np)
    
    image = image.resize((256, 256))
    logger.info(f"image shape after reading and resizing---->{np.array(image).shape}")
    return np.array(image)

@app.post('/predict')
async def predict(
    file: UploadFile = File(...)
):
    try:
        image = read_file_as_image(await file.read())
        #image = image.resize((256,256))
        image_batch = np.expand_dims(image,0)
        # Define a dictionary of custom layers you want to register
        #batch_shape = image_batch.shape
        logger.info(f'image batch---->{image_batch.shape}')

    # Wrap the loading code in custom_object_scope to register the custom layers
        with custom_object_scope(custom_objects):
            pred1 = VIT_MODEL.predict(image_batch)
        
        pred2 = RESNET_MODEL.predict(image_batch)
        pred3 = VGG_MODEL.predict(image_batch)
        pred4 = XCEPTION_MODEL.predict(image_batch)

        stacked_predictions = np.hstack((pred1, pred2, pred3, pred4))
        print(f'stacked pred len--->{len(stacked_predictions)}')

        meta_model_prediction = META_MODEL.predict(stacked_predictions)
        logger.info(f"meta model predictions no list--->{meta_model_prediction}")
        meta_model_prediction_list = np.squeeze(meta_model_prediction.tolist())
        logger.info(f"meta model prediction in list--->{meta_model_prediction_list}")

        logger.info(f"predicted class prob--->{meta_model_prediction}")
        final_prediction = sparse_binary_convertor(meta_model_prediction)
        #logger.info(f"predicted class--->{final_prediction}")
        predicted_class = get_labels_from_output(final_prediction)
        print(f'predicted class---->{predicted_class}')
        #confidence = np.max(predictions[0])

        label_order = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
        
        mapped_predictions = dict(zip(label_order, meta_model_prediction_list))
        logger.info(f"probability dictionary found --->{mapped_predictions}")

        response = {
        'class': predicted_class,
        'predictions_with_prob': mapped_predictions
        }

        return JSONResponse(content=response)
    except Exception as e:
        logger.error(f"Error during API request: {str(e)}")
        return str(e)

    
if __name__ == '__main__':
   uvicorn.run(app,host = '0.0.0.0' ,port = 5000, log_level="info")