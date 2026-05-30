import io
import os
import base64
import logging
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("skin-lesion-api")

app = FastAPI(
    title="Skin Lesion Classification & Grad-CAM API",
    description="Biomedical engineering pipeline targeting nested EfficientNetB0 feature maps."
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production requirements
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 7 Class Metadata Order matching model output indices (0-6)
CLASS_KEYS = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
CLASS_DISPLAY_NAMES = {
    "nv": "Melanocytic nevi (benign mole)",
    "mel": "Melanoma (malignant)",
    "bkl": "Benign keratosis",
    "bcc": "Basal cell carcinoma (malignant)",
    "akiec": "Actinic keratosis (pre-malignant)",
    "vasc": "Vascular lesion",
    "df": "Dermatofibroma"
}

# Global model variable
model = None

def get_head_layers(model_instance):
    """
    Finds the outer classification head layers dynamically by name 
    or falls back to layer classes if naming indexes shift during save/load.
    """
    gap, dropout, dense = None, None, None
    try:
        gap = model_instance.get_layer('global_average_pooling2d_3')
    except ValueError:
        pass
    try:
        dropout = model_instance.get_layer('dropout_3')
    except ValueError:
        pass
    try:
        dense = model_instance.get_layer('dense_3')
    except ValueError:
        pass

    # Dynamic fallback scan by layer class
    if not gap or not dense:
        for layer in reversed(model_instance.layers):
            if isinstance(layer, tf.keras.layers.Dense) and not dense:
                dense = layer
            elif isinstance(layer, tf.keras.layers.Dropout) and not dropout:
                dropout = layer
            elif isinstance(layer, tf.keras.layers.GlobalAveragePooling2D) and not gap:
                gap = layer
                
    return gap, dropout, dense

def build_mock_architecture():
    """
    Constructs a structurally identical dummy model matching the target pipeline,
    allowing the application to load and serve mock-inferences for local frontend testing
    in the absence of an actual best_model.keras file.
    """
    logger.info("Initializing mock EfficientNetB0-based structure...")
    base_extractor = tf.keras.applications.EfficientNetB0(
        include_top=False, 
        weights=None, 
        input_shape=(224, 224, 3)
    )
    base_extractor._name = 'efficientnetb0'
    
    inputs = tf.keras.Input(shape=(224, 224, 3), dtype=tf.float32)
    x = base_extractor(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_average_pooling2d_3')(x)
    x = tf.keras.layers.Dropout(0.3, name='dropout_3')(x)
    outputs = tf.keras.layers.Dense(7, activation='softmax', name='dense_3')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

@app.on_event("startup")
def load_keras_model():
    global model
    model_path = "best_model.keras"
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            logger.info(f"Loaded model file successfully from: {model_path}")
        except Exception as e:
            logger.error(f"Error loading {model_path}: {e}. Falling back to mock model.")
            model = build_mock_architecture()
    else:
        logger.warning(f"{model_path} not found. Running mock development model.")
        model = build_mock_architecture()

def preprocess_image(image_bytes: bytes):
    """
    Preprocess image: open with PIL, convert to RGB, resize to 224x224,
    and convert to array keeping values in [0, 255] range as float32.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32)  # Keep 0-255 range
    img_batch = np.expand_dims(img_array, axis=0)
    return img_resized, img_array, img_batch

@app.post("/predict")
async def predict_lesion(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=500, detail="Inference model is currently uninitialized.")
    
    try:
        content = await file.read()
        pil_img, img_array, img_batch = preprocess_image(content)
        
        # 1. Forward Pass to acquire predicted probabilities
        predictions = model(img_batch, training=False)
        probabilities = predictions[0].numpy().tolist()
        pred_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[pred_idx])
        
        # 2. Extract nested structures for Grad-CAM execution
        base_model = model.get_layer('efficientnetb0')
        top_conv_layer = base_model.get_layer('top_conv')
        
        # Sub-model mapping base model inputs -> base conv outputs & base model final activation
        base_sub_model = tf.keras.Model(
            inputs=base_model.inputs,
            outputs=[top_conv_layer.output, base_model.output]
        )
        
        gap_layer, dropout_layer, dense_layer = get_head_layers(model)
        
        # Trace gradients using tf.GradientTape
        with tf.GradientTape() as tape:
            # Trace base model outputs
            conv_outputs, base_outputs = base_sub_model(img_batch, training=False)
            tape.watch(conv_outputs)
            
            # Trace outer head layers
            x = gap_layer(base_outputs)
            if dropout_layer:
                x = dropout_layer(x, training=False)
            head_predictions = dense_layer(x)
            
            # Loss targets the output node of predicted class index
            loss = head_predictions[:, pred_idx]
            
        # Get gradient of prediction with respect to top_conv activation maps
        grads = tape.gradient(loss, conv_outputs)
        
        # Compute pooled gradient map channels
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weighted combination of feature map channels
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Apply ReLU activation and normalize heatmap bounds
        heatmap = tf.maximum(heatmap, 0.0)
        max_val = tf.reduce_max(heatmap)
        if max_val == 0:
            max_val = 1e-8
        heatmap = heatmap / max_val
        heatmap_np = heatmap.numpy()
        
        # 3. Resize heatmap and construct OpenCV superimposition
        heatmap_resized = cv2.resize(heatmap_np, (224, 224))
        heatmap_colored = np.uint8(255 * heatmap_resized)
        
        # Apply JET colormap and convert BGR (CV2 default) -> RGB
        heatmap_colormap = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
        heatmap_colormap_rgb = cv2.cvtColor(heatmap_colormap, cv2.COLOR_BGR2RGB)
        
        # Superimpose: 60% original image structure and 40% activation map overlay
        superimposed = cv2.addWeighted(
            img_array.astype(np.uint8), 
            0.6, 
            heatmap_colormap_rgb, 
            0.4, 
            0
        )
        
        # Save output superimposition to base64 PNG format
        pil_superimposed = Image.fromarray(superimposed)
        buffered = io.BytesIO()
        pil_superimposed.save(buffered, format="PNG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        predicted_key = CLASS_KEYS[pred_idx]
        
        return {
            "predicted_class": predicted_key,
            "display_name": CLASS_DISPLAY_NAMES[predicted_key],
            "confidence": confidence,
            "all_probabilities": probabilities,
            "gradcam_image": f"data:image/png;base64,{encoded_image}"
        }
        
    except Exception as e:
        logger.error(f"Inference pipeline execution error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference pipeline execution failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)