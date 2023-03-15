import streamlit as st
from transformers import pipeline
from PIL import Image
import io

from transformers import MobileViTFeatureExtractor, MobileViTForImageClassification

def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None
    
def load_model():
    model = MobileViTForImageClassification.from_pretrained("apple/mobilevit-small")
    return model
    
def load_feature_extractor():
    feature_extractor = MobileViTFeatureExtractor.from_pretrained("apple/mobilevit-small")
    return feature_extractor

def main():
    st.title('Image Classification')
    image = load_image()
    print(image)
    model = load_model()
    if image is not None:
        feature_extractor = load_feature_extractor()
        
        inputs = feature_extractor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # model predicts one of the 1000 ImageNet classes
        predicted_class_idx = logits.argmax(-1).item()
        st.metric("Predicted class:", model.config.id2label[predicted_class_idx])

if __name__ == "_main_":
    main()
