import streamlit as st
from transformers import pipeline

model_checkpoint = "Modfiededition/t5-base-fine-tuned-on-jfleg"
model = pipeline("text2text-generation", model=model_checkpoint)

def main():
    st.title("Grammar Correction")
    input = st.text_input("Enter the sentence")
    output = model(input)
    st.write(output[0]['generated_text'])
    
if __name__ == '__main__':
    main()