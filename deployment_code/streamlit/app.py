import streamlit as st
from transformers import BertTokenizerFast, EncoderDecoderModel
import torch

# BERT model setup for summarization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizerFast.from_pretrained('mrm8488/bert-mini2bert-mini-finetuned-cnn_daily_mail-summarization')
model = EncoderDecoderModel.from_pretrained('mrm8488/bert-mini2bert-mini-finetuned-cnn_daily_mail-summarization').to(device)

# Function to generate summary
def generate_summary(text, temperature=1.0, top_k=None, top_p=None):
    inputs = tokenizer([text], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    
    generation_kwargs = {
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
    }
    
    output = model.generate(input_ids, attention_mask=attention_mask, **generation_kwargs)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Streamlit app main function
def main():
    st.title("Extractive Summarization with MLflow Integration")

    # Display images
    col1, col2 = st.columns(2)
    with col1:
        st.image("teacher_student_image.png", caption="Teacher and Student")  # Adjust file path as needed

    # BERT Summarization Section
    st.subheader("BERT Text Summarization")
    text_input = st.text_area("Enter text here for summarization")

    # Generation parameters
    temperature = st.slider("Select Temperature", 0.1, 2.0, 1.0, 0.1)
    top_k = st.number_input("Select Top K (0 for no limit)", min_value=0, max_value=100, value=0, step=1)
    top_p = st.slider("Select Top P (Nucleus Sampling)", 0.1, 1.0, 0.9, 0.1)

    if st.button("Generate Summary"):
        if text_input:
            summary = generate_summary(text_input, temperature=temperature, top_k=top_k, top_p=top_p)
            st.write("Summary:")
            st.write(summary)
        else:
            st.warning("Please enter some text.")

if __name__ == "__main__":
    main()

