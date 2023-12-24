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






# import streamlit as st
# from transformers import BertTokenizerFast, EncoderDecoderModel
# import torch
# import mlflow.pyfunc
# import os

# # Import the logging function from your MLflow script
# # Replace 'your_mlflow_script' with the actual name of your MLflow script
# from your_mlflow_script import log_user_hyperparameters

# # BERT model setup for summarization
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# tokenizer = BertTokenizerFast.from_pretrained('mrm8488/bert-mini2bert-mini-finetuned-cnn_daily_mail-summarization')
# model = EncoderDecoderModel.from_pretrained('mrm8488/bert-mini2bert-mini-finetuned-cnn_daily_mail-summarization').to(device)

# # Function to generate summary
# def generate_summary(text, temperature=1.0, top_k=None, top_p=None):
#     inputs = tokenizer([text], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
#     input_ids = inputs.input_ids.to(device)
#     attention_mask = inputs.attention_mask.to(device)
    
#     generation_kwargs = {
#         "temperature": temperature,
#         "top_k": top_k,
#         "top_p": top_p,
#     }
    
#     output = model.generate(input_ids, attention_mask=attention_mask, **generation_kwargs)
#     return tokenizer.decode(output[0], skip_special_tokens=True)

# # Streamlit app main function
# def main():
#     st.title("Extractive Summarization with MLflow Integration")

#     # Display images
#     col1, col2 = st.beta_columns(2)
#     with col1:
#         st.image("teacher_student_image.png", caption="Teacher and Student")  # Adjust file path as needed

#     # BERT Summarization Section
#     st.subheader("BERT Text Summarization")
#     text_input = st.text_area("Enter text here for summarization")

#     # Generation parameters
#     temperature = st.slider("Select Temperature", 0.1, 2.0, 1.0, 0.1)
#     top_k = st.number_input("Select Top K (0 for no limit)", min_value=0, max_value=100, value=0, step=1)
#     top_p = st.slider("Select Top P (Nucleus Sampling)", 0.1, 1.0, 0.9, 0.1)

#     if st.button("Generate Summary"):
#         if text_input:
#             summary = generate_summary(text_input, temperature=temperature, top_k=top_k, top_p=top_p)
#             st.write("Summary:")
#             st.write(summary)

#             # Log the hyperparameters to MLflow
#             log_user_hyperparameters(temperature, top_k, top_p)
#         else:
#             st.warning("Please enter some text.")

# if __name__ == "__main__":
#     main()






# import streamlit as st
# import mlflow.pyfunc
# from sklearn import datasets
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os

# # MLflow server URI
# MLFLOW_SERVER_URI = "http://mlflow-tracking-server.default.svc.cluster.local:5000"
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./cs505.json"

# # Function to load model from MLflow
# def load_model_from_mlflow(model_name="demo_model", stage="Production"):
#     mlflow.set_tracking_uri(MLFLOW_SERVER_URI)
#     model_uri = f"models:/{model_name}/{stage}"
#     return mlflow.pyfunc.load_model(model_uri)

# # Load your initial model from MLflow
# model = load_model_from_mlflow()

# # Load Iris Data
# def load_iris_data():
#     iris = datasets.load_iris()
#     data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
#     data['target'] = iris.target
#     return data

# # Streamlit app main function
# def main():
#     st.title("MLFlow Demo with Iris Dataset")

#     # Model reloading section
#     st.subheader("Model Reloading")
#     if st.button("Reload Model from MLflow"):
#         global model
#         model = load_model_from_mlflow()  # Reload the model from MLflow
#         st.success("Model reloaded from MLflow!")

#     # Data section
#     data = load_iris_data()
#     if st.checkbox("Show dataset"):
#         st.write(data)

#     # Model testing section
#     test_size = st.slider("Test Size for Splitting Data", min_value=0.1, max_value=0.9, value=0.3, step=0.1)
#     if st.button("Test Model"):
#         if not data.empty:
#             X = data.iloc[:, :-1]
#             y = data['target']
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
#             predictions = model.predict(X_test)

#             accuracy = accuracy_score(y_test, predictions)
#             report = classification_report(y_test, predictions)

#             st.write("Accuracy:", accuracy)
#             st.text(report)

#             # Plotting
#             fig, ax = plt.subplots()
#             sns.scatterplot(x=X_test.iloc[:, 0], y=X_test.iloc[:, 1], hue=y_test, style=predictions, ax=ax)
#             plt.xlabel(data.columns[0])
#             plt.ylabel(data.columns[1])
#             st.pyplot(fig)
#         else:
#             st.error("Data is not loaded. Please load the data to test the model.")

# if __name__ == "__main__":
#     main()


# import streamlit as st
# from transformers import BertTokenizerFast, EncoderDecoderModel
# import torch
# import mlflow.pyfunc
# import os

# # BERT model setup for summarization
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# tokenizer = BertTokenizerFast.from_pretrained('mrm8488/bert-mini2bert-mini-finetuned-cnn_daily_mail-summarization')
# model = EncoderDecoderModel.from_pretrained('mrm8488/bert-mini2bert-mini-finetuned-cnn_daily_mail-summarization').to(device)

# # Function to generate summary
# def generate_summary(text):
#     inputs = tokenizer([text], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
#     input_ids = inputs.input_ids.to(device)
#     attention_mask = inputs.attention_mask.to(device)
#     output = model.generate(input_ids, attention_mask=attention_mask)
#     return tokenizer.decode(output[0], skip_special_tokens=True)

# # MLflow setup
# MLFLOW_SERVER_URI = "http://mlflow-tracking-server.default.svc.cluster.local:5000"
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./cs505.json"

# def load_model_from_mlflow(model_name="demo_model", stage="Production"):
#     mlflow.set_tracking_uri(MLFLOW_SERVER_URI)
#     model_uri = f"models:/{model_name}/{stage}"
#     return mlflow.pyfunc.load_model(model_uri)

# # Load your initial model from MLflow
# # Assuming the MLflow model is different from the BERT summarization model
# mlflow_model = load_model_from_mlflow()

# # Streamlit app main function
# def main():
#     st.title("BERT Summarization with MLflow Integration")

#     # BERT Summarization Section
#     st.subheader("BERT Text Summarization")
#     text_input = st.text_area("Enter text here for summarization")
#     if st.button("Generate Summary"):
#         if text_input:
#             summary = generate_summary(text_input)
#             st.write("Summary:")
#             st.write(summary)
#         else:
#             st.warning("Please enter some text.")

#     # Model reloading section (for MLflow model, if needed)
#     st.subheader("Reload MLflow Model")
#     if st.button("Reload MLflow Model"):
#         global mlflow_model
#         mlflow_model = load_model_from_mlflow()  # Reload the MLflow model
#         st.success("MLflow model reloaded successfully!")



# if __name__ == "__main__":
#     main()



# import streamlit as st
# from transformers import BertTokenizerFast, EncoderDecoderModel
# import torch
# import mlflow.pyfunc
# import os

# # Import the logging function from your MLflow script
# # Replace 'your_mlflow_script' with the actual name of your MLflow script
# from your_mlflow_script import log_user_hyperparameters

# # BERT model setup for summarization
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# tokenizer = BertTokenizerFast.from_pretrained('mrm8488/bert-mini2bert-mini-finetuned-cnn_daily_mail-summarization')
# model = EncoderDecoderModel.from_pretrained('mrm8488/bert-mini2bert-mini-finetuned-cnn_daily_mail-summarization').to(device)

# # Function to generate summary
# def generate_summary(text, temperature=1.0, top_k=None, top_p=None):
#     inputs = tokenizer([text], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
#     input_ids = inputs.input_ids.to(device)
#     attention_mask = inputs.attention_mask.to(device)
    
#     generation_kwargs = {
#         "temperature": temperature,
#         "top_k": top_k,
#         "top_p": top_p,
#     }
    
#     output = model.generate(input_ids, attention_mask=attention_mask, **generation_kwargs)
#     return tokenizer.decode(output[0], skip_special_tokens=True)

# # Streamlit app main function
# def main():
#     st.title("Extractive Summarization with MLflow Integration")

#     # Display images
#     col1, col2 = st.beta_columns(2)
#     with col1:
#         st.image("teacher_student_image.png", caption="Teacher and Student")  # Adjust file path as needed

#     # BERT Summarization Section
#     st.subheader("BERT Text Summarization")
#     text_input = st.text_area("Enter text here for summarization")

#     # Generation parameters
#     temperature = st.slider("Select Temperature", 0.1, 2.0, 1.0, 0.1)
#     top_k = st.number_input("Select Top K (0 for no limit)", min_value=0, max_value=100, value=0, step=1)
#     top_p = st.slider("Select Top P (Nucleus Sampling)", 0.1, 1.0, 0.9, 0.1)

#     if st.button("Generate Summary"):
#         if text_input:
#             summary = generate_summary(text_input, temperature=temperature, top_k=top_k, top_p=top_p)
#             st.write("Summary:")
#             st.write(summary)

#             # Log the hyperparameters to MLflow
#             log_user_hyperparameters(temperature, top_k, top_p)
#         else:
#             st.warning("Please enter some text.")

# if __name__ == "__main__":
#     main()


    

