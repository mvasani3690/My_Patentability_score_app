import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

import torch
from pprint import pprint
from datasets import load_dataset

# ----- Data Loading ------
dataset_dict = load_dataset('HUPD/hupd',
    name='sample',
    data_files="https://huggingface.co/datasets/HUPD/hupd/blob/main/hupd_metadata_2022-02-22.feather", 
    icpr_label=None,
    train_filing_start_date='2016-01-01',
    train_filing_end_date='2016-01-21',
    val_filing_start_date='2016-01-22',
    val_filing_end_date='2016-01-31',
)
# Here we can see the `train` and `val` splits, along with the
# location of the cached data files
print('Dataset contents:')
print(dataset_dict)

print('Dataset cache location:')
print(dataset_dict.cache_files)

# Data
train_dataset = dataset_dict["train"]
val_dataset = dataset_dict["validation"]
print(f'Train dataset shape: {train_dataset.shape}')
print(f'Validation dataset shape: {val_dataset.shape}')

# List all available fields
print(f'Dataset fields:')
print(train_dataset.column_names)

# Example: preprocess the abstract field of the dataset
# using HF tokenizers
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('roberta-base')

# We tokenize in batches, so tokenization is quite fast
train_dataset = train_dataset.map(
    lambda e: tokenizer(e['abstract'], truncation=True, padding='max_length'),
    batched=True,
    desc="Tokenizing training files"
)
val_dataset = val_dataset.map(
    lambda e: tokenizer(e['abstract'], truncation=True, padding='max_length'),
    batched=True,
    desc="Tokenizing training files"
)

# Since we've tokenized the dataset, we have a new cache location
print('Dataset cache location after tokenization:')
print(train_dataset.cache_files)

# And we have added some fields to our dataset
print('Dataset fields after tokenization:')
print(train_dataset.column_names)


# Load the BERT tokenizer and model for sequence classification
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased', num_labels=6)

# Function to retrieve abstract and claims text based on filing number
def get_text_data(filing_number):
    # Check if the filing number exists in the dataset
    if filing_number >= len(train_dataset) or filing_number < 0:
        return None, None  # Return None if the filing number is out of range or negative
    
    # Access the data corresponding to the filing number
    data = train_dataset[filing_number]
    
    # Retrieve the abstract and claims text from the data
    abstract = data.get('abstract', None)
    claims = data.get('claims', None)
    
    return abstract, claims



# Streamlit app

st.markdown("Link to app - [Patentabiity app](https://huggingface.co/spaces/mvasani/Patentatbility_score_app)")
def main():
    st.title("Patentability Score App")
    
    # Dropdown menu to select the application filing number
    filing_number = st.selectbox("Select Application Filing Number", range(len(train_dataset)))
    
    # Display abstract and claims text boxes based on selected filing number
    abstract, claims = get_text_data(filing_number)
    st.subheader("Abstract:")
    st.text_area("Abstract Text", abstract, height=200, key='abstract_text')
    st.subheader("Claims:")
    st.text_area("Claims Text", claims, height=400, key='claims_text')
    
    # Submit button to calculate and display the patentability score
    if st.button("Submit"):
        # Tokenize the abstract and claims texts
        inputs = tokenizer(abstract, claims, return_tensors="pt", padding=True, truncation=True)
        
        # Perform inference with the model to get the logits
        with torch.no_grad():
            logits = model(**inputs).logits
        
        # Calculate the patentability score
        score = torch.softmax(logits, dim=1).tolist()[0]
        
        # Display the patentability score
        st.subheader("Patentability Score:")
        st.write("REJECTED:", score[0])
        st.write("ACCEPTED:", score[1])
        st.write("PENDING:", score[2])
        st.write("CONT-REJECTED:", score[3])
        st.write("CONT-ACCEPTED:", score[4])
        st.write("CONT-PENDING:", score[5])

if __name__ == "__main__":
    main()
