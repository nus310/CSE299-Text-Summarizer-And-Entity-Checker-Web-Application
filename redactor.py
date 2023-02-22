import streamlit as st
import os
import nltk
#NLP Pkgs
import spacy
from spacy import displacy
from nltk.tokenize import sent_tokenize, word_tokenize
nlp = spacy.load('en_core_web_sm')

# -------------------Functions to Sanitize and Redact -------------------------
def sanitize_names(text):
    docx = nlp(text)
    redacted_sentences = []
    with docx.retokenize() as retokenizer:
        for ent in docx.ents:
            retokenizer.merge(ent)   #finding out entities from docx     
        for ent in docx: #finding out token from docx
            if ent.ent_type_ == 'PERSON': 
                redacted_sentences.append('[REDACTED NAME]')
        else:
            redacted_sentences.append(tokenize.string)
    return "".join(redacted_sentences)

    






#--------- UI DESIGN BY streamlit---------------------
def main():

    st.title("Document Redactor App")  # works as print for a title
    st.text("Built with Streamlit and SpaCy")

    activities = ['Redaction', 'Downloads', 'About']
    choice = st.sidebar.selectbox("Select Choice", activities)

    # if st.button("sub"):
    # 	st.write("hello")

    if choice == 'Redaction':
        st.subheader("Redaction of Terms")
        rawtext = st.text_area("Enter Text","Type Here")
        redaction_item = ["names","places","org","date"]
        redaction_choice = st.selectbox("Select Item to Censor",redaction_item)
        save_option = st.radio("Save to File",("Yes","No"))
        if st.button("Submit"):
            result = sanitize_names(rawtext)
            st.write(result)
            
    		
        
        
        
        

    elif choice == 'Downloads':
        st.subheader("Downloads of Terms")

    elif choice == 'About':
        st.subheader("About of Terms")


if __name__ == "__main__":
    main()
