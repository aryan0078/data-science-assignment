import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
df1 = pd.read_csv('flipkart_com-ecommerce_sample.csv',
                  encoding='unicode_escape')
df2 = pd.read_csv('amz_com-ecommerce_sample.csv',encoding= 'unicode_escape')
st.title("Ecommerce Data Analysis")
st.subheader("Flipkart Data")
st.write(df1)
st.subheader("Amazon Data")
st.write(df2)
st.subheader("Similar Products search [WITHOUT ML]")
st.write("Enter the product name")
product = st.text_input("Product Name")
def match_smiliar_product(df1,df2,product_name):
    df1['product_name'] = df1['product_name'].str.lower()
    df2['product_name'] = df2['product_name'].str.lower()
    product_name = product_name.lower()
    df1['match'] = df1['product_name'].apply(lambda x: 1 if product_name in x else 0)
    df2['match'] = df2['product_name'].apply(lambda x: 1 if product_name in x else 0)
    df1 = df1[df1['match'] == 1]
    df2 = df2[df2['match'] == 1]
    return df1,df2
if st.button("Search"):
    df1,df2 = match_smiliar_product(df1,df2,product)
    df = pd.DataFrame()
    df["Product Name in Flipkart"] = df1["product_name"]
    df["Retail Price in Flipkart"] = df1["retail_price"]
    df["Discounted Price in Flipkart"] = df1["discounted_price"]
    df["Product Name in Amazon"] = df2["product_name"]
    df["Retail Price in Amazon"] = df2["retail_price"]
    df["Discounted Price in Amazon"] = df2["discounted_price"]
    st.write(df)

st.subheader("Similar Products search [WITH ML]")
st.write("Enter the product name")
product = st.text_input("Product Name:")
def tensorflow_search(product_name,df1,df2):
    import tensorflow as tf
    import tensorflow_hub as hub
    df1['product_name'] = df1['product_name'].str.lower()
    df2['product_name'] = df2['product_name'].str.lower()
    product_name = product_name.lower()
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    df1['embeddings'] = df1['product_name'].apply(lambda x: embed([x]).numpy())
    df2['embeddings'] = df2['product_name'].apply(lambda x: embed([x]).numpy())
    product_embedding = embed([product_name]).numpy()
    
    
    def cosine_similarity(v1, v2):
        return np.dot(v1, v2.T) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    df1['similarity'] = df1['embeddings'].apply(lambda x: cosine_similarity(x,product_embedding))
    df2['similarity'] = df2['embeddings'].apply(lambda x: cosine_similarity(x,product_embedding))
    df1 = df1.sort_values(by='similarity',ascending=False)
    df2 = df2.sort_values(by='similarity',ascending=False)
    return df1,df2
if st.button("Search With ML"):
    df1,df2 = tensorflow_search(product,df1,df2)
    df = pd.DataFrame()
    df["Product Name in Flipkart"] = df1["product_name"]
    df["Retail Price in Flipkart"] = df1["retail_price"]
    df["Discounted Price in Flipkart"] = df1["discounted_price"]
    df["Product Name in Amazon"] = df2["product_name"]
    df["Retail Price in Amazon"] = df2["retail_price"]
    df["Discounted Price in Amazon"] = df2["discounted_price"]
    st.write(df)
    

    
    



    




    
    



    










