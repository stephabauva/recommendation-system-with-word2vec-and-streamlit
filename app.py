

import streamlit as st

import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from gensim.models import Word2Vec 
#import matplotlib.pyplot as plt

import pickle
from sklearn.externals import joblib

from google_images_search import GoogleImagesSearch
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#importing google_images_download module 
from google_images_download import google_images_download

#create object
response = google_images_download.googleimagesdownload()

#to resize images
from PIL import Image
from resizeimage import resizeimage
import time

os.chdir('/Users/Stef/Documents/simplon/streamlit/recommendation_system')

st.title('Recommendation system app')

data = ('/Users/Stef/Documents/simplon/streamlit/recommendation_system/df2.csv')
"THE DATABASE"

@st.cache
def load_data(nrows):
    df = pd.read_csv(data,nrows=nrows)
    return df
df = load_data(5000)

st.dataframe(df)

######################################
# # remove missing values
# df.dropna(inplace=True)

# # shuffle customer ID's
# random.shuffle(customers)

# # extract 90% of customer ID's
# customers_train = [customers[i] for i in range(round(0.9*len(customers)))]

# # split data into train and validation set
# train_df = df[df['CustomerID'].isin(customers_train)]
# validation_df = df[~df['CustomerID'].isin(customers_train)]

# # list to capture purchase history of the customers
# purchases_train = []

# # populate the list with the product codes
# for i in tqdm(customers_train):
#     temp = train_df[train_df["CustomerID"] == i]["StockCode"].tolist()
#     purchases_train.append(temp)

# # list to capture purchase history of the customers
# purchases_val = []

# # populate the list with the product codes
# for i in tqdm(validation_df['CustomerID'].unique()):
#     temp = validation_df[validation_df["CustomerID"] == i]["StockCode"].tolist()
#     purchases_val.append(temp)

# # train word2vec model
# model = Word2Vec(window = 10, sg = 1, hs = 0,
#                  negative = 10, # for negative sampling
#                  alpha=0.03, min_alpha=0.0007,
#                  seed = 14)

# model.build_vocab(purchases_train, progress_per=200)

# model.train(purchases_train, total_examples = model.corpus_count, 
#             epochs=10, report_delay=1)

# # save word2vec model
# model.save("word2vec_2.model")

###################################################################
#load pre_trained model
open_model = open('/Users/Stef/Documents/simplon/streamlit/recommendation_system/Rec_sys_model.pkl','rb')
model = joblib.load(open_model)

#As we do not plan to train the model any further, we are calling init_sims(), which will make the model much more memory-efficient.
model.init_sims(replace=True)

# extract all vectors
X = model[model.wv.vocab]

###UMAP

#create a product-ID and product-description dictionary to easily map a product's description to its ID and vice versa
train_df = pd.read_csv("/Users/Stef/Documents/simplon/streamlit/recommendation_system/train_df.csv")

products = train_df[["StockCode", "Description"]]

# remove duplicates
products.drop_duplicates(inplace=True, subset='StockCode', keep="last")

# create product-ID and product-description dictionary
products_dict = products.groupby('StockCode')['Description'].apply(list).to_dict()

#generate with one product's vector as input
def similar_products(v, n = 6):
    
    # extract most similar products for the input vector
    ms = model.similar_by_vector(v, topn= n+1)[1:]
    
    # extract name and similarity score of the similar products
    new_ms = []
    for j in ms:
        pair = (products_dict[j[0]][0], round(j[1],2))
        new_ms.append(pair)
        
    return new_ms 

#checkbox to execute code
checkbox_1 = st.sidebar.checkbox('Execute recommendation based on a single item')

#add a selectbox
option = st.sidebar.selectbox(
    'Select a product ID',
     df['StockCode'])

product_name = df.loc[df['StockCode'] == option, 'Description'].tolist()[0]

st.sidebar.text(product_name)

#check if product folder aready exist, else download image
def item(product_name):
    arguments = {"keywords":product_name,"limit":1,"print_urls":True} #creating list of arguments
    os.chdir("downloads")
    if os.path.isdir(product_name) :
        os.chdir(product_name)
        image = os.listdir()
        img=mpimg.imread(image[0]) 
        imgplot = plt.imshow(img)
        plt.axis('off')
        st.sidebar.pyplot()
        os.chdir("../..")
        
       
    else : 
        os.chdir("..")
        paths = response.download(arguments)   #passing the arguments to the function
        os.chdir("downloads")
        os.chdir(product_name)
        image = os.listdir()
        os.rename(image[0], product_name+'.jpg')
        image = os.listdir()
        img=mpimg.imread(image[0])
        imgplot = plt.imshow(img)
        plt.axis('off')
        st.sidebar.pyplot()
        os.chdir("../..")
         

def reco(product_name):
    arguments = {"keywords":product_name,"limit":1,"print_urls":True} #creating list of arguments
    os.chdir("downloads")
    if os.path.isdir(product_name) :
        os.chdir(product_name)
        image = os.listdir()
        img=mpimg.imread(image[0]) #normally image[0] but jupyter add a .ipynb_checkpoints in 1st position, put image[0] for streamlit
        list_img.append(img)
        os.chdir("../..")
       
    else : 
        os.chdir("..")
        paths = response.download(arguments)   #passing the arguments to the function
        os.chdir("downloads")
        os.chdir(product_name)
        image = os.listdir()
        #os.rename(image[0], product_name+'.jpg')
        image = os.listdir()
        img=mpimg.imread(image[0])
        list_img.append(img)
        os.chdir("../..")
    return list_img

item(product_name)

if checkbox_1 == True:
    st.write("YOUR RECOMMENDATIONS")
    prediction = similar_products(model[option])
    prediction2 = pd.DataFrame(similar_products(model[option]), columns = ["Product Name","% similarity"])
    st.table(prediction2)
    
    #create en empty list to store all images
    list_img = []
    for i in prediction:
        reco(str(i[0]))

    #plot images
    j=0
    fig = plt.figure()
    fig.set_size_inches(30, 20)
    for pic in list_img:
        ax = fig.add_subplot(2,3,j+1)
        plt.axis('off')
        plt.grid(b=None)
        ax.imshow(pic)
        plt.title(j, fontsize=100, y=1)
        j+=1
    st.pyplot()


#checkbox to execute code
checkbox_2 = st.sidebar.checkbox("Execute recommendation based on all customer's purchases")

# extract 90% of customer ID's
customers = df["CustomerID"].unique().tolist()
# shuffle customer ID's
random.shuffle(customers)
# extract 90% of customer ID's
customers_train = [customers[i] for i in range(round(0.9*len(customers)))]

#validation set
validation_df = df[~df['CustomerID'].isin(customers_train)]

# list to capture purchase history of the customers
purchases_val = []

# populate the list with the product codes
for i in tqdm(validation_df['CustomerID'].unique()):
    temp = validation_df[validation_df["CustomerID"] == i]["StockCode"].tolist()
    purchases_val.append(temp)

value = st.sidebar.slider('Select a customer', 0, len(purchases_val), 0)

#execute recommendation if checkbox_2 is...well...checked
if checkbox_2 == True : 
    #function to aggregate all products of a customer
    def aggregate_vectors(products):
        product_vec = []
        for i in products:
            try:
                product_vec.append(model[i])
            except KeyError:
                continue
            
        return np.mean(product_vec, axis=0)


    st.write("YOUR RECOMMENDATIONS")
    prediction2 = similar_products(aggregate_vectors(purchases_val[value]))
    prediction2_bis = pd.DataFrame(prediction2, columns = ["Product Name","Related"])
    st.table(prediction2_bis)

      #create en empty list to store all images
    list_img = []
    for k in prediction2:
        reco(str(k[0]))

    #plot images
    j=0
    fig = plt.figure()
    fig.set_size_inches(30, 20)
    for pic2 in list_img:
        ax = fig.add_subplot(2,3,j+1)
        plt.axis('off')
        plt.grid(b=None)
        ax.imshow(pic2)
        plt.title(j, fontsize=100, y=1)
        j+=1
    st.pyplot()
