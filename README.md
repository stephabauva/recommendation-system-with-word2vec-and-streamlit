# recommendation-system-with-word2vec-and-streamlit
Download the database at https://www.kaggle.com/steph2019/customers-purchases-with-product-id

step 1 - Clone this repository and add the database downloaded above

step 2 - Navigate in the folder where you cloned the directory and enter 
```pip3 install -r requirements.txt```

step 3 - Using your terminal, navigate to the directory containing the app.py file

step 4 - Type the following in your terminal: 
```streamlit run app_reco_v3.py```

Your browser should automatically open, otherwise copy-paste the localhost address in your browser.

On the app you can allow (check or uncheck) the two recommendations on the sidebar. 
The first one recommends items based on one individual input item. 
The second that comes with a slider, does some recommendations based on all a customer's purchases. Each number on the slider is an individual customer.
Then, the algorithm downloads those items's pictures on Google Image and show you what the recommendations look like (might take a minute).
