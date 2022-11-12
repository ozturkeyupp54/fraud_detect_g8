import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image 
import streamlit.components.v1 as components
import codecs
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import sweetviz as sv
import os
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import tensorflow as tf
from keras import Model
from keras.models import load_model
# import xgboost as xgb



import streamlit as st

from tempfile import NamedTemporaryFile

st.set_option('deprecation.showfileUploaderEncoding', True)



def st_display_sweetviz(report_html,width=1000,height=500):
        report_file = codecs.open(report_html,'r')
        page = report_file.read()
        components.html(page,width=width,height=height,scrolling=True)
        
# today=st.date_input("Today is", datetime.datetime.now())
def explore_data(dataset):
        df = pd.read_csv(os.path.join(dataset))
        return df 

def main():
    st.write("""# Fraud Detection""")

    # st.write("")
    
    # choice = st.sidebar.selectbox("Menu",menu)

    # if  choice == "Random Forest Model":
    #     st.subheader("Random Forest Classifier")
    st.image('image_1.gif')

    if st.checkbox("Single Forecast"):
        filename = 'random_pickle_model'
        model = pickle.load(open(filename, 'rb'))
        # scaled_random= pickle.load(open("random_pipeline","rb"))

        # st.sidebar.title("Final Model ")
        # st.sidebar.header("Sidebar header")

        V3=st.sidebar.number_input(label='V3',min_value=-15.0,max_value=15.0,step=0.001,)
        v10=st.sidebar.number_input(label='V10',min_value=-15.0,max_value=15.0,step=0.001,)
        v11=st.sidebar.number_input(label='V11',min_value=-15.0,max_value=15.0,step=0.001,)
        v12=st.sidebar.number_input(label='V12',min_value=-15.0,max_value=15.0,step=0.001,)
        v14=st.sidebar.number_input(label='V14',min_value=-15.0,max_value=15.0,step=0.001)
        v16=st.sidebar.number_input(label='V16',min_value=-15.0,max_value=15.0,step=0.001,)
        v17=st.sidebar.number_input(label='V17',min_value=-15.0,max_value=15.0,step=0.001,)
        

        dict={
            "V14":v14,
            "V17":v17,
            "V10":v10,
            "V12":v12,
            "V11":v11,
            "V16":v16,
            "V3":V3
        }

        df= pd.DataFrame.from_dict([dict])

        st.table(df)

        if st.button("Predict"):
            predictions = model.predict(df)

            df["pred"] = predictions

            st.write(predictions[0])
            if predictions:
                st.image('iamge_2.gif')

    if st.checkbox("Multiple Forecast"):
        filename = 'random_pickle_model'
        model = pickle.load(open(filename, 'rb'))
        # buffer = 
        # temp_file = NamedTemporaryFile(delete=False)
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            df_df = pd.read_csv(uploaded_file)
            st.write(df_df)


            importance_ = ['V14','V17','V10','V12','V11','V16','V3']
            X5 = df_df[importance_]
            st.write(X5)
            if st.button("Predict_multiple"):
                predictions = []
                

                predictions = model.predict(X5)
                predictions = pd.DataFrame(predictions)
                predictions.dropna(inplace=True)
                predictions = predictions.drop(predictions[predictions[0]!=1].index)
                t = predictions[predictions >0]
                st.write(t)

if __name__ == '__main__':
        main()










