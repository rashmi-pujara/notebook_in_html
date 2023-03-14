import io
import os
from PIL import Image
import streamlit as st
import wget
import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, f1_score, mean_squared_error, confusion_matrix
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, optimizers
import plotly.express as px



def pipeline(df, df_object, df_float):
    pipe = make_pipeline(
        ColumnTransformer(
            transformers=[
                ("encode", OneHotEncoder(), ["hair_color",	"skin_color",	"eye_color", "sex",	"gender",	"species",	"films"]),
            ],
            remainder="passthrough",
        ),
    )
    enco = pipe.fit_transform(df_object).toarray()
    frames =  [enco, df_float.values]
    X = np.concatenate(frames, axis = 1)
    ordinalencoder = OrdinalEncoder()
    y = (ordinalencoder.fit_transform(df.homeworld.values.reshape(-1,1)))
    return X,y



def split(X,y):
    rus = RandomOverSampler(random_state=0)
    rus.fit(X, y)
    X_train_smote, y_train_smote = rus.fit_resample(X, y)
    return train_test_split(X_train_smote,y_train_smote,test_size=0.3,random_state=42)

def baseline(df, df_object, df_float):
    X,y=pipeline(df, df_object,df_float)
    X_train,X_test,y_train,y_test = split(X,y)
    #X_train_tf = tf.convert_to_tensor(X_train.astype(np.float64))
    X_test_tf = tf.convert_to_tensor(X_test.astype(np.float64))
    #y_train_tf = tf.convert_to_tensor(y_train.astype(np.float64))
    y_test_tf = tf.convert_to_tensor(y_test.astype(np.float64))
    model_1 = tf.keras.models.load_model('final_model.h5')
    #y_pred_1 = model_1.predict(X_test_tf)
    model_1.evaluate(X_test_tf, y_test_tf)    
    df_history_1 = pd.read_csv('df_history_1.csv')
    fig = px.line(df_history_1, y=['loss', 'val_loss'], labels=['Loss', 'Validation Loss'])
    st.plotly_chart(fig, use_container_width=True)

def main():
    st.title('streamlit model demo')

    df = pd.read_csv("star_wars_character_dataset.csv")

    ## removing max mass to adjust the mean and std and filling null values with mean data
    df.drop(df[df.mass == np.max(df.mass)].index,inplace=True)
    df.mass.fillna(df.mass.mean(),inplace=True)

    ## removing max birth_year to adjust the mean and std and filling null values with mean data
    df.drop(df[df.birth_year == np.max(df.birth_year)].index,inplace=True)
    df.drop(df[df.birth_year == np.max(df.birth_year)].index,inplace=True)
    df.birth_year.fillna(df.birth_year.mean(),inplace=True)

    df["hair_color"] = df.hair_color.str.split(',').str[0]
    df["eye_color"] = df.eye_color.str.split(',').str[0]
    df["skin_color"] = df.skin_color.str.split(',').str[0]

    df.hair_color.fillna('none',inplace=True)

    ## filling null values for sex, gender, homeworld and species with random 
    df.gender.fillna(random.choice(['masculine','feminine']),inplace=True)
    df.sex.fillna(random.choice(['female','male','hermaphroditic']),inplace=True)
    df.homeworld.fillna(random.choice(['Naboo','Tatooine']),inplace=True)
    df.species.fillna(random.choice(['Human','Droid']),inplace=True)

    df.drop(['vehicles','starships'],axis=1,inplace=True)
    df.height.fillna(df.height.mean(),inplace=True)

    df_object = df[['hair_color','skin_color','eye_color','sex','gender','species','films']]
    df_float = df[['height','mass','birth_year']]


    baseline(df, df_object, df_float)

if __name__ == '__main__':
    main()