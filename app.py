import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from streamlit_option_menu import option_menu
import plotly.express as px
import os
import pickle


# carregando modelo
with open('logistic_model.pkl','rb') as f:
    one_hot , min_max, l_enc, logistic_model = pickle.load(f)

# função de predição
def dropout_prediction(input_data):
    result = logistic_model.predict_proba(input_data)[:,1]
    return result

# carregando csv
def get_data(): 
     return pd.read_csv(os.path.join(os.getcwd(),'data.csv'),
                        usecols = lambda column:column not in ['Unnamed: 0','cod_curso','cod_matricula','cod_ciclo_matricula'])

df = get_data()


# setando layout
st.set_page_config(layout="wide")


# funções para barplot com porcentagem
def percentage_above_bar_relative_to_xgroup(ax):
    all_heights = [[p.get_height() for p in bars] for bars in ax.containers]
    for bars in ax.containers:
        for i, p in enumerate(bars):
            total = sum(xgroup[i] for xgroup in all_heights)
            percentage = f'{(100 * p.get_height() / total) :.1f}%'
            ax.annotate(percentage, (p.get_x() + p.get_width() / 2, p.get_height()), size=12, ha='center', va='bottom')


def show_bars(x_value, hue_value, dataset):

    fig, ax = plt.subplots(figsize=(15, 12))
    
    ax = sns.countplot(x = x_value, hue = hue_value, data = dataset,palette='Set2')
    ax.set(xlabel = x_value, ylabel = 'quantidade')
    percentage_above_bar_relative_to_xgroup(ax)

    st.pyplot(fig)
    
    
with st.sidebar:
    selected = option_menu('Student Dropout Forecast',
                           ['Visualização de dados',
                            'Prever evasão'],
                           icons = ['activity'],
                           default_index=0)