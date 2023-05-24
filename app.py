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

    fig, ax = plt.subplots(figsize=(15, 8))
    
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
    
    
# student dropout visualizations
if(selected=='Visualização de dados'):
    # page title
    st.title('Visualização de dados')
    
    # dataset view
    st.subheader('Base de dados')
    st.write(df)
    
    st.subheader('Proporção de evadidos e concluintes na base de dados')
    
    situacao_counts = df['situacao'].value_counts().reset_index(name='quantidade')
    labels = ['Evadido', 'Concluinte']
    fig = px.pie(situacao_counts, values='quantidade', names='index',width=420,height=420)
    fig.update_layout(
    title="<b>Evadido x Concluinte</b>"
    )
  
    st.write(fig)
    
    
    # ---- Renda com mais evasões ----
    st.subheader('Qual a renda com maior número de evasões?')
    top_evadidos_renda = df.groupby("renda")["situacao"].count().reset_index(name='count').sort_values("count",ascending=False)
    
    fig = px.bar(top_evadidos_renda[:6], x="count", y="renda", orientation="h", color="renda")
    fig.update_layout(
        title="<b>Números de evasão por Renda</b>",
        xaxis_title="Qtde. Evadidos",
        yaxis_title="Renda per Capita"
    )
    st.write(fig)
    
    # insight
    st.info('Dentre os evadidos, as rendas mais presentes são 0<RFP<=0,5 e 0,5<RFP<=1,5')
    
    # ---- Faixa etária com mais evasões ----
    st.subheader('Qual a faixa etária com maior número de evasões?')
    
    
    top_evadidos_idade = df.groupby("faixa_etaria")["situacao"].count().reset_index(name='count').sort_values("count", ascending=False)

    fig = px.bar(top_evadidos_idade[:10], x="count", y="faixa_etaria", orientation="h", color="faixa_etaria")
    fig.update_layout(
    title="<b>Top 10 Números de evasão por faixa etária</b>",
    xaxis_title="Evadidos",
    yaxis_title="Faixa etária"
    )
    
    st.write(fig)
    
    # insights
    st.info('Dentre os evadidos, a faixa etária 20-24 é a mais presente.')
    st.info('A faixa etária que mais contribui para evasão está entre 20 e 29 anos, pois a maior parte dos dados de evadidos está concentrada nessa faixa.')
    
    # ---- visualizando proporções ----
    
    st.subheader('Visualização dinâmica')
    option = st.selectbox("Escolha uma opção:",('Faixa etária x Situação','Turno x Situação', 'Situação x Eixo Tec.',
                                                     'Cor x Situação','Sexo x Situação'))
    
    if option == "Faixa etária x Situação":
        show_bars("faixa_etaria","situacao",df)
        
    if option == "Turno x Situação":
        show_bars("turno","situacao",df)
    
    if option == "Situação x Eixo Tec.":
        show_bars("situacao","eixo_tec",df)
    
    if option == "Cor x Situação":
        show_bars("cor","situacao",df)
        
    if option == "Sexo x Situação":
        show_bars("sexo","situacao",df)

    # info
    st.text('---'*100)
    st.markdown('##### A faixa etária de 20 a 29 anos conta com o maior número de alunos, assim como o maior número de evadidos. É importante analisar os alunos dessa faixa já que a maior parte das ocorrências estarão nela.')
    
    # criando os DFs necessários
    evadidos_df = df[df['situacao'] == 'E'] # pegando os alunos que evadiram
    concluintes_df = df[df['situacao'] == 'C'] # pegando os alunos que concluíram
    evadidos_20_24 = evadidos_df[evadidos_df['faixa_etaria']=='20-24']
    concluintes_20_24 = concluintes_df[concluintes_df['faixa_etaria']=='20-24']
    evadidos_25_29 = evadidos_df[evadidos_df['faixa_etaria']=='25-29']
    concluintes_25_29 = concluintes_df[concluintes_df['faixa_etaria']=='25-29']
    df_20_29_evadidos = pd.concat([evadidos_20_24, evadidos_25_29])
    df_20_29_concluintes =  pd.concat([concluintes_20_24, concluintes_25_29])
    df_20_29 = pd.concat([df_20_29_evadidos, df_20_29_concluintes])
    
    # Sunburst vizualizations
    st.subheader('20-29 - A concentração de evadidos é mais presente em quais cursos e tipos de cursos?')
    
    st.info('Clique em alguma área do centro do gráfico se quiser expandir.')
    
    fig = px.sunburst(df_20_29_evadidos, path=['tipo_curso', 'nome_curso'],color='tipo_curso',
                  color_discrete_map={'tecnologia':'black','bacharelado':'gold','licenciatura':'blue'})

    fig.update_layout(title='<b>Evadidos na faixa 20-29 por cursos e seus tipos</b>',height=670)
    st.write(fig)
    
    
    st.subheader('20-29 - Dentre os maiores números de evasões dentro de tecnologia, qual curso apresenta maior proporção de evadidos?')
    
    df_tec = df_20_29[(df_20_29['nome_curso']=='construcao de edificios')|
                          (df_20_29['nome_curso']=='analise e desenvolvimento de sistemas')|
                          (df_20_29['nome_curso']=='agroecologia')|
                          (df_20_29['nome_curso']=='sistemas para internet')|
                          (df_20_29['nome_curso']=='redes de computadores')|
                          (df_20_29['nome_curso']=='automacao industrial')]

    fig = px.sunburst(df_tec, path=['nome_curso', 'situacao'],color='nome_curso')
    fig.update_traces(textinfo="label+percent parent")
    fig.update_layout(title='<b>Evadidos x Concluintes na faixa 20-29 dos cursos de tecnologia</b>',height=670)
    st.write(fig)
    
    
    st.subheader('20-29 - Dentre os maiores números de evasões dentro de bacharelado, qual curso apresenta maior proporção de evadidos?')

    df_bach = df_20_29[(df_20_29['nome_curso']=='administracao')|
                          (df_20_29['nome_curso']=='engenharia eletronica')|
                          (df_20_29['nome_curso']=='engenharia civil')|
                          (df_20_29['nome_curso']=='engenharia de computacao')|
                          (df_20_29['nome_curso']=='engenharia de controle e automacao')]


    fig = px.sunburst(df_bach, path=['nome_curso', 'situacao'],color='nome_curso')
    fig.update_traces(textinfo="label+percent parent")
    fig.update_layout(title='<b>Evadidos x Concluintes na faixa 20-29 dos cursos de bacharelado</b>',height=670)
    st.write(fig)
    
    st.subheader('20-29 - Dentre os maiores números de evasões dentro de licenciatura, qual curso apresenta maior proporção de evadidos?')

    df_lic = df_20_29[(df_20_29['nome_curso']=='matematica')|
                          (df_20_29['nome_curso']=='quimica')|
                          (df_20_29['nome_curso']=='ciencias biologicas')|
                          (df_20_29['nome_curso']=='fisica')|
                          (df_20_29['nome_curso']=='educacao fisica')]


    fig = px.sunburst(df_lic, path=['nome_curso', 'situacao'],color='nome_curso')
    fig.update_traces(textinfo="label+percent parent")
    fig.update_layout(title='<b>Evadidos x Concluintes na faixa 20-29 dos cursos de licenciatura<b>',height=670)
    st.write(fig)
    
    
    #  --- correlation between ----
    st.text('---'*100)
    st.subheader('A carga horária influencia na proporção de evadidos?')
    
    df_carga_evasao = pd.crosstab(df['carga_horaria'],df['situacao']).reset_index()
    
    fig = px.scatter(df_carga_evasao, x="carga_horaria", y="E",size='E', color='carga_horaria',trendline="ols")
    fig.update_layout(title='<b>Evasão x Carga Horária</b>',  yaxis_title="Qtde. Evadidos", )
    st.write(fig)
    
    st.markdown('A julgar pela linha da regressão parece haver uma correlação negativa e podemos dizer que:')
    st.markdown('- quanto menor a carga horária, maior a proporção de evadidos')
    st.markdown('- quanto maior a carga horária, menor a proporção de evadidos')
    
    st.subheader('O fator de esforço influencia na proporção de evadidos?')
    
    new_df = pd.crosstab(df['fator_esforco'],df['situacao']).reset_index()
    fig = px.scatter(new_df, x="fator_esforco", y="E",size='E',color='fator_esforco',trendline="ols")
    fig.update_layout(title='<b>Evasão x Fator de esforço</b>',  yaxis_title="Qtde. Evadidos", )
    
    st.write(fig)
    st.markdown('Dada a linha de regressão linear podemos dizer que o número de evadidos diminui ao aumentar-se o fator de esforço.')
    
    # indo
    st.info('Mas vale ressaltar que dado o baixo valor de R² essas tendências acimas observadas para carga horária e fator de esforço não são muito confiáveis.')
    
    
    