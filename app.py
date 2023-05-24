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
    st.text('Github Account:')
    st.markdown('<a href="https://github.com/math3usvalenca" style="text-decoration:none;background:green;padding:5px;color:white;border-radius:3px">Matheus Valença</a>', unsafe_allow_html=True)
    
    st.text('Project repositories:')
    st.markdown('<a href="https://github.com/math3usvalenca/machine-learning-no-combate-a-evasao-estudantil" style="text-decoration:none;background:#4287f5;padding:5px;color:white;border-radius:3px">EDA/ML</a>', unsafe_allow_html=True)
    st.markdown('<a href="https://github.com/math3usvalenca/data-visualization-and-prediction-of-student-dropout" style="text-decoration:none;background:#4287f5;padding:5px;color:white;border-radius:3px">Sreamlit</a>', unsafe_allow_html=True)
  
    
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
    
    st.text('Para ver todos os insights retirados acesse o link abaixo:')
    
    link = '[Notebook](https://nbviewer.org/github/math3usvalenca/machine-learning-no-combate-a-evasao-estudantil/blob/main/analise-de-dados-estudantis/AED.ipynb)'
    st.markdown(link, unsafe_allow_html=True)
    
if(selected=='Prever evasão'):
    # page title
    st.title('Prevendo evasão')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
       nome_curso = st.selectbox('Nome do curso',
                                 ('analise e desenvolvimento de sistemas','administracao','agroecologia','alimentos','automacao industrial',
                                  'ciencias biologicas','construcao de edificios','design de interiores','design grafico','educacao fisica',
                                  'engenharia de controle e automacao','engenharia mecanica','engenharia de computacao','engenharia civil','engenharia eletronica',
                                  'fisica','geoprocessamento','gestao ambiental','gestao comercial','matematica','medicina veterinaria','negocios imobiliarios',
                                  'quimica','redes de computadores','sistemas para internet','seguranca no trabalho',
                                  'sistemas de telecomunicacoes','telematica')
                                 )
        
    with col2:
        carga_horaria = st.number_input('Carga horária')
        
    with col3:
        eixo_tec = st.selectbox('Eixo tecnológico',
                                ('infraestrutura',
                                 'gestao e negocios',
                                 'recursos naturais',
                                 'producao alimenticia',
                                 'desenvolvimento educacional e social',
                                 'informacao e comunicacao',
                                 'producao cultural e design',
                                 'ambiente e saude',
                                 'seguranca',
                                 'controle e processos industriais'))
        
    with col1:
        fator_esforco = st.number_input('Fator esforço')
        
    with col2:
        tipo_curso = st.selectbox('Tipo de curso',
                                  ('tecnologia',
                                   'bacharelado',
                                    'licenciatura'))
        
    with col3:
        turno = st.selectbox('Turno',
                             ('integral','noturno','matutino',
                              'vespertino')
                             )
        
    import datetime
    
    with col1:
        inicio_curso = st.date_input('Inicio do curso',
                                     datetime.date(2023,1,1))
        
    with col2:
        final_esperado = st.date_input('Final esperado',
                                     datetime.date(2023,1,1))    
    
    with col3:
        renda = st.selectbox('Renda',
                             ('0<RFP<=0,5',
                              '0,5<RFP<=1',
                              '1,0<RFP<=1,5',
                              '1,5<RFP<=2,5',
                            '2,5<RFP<=3,5',
                            'RFP>3,5'))
    with col1:
        sexo = st.selectbox('Sexo',
                            ('M','F'))
        
    with col2:
        faixa_etaria = st.selectbox('Faixa etária',
                                    ('15-19','20-24',
                                    '25-29','30-34',
                                    '35-39','40-44',
                                    '45-49','50-54',
                                    '55-59'))
        
    with col3:
        cor = st.selectbox('Cor',
                           ('branca','parda','preta',
                            'amarela','indigena')) 
    
    # prevendo
    if st.button('Prever'):
      
        # conversão de variáveis
        inicio_curso =  inicio_curso.strftime("%d/%m/%Y")  
        final_esperado =  final_esperado.strftime("%d/%m/%Y")
        
        carga_horaria = int(carga_horaria)

        # lista de colunas
        list_cols = ['nome_curso','carga_horaria','eixo_tec','fator_esforco','tipo_curso','turno','inicio_curso','final_esperado',
                     'renda','sexo','faixa_etaria',
                    'cor']
        
        # criando um novo DF
        d = {'nome_curso': [nome_curso], 'carga_horaria': [carga_horaria],'eixo_tec':[eixo_tec],'fator_esforco':[fator_esforco],
             'tipo_curso':[tipo_curso],'turno':[turno],'inicio_curso':[inicio_curso],'final_esperado':[final_esperado],
             'renda':[renda],'sexo':[sexo],'faixa_etaria':[faixa_etaria],'cor':[cor]}
        
        new_student = pd.DataFrame(data=d)
        
        # guardando variávies categóricas                
        X_cat = new_student[['nome_curso','eixo_tec','tipo_curso','turno','inicio_curso','final_esperado','renda','sexo','faixa_etaria','cor']]
        # codificando variáveis categóricas com OneHotEnconding
        X_encoded = one_hot.transform(X_cat).toarray()
        X_encoded = pd.DataFrame(X_encoded) # transformando em DataFrame
        X_encoded.columns = one_hot.get_feature_names_out() # renomeando as colunas
        
        # guardando variávides numéricas 
        X_numerical = new_student[['carga_horaria','fator_esforco']]
        # concatenando os DFs
        X_all = pd.concat([X_encoded, X_numerical],axis=1)
        # escalonando o X_all cin MinMax
        X_scaled = min_max.transform(X_all)
        
        # chamando a função para realizar a previsão
        prediction = dropout_prediction(X_scaled)
        
        st.text('Aluno informado:')
        st.write(new_student)
        # exibindo resultados
        if(prediction > 0.5):
            st.warning(f'{round(prediction[0]*100)} % de chances de evasão')
        else:
            st.success(f'Apenas {round(prediction[0]*100)} % de chances de evasão')

    st.info('O modelo que está sendo utilizado é a regressão logística. Esse modelo tem um Recall de 0.89 para classes positivas. Em outras palavras, se o estudante informado for realmente um potencial evadido existem 89 por cento de chances do modelo o classificar corretamente como evadido.')
    
    st.text('Métricas do modelo')
    
    metrics = {'Precision': [0.84], 'Recall': [0.89],'Accuracy':[0.80]}
        
    df_metrics = pd.DataFrame(data=metrics)
    
    st.table(df_metrics)