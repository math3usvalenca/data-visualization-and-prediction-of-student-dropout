import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from streamlit_option_menu import option_menu
from matplotlib.gridspec import GridSpec
import plotly.express as px
import os
import pickle
import datetime
from PIL import Image

# setando layout
st.set_page_config(layout="wide")


# carregando modelo
with open('logreg_model.pkl','rb') as f:
    one_hot , min_max, l_enc, logistic_model = pickle.load(f)

# função de predição
def dropout_prediction(input_data):
    result = logistic_model.predict_proba(input_data)[:,1]
    return result

# carregando csv
@st.cache_data
def get_data(): 
     return pd.read_csv(os.path.join(os.getcwd(),'student_data.csv'),
                        usecols = lambda column:column not in ['cod_curso','codigo_da_matricula','codigo_do_ciclo_matricula'])

df = get_data()

@st.cache_data
def get_evadidos():
    evadidos_df = df[df['categoria_de_situacao'] == 'E'] # pegando os alunos que evadiram
    return evadidos_df

@st.cache_data
def get_concluintes():
    concluintes_df = df[df['categoria_de_situacao'] == 'C'] # pegando os alunos que concluíram
    return concluintes_df


evadidos_df = get_evadidos()
concluintes_df = get_concluintes()

@st.cache_data
def get_evadidos_20_29():
   evadidos_20_29 = evadidos_df.query('faixa_etaria=="20-24" | faixa_etaria=="20-29"')
   return evadidos_20_29

@st.cache_data
def get_concluintes_20_29():
   concluintes_20_29 = concluintes_df.query('faixa_etaria=="20-24" | faixa_etaria=="20-29"')
   return concluintes_20_29

evadidos_20_29 = get_evadidos_20_29()
concluintes_20_29 = get_concluintes_20_29()

df_20_29 = pd.concat([evadidos_20_29, concluintes_20_29])
df_30_49 = df[(df['faixa_etaria']=='30-34') | (df['faixa_etaria']=='35-39') | (df['faixa_etaria']=='40-44') | (df['faixa_etaria']=='45-49')]
ages = pd.crosstab(evadidos_df['idade'],evadidos_df['categoria_de_situacao']).reset_index()

# função para barplot com porcentagem 
# @st.cache_data
# def show_bars(subplot_value, x_value, hue_value, dataset):
#     plt.subplot(subplot_value)
#     ax = sns.countplot(x = x_value, hue = hue_value, data = dataset,palette='plasma')
#     ax.set(xlabel = x_value, ylabel = 'Quantidade')
    
    
#     all_heights = [[p.get_height() for p in bars] for bars in ax.containers]
    
#     for bars in ax.containers:
#         for x, p in enumerate(bars):
#             total = sum(xgroup[x] for xgroup in all_heights) 
#             percentage = f'{(100 * p.get_height() / total) :.1f}%'
#             ax.annotate(f'{p.get_height()}\n{percentage}', (p.get_x() + p.get_width() / 2, p.get_height()), size=12, ha='center', va='bottom')
   

#     x_ticks = [item.get_text() for item in ax.get_xticklabels()]
#     if len(x_ticks[0]) > 10:
#         ax.set_xticklabels(x_ticks, rotation=45, fontsize=12)
#     else:
#         ax.set_xticklabels(x_ticks, fontsize=12)
 
 
# função para exibir gŕaficos sunburst
@st.cache_data
def show_sunburst(course_type,courses_names=[],age_group="20-29"):
    df = pd.DataFrame()
    if age_group == '20-29':
        rnge = len(courses_names)
        for x in range(rnge):
            new_df = df_20_29[df_20_29['nome_do_curso']==courses_names[x]]
            df = pd.concat([df,new_df])
    if age_group == '30-49':
        rnge = len(courses_names)
        for x in range(rnge):
            new_df = df_30_49[df_30_49['nome_do_curso']==courses_names[x]]
            df = pd.concat([df,new_df])

    fig = px.sunburst(df, path=['nome_do_curso', 'categoria_de_situacao'],color='nome_do_curso',labels={"count": "quantidade"})
    fig.update_traces(textinfo="label+percent parent")
    fig.update_layout(title=f'<b> Percentual de Evadidos x Concluintes na faixa {age_group} dos cursos de {course_type}</b>',height=670)
    st.write(fig)

    
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
    st.write(df.head())
    
    st.subheader('Proporção de evadidos e concluintes na base de dados')
    
    situacao_counts = df['categoria_de_situacao'].value_counts().reset_index(name='quantidade')
    labels = ['Evadido', 'Concluinte']
    fig = px.pie(situacao_counts, values='quantidade', names='index',width=420,height=420)
    fig.update_layout(
    title="<b>Evadido x Concluinte</b>"
    )
  
    st.write(fig)
    
    
    # 
    st.subheader('Como está distribuída a evesão pelos Campus?')
    fig = plt.figure(constrained_layout=True,figsize=(15,9))

    gs = GridSpec(1,3, figure=fig)

    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0, 1:3])

    numero_de_campus = len(df['unidade_de_ensino'].unique())
    numero_de_evadidos = len(evadidos_df)
    ax1.text(0.40,0.75, 'Existem',fontsize=14, ha='center')
    ax1.text(0.40,0.63,f'{numero_de_evadidos}', fontsize=64, color='red', ha='center')
    ax1.text(0.40,0.59,'evadidos espalhados por', fontsize=14, ha='center')
    ax1.text(0.40, 0.50, numero_de_campus, fontsize=44, ha='center', color='blue', style='italic', weight='bold')
    ax1.text(0.40, 0.45, 'Campus diferentes', fontsize=14, ha='center')
    ax1.axis('off')

    ax2 = sns.countplot(x="unidade_de_ensino", data=evadidos_df, palette="plasma", order=evadidos_df['unidade_de_ensino'].value_counts().index[0:10])
    ax2.set_title('Top 10 números de evadidos por Campus', size=18)

    ax2.bar_label(ax2.containers[0])
    plt.ylabel('Quantidade')
    plt.xlabel('Unidade de ensino')
    for tick in ax2.get_xticklabels():
        tick.set_rotation(25)
  
    st.write(fig)
        
    
    st.subheader('Quais Campus têm maiores percentuais de evadidos?')
    
    image = Image.open('./images/image.png')
    st.image(image)
    
    # fig = plt.figure(constrained_layout=True,figsize=(18,20))
    # show_bars(211,"unidade_de_ensino","categoria_de_situacao",df)
    # plt.text(1.90,2000,'Os Campus Campina Grande, Picuí e João Pessoa \n apresentam os maiores percentuais de evasão' ,fontsize=14, ha='center')
    # plt.title('Evadidos por Campus',fontsize=14)
    # st.write(fig)
    
    
    
    
    # ---- Renda com mais evasões ----
    st.subheader('Qual a renda com maior número de evasões?')
    top_evadidos_renda = evadidos_df.groupby("renda")["categoria_de_situacao"].count().reset_index(name='count').sort_values("count",ascending=False)

    fig = px.bar(top_evadidos_renda[:6], x="count", y="renda", orientation="h", color="renda", text='count', width=800)
    fig.update_layout(
        title="<b>Números de evasão por Renda</b>",
        xaxis_title="Qtde. Evadidos",
        yaxis_title="Renda per Capita"
    )
    
    fig.add_annotation(
        x=1250,
        y=1.93,
        text="Dentre os evadidos, as rendas mais presentes estão entre 0<RFP<=0,5 e 0,5<RFP<=1,5",
        font=dict(
            size=11,
            color="white"
        ),
        showarrow=False,
    )
    st.write(fig)
      
    # ---- Faixa etária com mais evasões ----
    st.subheader('Qual a faixa etária com maior número de evasões?')
    
    
    top_evadidos_idade = evadidos_df.groupby("faixa_etaria")["categoria_de_situacao"].count().reset_index(name='count').sort_values("count", ascending=False)
    

    fig = px.bar(top_evadidos_idade[:10], x="count", y="faixa_etaria", orientation="h", text='count',color="faixa_etaria", width=900)
    fig.update_layout(
        title="<b>Top 10 Números de evasão por faixa etária</b>",
        xaxis_title="Qtde. Evadidos",
        yaxis_title="Faixa etária"
    )


    fig.add_annotation(
        x=950,
        y=4.93,
        text="A faixa etária que mais contribui para evasão está entre 20 e 29 anos,<br> pois a maior parte dos dados de evadidos está concentrada nessa faixa",
        font=dict(
            size=14,
            color="white"
        ),
        showarrow=False,
    )

    st.write(fig)
 
    # ---- visualizando proporções ----
    
    st.subheader('Visualização dinâmica')
    option = st.selectbox("Escolha uma opção:",('Faixa etária x Situação','Turno x Situação', 'Situação x Eixo Tec.',
                                                     'Cor x Situação','Sexo x Situação'))
    
    if option == "Faixa etária x Situação":
        image = Image.open('./images/evasao_faixa.png')
        st.image(image)
      
    if option == "Turno x Situação":
        image = Image.open('./images/evasao_turno.png')
        st.image(image)
      
    if option == "Situação x Eixo Tec.":
        image = Image.open('./images/evasao_eixo.png')
        st.image(image)
  
    if option == "Cor x Situação":
        image = Image.open('./images/evasao_cor.png')
        st.image(image)
      
    if option == "Sexo x Situação":
        image = Image.open('./images/evasao_sexo.png')
        st.image(image)
   
    # info
    st.text('---'*100)
    st.markdown('##### A faixa etária de 20 a 29 anos conta com o maior número de alunos, assim como o maior número de evadidos. É importante analisar os alunos dessa faixa já que a maior parte das ocorrências estarão nela.')
    
    # Sunburst vizualizations
    st.subheader('20-29 - A concentração de evadidos é mais presente em quais cursos e tipos de cursos?')
    
    st.info('Clique em alguma área do centro do gráfico se quiser expandir.')
    fig = px.sunburst(evadidos_20_29, path=['tipo_de_curso', 'nome_do_curso'],color='tipo_de_curso',
                  color_discrete_map={'tecnologia':'black','bacharelado':'gold','licenciatura':'blue'},labels={"count": "quantidade","labels":"curso"})

    fig.update_layout(title='<b>Concentração de Evadidos na faixa 20-29 por cursos e seus tipos</b>',height=670)

    st.write(fig)
    
    st.subheader('Visualização dinâmica')
    option = st.selectbox("Escolha uma opção:",('Na faixa 20-29 o percentual de evadidos é maior em quais cursos de tecnologia?',
                                                'Na faixa 20-29 o percentual de evadidos é maior em quais cursos de bacharelado?',
                                                'Na faixa 20-29 o percentual de evadidos é maior em quais cursos de licenciatura?',
                                                ))
    
    if option == 'Na faixa 20-29 o percentual de evadidos é maior em quais cursos de tecnologia?':
        show_sunburst('tecnologia',['construcao de edificios','analise e desenvolvimento de sistemas','agroecologia',
                           'sistemas para internet','redes de computadores','automacao industrial'])
        
    elif option == 'Na faixa 20-29 o percentual de evadidos é maior em quais cursos de bacharelado?':
        show_sunburst('bacharelado',['administracao','engenharia eletronica','engenharia civil','engenharia de computacao',
                            'engenharia de controle e automacao'])
    elif option == 'Na faixa 20-29 o percentual de evadidos é maior em quais cursos de licenciatura?':
        show_sunburst('licenciatura',['matematica','quimica','ciencias biologicas',
                           'fisica','letras - lingua portuguesa'])
    
    #  --- correlation between ----
    st.text('---'*100)
    
    st.subheader('A carga horária influencia na proporção de evadidos?')
    
    df = pd.crosstab(evadidos_df['carga_horaria_do_curso'],evadidos_df['categoria_de_situacao']).reset_index()
    
    fig = px.scatter(df, x="carga_horaria_do_curso", y="E",size='E', color='E',trendline="ols")
    fig.update_layout(title='<b>Evasão x Carga Horária</b>',  yaxis_title="Qtde. Evadidos", )
   
    st.write(fig)
    st.info('Não há uma correlação significativa entre a carga horária e a evasão')
    
    st.subheader('A idade possui alguma relação com a evasão?')
    # ages = pd.crosstab(evadidos_df['idade'],evadidos_df['categoria_de_situacao']).reset_index()
    fig = px.scatter(ages, x="idade", y="E",size='E', color='E',trendline="ols",width=900)
    fig.update_layout(title='<b>Evasão x Idade</b>',  yaxis_title="Qtde. Evadidos", )

    fig.add_annotation(
        x=49,
        y=300,
        text="Quando diminui-se a idade, <br> aumenta-se o número de evadidos",
        font=dict(
            size=14,
            color="white"
        ),
        showarrow=False,
    )

    st.write(fig)
    
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
                                  'sistemas de telecomunicacoes','telematica','letras - lingua portuguesa','design de interiores')
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
        idade = st.number_input('Idade') 
      
     
    with col2:
        unidade_ensino = st.selectbox('Unidade de ensino',
                                  ('Campus João Pessoa',
                                   'Campus Cajazeiras',
                                    'Campus Picuí','Campus Campina Grande','Campus Sousa','Campus Monteiro',
                                    'Campus Patos','Campus Guarabira','Campus Cabedelo','Campus Princesa Isabel'))
        
     
    with col3:
        mes_ocorrencia = st.date_input('Data de mudança de Situação',
                                     datetime.date(2023,1,1))
        
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
                              'vespertino','nao se aplica')
                             )
           
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
        mes_ocorrencia =  mes_ocorrencia.strftime("%d/%m/%Y")
        carga_horaria = int(carga_horaria)

        # lista de colunas
        # list_cols = ['nome_do_curso','carga_horaria_do_curso','eixo_tecnologico',
        #              'fator_de_esforco_de_curso','tipo_de_curso','turno','data_de_inicio_do_ciclo','data_de_fim_previsto_do_ciclo',
        #              'renda','sexo','faixa_etaria','mes_de_ocorrencia_da_situcao','idade',
        #             'cor']
        
        # criando um novo DF
        d = {'nome_do_curso': [nome_curso], 'carga_horaria_do_curso': [carga_horaria],'eixo_tecnologico':[eixo_tec],
             'fator_de_esforco_de_curso':[fator_esforco],'tipo_de_curso':[tipo_curso],'turno':[turno],
            'data_de_inicio_do_ciclo':[inicio_curso],'data_de_fim_previsto_do_ciclo':[final_esperado],
             'renda':[renda],'sexo':[sexo],'idade':[idade], 'unidade_de_ensino':[unidade_ensino], 'mes_de_ocorrencia_da_situacao':[mes_ocorrencia],
             'faixa_etaria':[faixa_etaria],'cor':[cor],
             }
        
        new_student = pd.DataFrame(data=d)
        
        # guardando variávies categóricas                
        X_cat = new_student[['nome_do_curso','eixo_tecnologico',
           'tipo_de_curso','turno','data_de_inicio_do_ciclo','data_de_fim_previsto_do_ciclo','renda','sexo','faixa_etaria','cor',
           'unidade_de_ensino','mes_de_ocorrencia_da_situacao']]
        
        # codificando variáveis categóricas com OneHotEnconding
        X_encoded = one_hot.transform(X_cat).toarray()
        X_encoded = pd.DataFrame(X_encoded) # transformando em DataFrame
        X_encoded.columns = one_hot.get_feature_names_out() # renomeando as colunas
        
        # guardando variávides numéricas 
        X_numerical = new_student[['carga_horaria_do_curso','fator_de_esforco_de_curso','idade']]
        # concatenando os DFs
        X_all = pd.concat([X_encoded, X_numerical],axis=1)
        # escalonando o X_all com MinMax
        X_scaled = min_max.transform(X_all)
        
        # chamando a função para realizar a previsão
        prediction = dropout_prediction(X_scaled)
        
        st.text('Aluno informado:')
        st.write(new_student)
        # exibindo resultados
        if(prediction > 0.5):
            st.warning(f'{round(prediction[0]*100)} % de chances de ser da classe Evadido')
        else:
            st.success(f'Apenas {round(prediction[0]*100)} % de chances de ser da classe Evadido')

    st.info('O modelo que está sendo utilizado é a Regressão Logística. Esse modelo tem um Recall de 0.90 para classes positivas. Em outras palavras, se o estudante informado for realmente um potencial evadido existem 90 por cento de chances do modelo o classificar corretamente como evadido.')
    
    st.text('Métricas do modelo')
    
    metrics = {'Precision': [0.86], 'Recall': [0.90],'Accuracy':[0.81]}
        
    df_metrics = pd.DataFrame(data=metrics)
    
    st.table(df_metrics)