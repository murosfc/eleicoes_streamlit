import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

#dados de suporte
X = pd.read_csv('X.csv')
y = pd.read_csv('y.csv')
candidatos = pd.read_csv('num_nome.csv')
nomes = np.array(candidatos['NM_CANDIDATO'])
nomes = np.insert(nomes, 0, '')
status = {0:'NÃO ELEITO', 1:'ELEITO', 2:'ELEITO', 3:'NÃO ELEITO', 4:'SUPLENTE'} 
#status corretos {0:'#NULO#', 1:'ELEITO POR MÉDIA', 2:'ELEITO POR QP', 3:'NÃO ELEITO', 4:'SUPLENTE'}. O utilizado é para atender ao enunciado
#Função buscar resultado
def getResult(candidato):
    resultado = 0
    cand = np.array(candidato).reshape(-1,1)   
    scaler = MinMaxScaler(feature_range = (0, 1))    
    candidato = scaler.fit_transform(cand)     
    resultado = int(modelo.predict(candidato.reshape(1,7)))    
    return status[resultado]

#Função modelo
def rna_mlp():
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=101)
    scalerTrain = MinMaxScaler(feature_range = (0, 1))
    scalerTest = MinMaxScaler(feature_range = (0, 1))
    X_train = scalerTrain.fit_transform(X_train)
    X_test = scalerTest.fit_transform(X_test) 
    modelo = MLPClassifier(verbose = True, hidden_layer_sizes=(7,7), max_iter = 10000)   
    modelo.fit(X_train, y_train)
    return modelo


#NM_PARTIDO	NR_IDADE_DATA_POSSE	DS_GENERO	DS_GRAU_INSTRUCAO	DS_COR_RACA	VR_RECEITA	VR_DESPESA_CONTRATADA
#WEB APP
st.set_page_config('Previsão de eleição. Por: Felipe Muros')
modelo = rna_mlp()
st.title("Previsão de eleição de vereador em Campos dos Goytacazes - RJ")
raca_opt = ['NÃO INFORMADO','PARDA', 'BRANCA', 'PRETA', 'INDÍGENA', 'AMARELA']
partido_opt = ['PATRIOTA', 'PARTIDO SOCIAL DEMOCRÁTICO',
       'PARTIDO REPUBLICANO DA ORDEM SOCIAL',
       'PARTIDO DA SOCIAL DEMOCRACIA BRASILEIRA',
       'PARTIDO RENOVADOR TRABALHISTA BRASILEIRO',
       'PARTIDO DA MOBILIZAÇÃO NACIONAL', 'PARTIDO LIBERAL',
       'REPUBLICANOS', 'PARTIDO DEMOCRÁTICO TRABALHISTA',
       'PARTIDO SOCIALISTA BRASILEIRO', 'REDE SUSTENTABILIDADE',
       'PARTIDO DOS TRABALHADORES', 'PODEMOS', 'PARTIDO SOCIAL CRISTÃO',
       'PARTIDO TRABALHISTA BRASILEIRO', 'DEMOCRACIA CRISTÃ', 'CIDADANIA',
       'PARTIDO TRABALHISTA CRISTÃO', 'AVANTE', 'DEMOCRATAS',
       'PARTIDO DA MULHER BRASILEIRA', 'PARTIDO COMUNISTA DO BRASIL',
       'MOVIMENTO DEMOCRÁTICO BRASILEIRO', 'PARTIDO SOCIAL LIBERAL',
       'SOLIDARIEDADE', 'PARTIDO SOCIALISMO E LIBERDADE']
escolaridade_opt = ['ENSINO MÉDIO COMPLETO', 'ENSINO FUNDAMENTAL COMPLETO',
       'ENSINO FUNDAMENTAL INCOMPLETO', 'SUPERIOR INCOMPLETO',
       'SUPERIOR COMPLETO', 'ENSINO MÉDIO INCOMPLETO', 'LÊ E ESCREVE']

st.sidebar.title('Insira os dados do candidato')

with st.sidebar.form(key ='Form1'):
    idade = st.sidebar.number_input('Idade', step=1)
    sexo = st.sidebar.selectbox('Sexo',['MASCULINO','FEMININO'])
    raca = st.sidebar.selectbox('Raça', raca_opt)
    partido = st.sidebar.selectbox('Partido', partido_opt)
    escolaridade = st.sidebar.selectbox('Escolaridade', escolaridade_opt)
    receita = st.sidebar.number_input('Receita mensal bruta')
    despesa = st.sidebar.number_input('Despesa total com campanha')
    submitted = st.form_submit_button(label='Submeter')

if submitted:
    arry_cand = np.array([[partido, idade, sexo, escolaridade, raca, receita, despesa]])
    candidato = pd.DataFrame(arry_cand, columns=['NM_PARTIDO','NR_IDADE_DATA_POSSE', 
    'DS_GENERO', 'DS_GRAU_INSTRUCAO', 'DS_COR_RACA', 'VR_RECEITA', 'VR_DESPESA_CONTRATADA'])       
    le = LabelEncoder()
    candidato['NM_PARTIDO']=le.fit_transform(candidato['NM_PARTIDO']) 
    candidato['DS_GENERO']=le.fit_transform(candidato['DS_GENERO']) 
    candidato['DS_GRAU_INSTRUCAO']=le.fit_transform(candidato['DS_GRAU_INSTRUCAO']) 
    candidato['DS_COR_RACA']=le.fit_transform(candidato['DS_COR_RACA']) 
    cand_encoded = candidato.to_numpy()
    st.text('Com os dados inseridos, a previsão é que o candidato seria:')
    st.text(getResult(cand_encoded))
    
   
    


     



