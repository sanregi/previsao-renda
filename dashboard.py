import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text


# Configuração da página
st.set_page_config(
    page_title="Projeto #02 | Previsão de Renda",
    page_icon="https://raw.githubusercontent.com/sanregi/previsao-renda/main/favicon.ico",
    layout="wide",
    initial_sidebar_state="auto",
)


# Barra lateral
st.sidebar.markdown('''
<div style="text-align:center">
<img src="https://github.com/sanregi/previsao-renda/blob/main/newebac_logo_black_half.png?raw=true" alt="ebac-logo" width=50%>
</div>

# **Profissão: Cientista de Dados**
### [**Projeto #02** | Previsão de Renda](https://github.com/sanregi/previsao-renda)

**Por:** [Regis Sandes](https://www.linkedin.com/in/regis-sandes/)<br>
**Data:** 03 de março de 2024.<br>
<!-- **Última atualização:** 03 de março de 2024. -->

---
''', unsafe_allow_html=True)

# Índice na barra lateral
with st.sidebar.expander("Índice", expanded=False):
    st.markdown('''
    - [Etapa 1 CRISP - DM: Entendimento do Negócio](#1)
    - [Etapa 2 Crisp-DM: Entendimento dos Dados](#2)
        > - [Dicionário de Dados](#dicionario)
        > - [Carregando os Pacotes](#pacotes)
        > - [Carregando os Dados](#dados)
        > - [Entendimento dos Dados - Univariada](#univariada)
        >> - [Estatísticas Descritivas das Variáveis Quantitativas](#describe)
        > - [Entendimento dos Dados - Bivariadas](#bivariada)
        >> - [Matriz de Correlação](#correlacao)
        >> - [Matriz de Dispersão](#dispersao)
        >>> - [Clustermap](#clustermap)
        >>> - [Linha de Tendência](#tendencia)
        >> - [Análise das Variáveis Qualitativas](#qualitativas)
    - [Etapa 3 Crisp-DM: Preparação dos Dados](#3)
    - [Etapa 4 Crisp-DM: Modelagem](#4)
        > - [Divisão da Base em Treino e Teste](#train_test)
        > - [Seleção de Hiperparâmetros do Modelo com For Loop](#for_loop)
        > - [Rodando o Modelo](#rodando)
    - [Etapa 5 Crisp-DM: Avaliação dos Resultados](#5)
    - [Etapa 6 Crisp-DM: Implantação](#6)
        > - [Simulação](#simulacao)
    ''', unsafe_allow_html=True)


# Bibliotecas/pacotes na barra lateral
with st.sidebar.expander("Bibliotecas/Pacotes", expanded=False):
    st.code('''
    import streamlit as st
    import io

    import numpy as np
    import pandas as pd

    from ydata_profiling import ProfileReport
    from streamlit_pandas_profiling import st_profile_report

    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor
    from sklearn import tree
    ''', language='python')



# Título principal
st.markdown('# <div style="text-align:center"> [Previsão de Renda](https://github.com/sanregi/previsao-renda) </div>',
            unsafe_allow_html=True)

st.divider()

# Etapa 1: Entendimento do Negócio
st.markdown('''
## Etapa 1 CRISP - DM: Entendimento do Negócio <a name="1"></a>
''', unsafe_allow_html=True)

st.markdown('''
Uma instituição financeira deseja compreender melhor o perfil de renda de seus novos clientes para diversas finalidades, como ajustar os limites de cartões de crédito dos novos clientes, sem a necessidade de solicitar comprovantes de renda. Para isso, realizou um estudo com alguns clientes, verificando suas rendas por meio de comprovantes de renda e outros documentos, e pretende construir um modelo preditivo para esta renda com base em algumas variáveis que já possui em seu banco de dados.
''')


# Etapa 2: Entendimento dos Dados
st.markdown('''
## Etapa 2 Crisp-DM: Entendimento dos Dados<a name="2"></a>
''', unsafe_allow_html=True)

# Dicionário de Dados
st.markdown('''
### Dicionário de Dados <a name="dicionario"></a>

| Variável              | Descrição                                                                                                  | Tipo             |
| --------------------- |:----------------------------------------------------------------------------------------------------------:| ----------------:|
| data_ref              | Data de referência de coleta das variáveis                                                                 | object           |
| id_cliente            | Código identificador exclusivo do cliente                                                                  | int              |
| sexo                  | Sexo do cliente (M = 'Masculino'; F = 'Feminino')                                                          | object (binária) |
| posse_de_veiculo      | Indica se o cliente possui veículo (True = 'Possui veículo'; False = 'Não possui veículo')                 | bool (binária)   |
| posse_de_imovel       | Indica se o cliente possui imóvel (True = 'Possui imóvel'; False = 'Não possui imóvel')                    | bool (binária)   |
| qtd_filhos            | Quantidade de filhos do cliente                                                                            | int              |
| tipo_renda            | Tipo de renda do cliente (Empresário, Assalariado, Servidor público, Pensionista, Bolsista)                | object           |
| educacao              | Grau de instrução do cliente (Primário, Secundário, Superior incompleto, Superior completo, Pós graduação) | object           |
| estado_civil          | Estado civil do cliente (Solteiro, União, Casado, Separado, Viúvo)                                         | object           |
| tipo_residencia       | Tipo de residência do cliente (Casa, Governamental, Com os pais, Aluguel, Estúdio, Comunitário)            | object           |
| idade                 | Idade do cliente em anos                                                                                   | int              |
| tempo_emprego         | Tempo no emprego atual                                                                                     | float            |
| qt_pessoas_residencia | Quantidade de pessoas que moram na residência                                                              | float            |
| **renda**             | Valor numérico decimal representando a renda do cliente em reais                                           | float            |
''', unsafe_allow_html=True)




st.markdown('''
### Carregando os dados 
''', unsafe_allow_html=True)

# Carregando os dados
renda = pd.read_csv("G:\Ebac 2\Modulo 16\MOG 16 ex 01\previsao_de_renda.csv")

# Exibindo informações sobre os dados
st.write('Quantidade total de linhas:', len(renda))
st.write('Quantidade de linhas duplicadas:', renda.duplicated().sum())
st.write('Quantidade após remoção das linhas duplicadas:', len(renda.drop_duplicates()))

# Removendo linhas duplicadas
renda.drop_duplicates(inplace=True, ignore_index=True)


# Exibindo informações sobre os dados após remoção de duplicatas
st.write('Quantidade total de linhas após remoção de duplicatas:', len(renda))

# Exibindo as primeiras linhas do dataframe
st.dataframe(renda.head())


st.markdown('''
### Entendimento dos dados - Univariada <a name="univariada"></a>
''', unsafe_allow_html=True)

# Análise univariada
with st.expander("Relatório Interativo de Análise Exploratória de Dados", expanded=True):
    prof = ProfileReport(df=renda,
                         minimal=False,
                         explorative=True,
                         dark_mode=True,
                         orange_mode=True)
    st_profile_report(prof)


# Estatísticas descritivas das variáveis quantitativas
st.markdown('''
#### Estatísticas Descritivas das Variáveis Quantitativas <a name="describe"></a>
''', unsafe_allow_html=True)
st.write(renda.describe().transpose())

# Análise bivariada
st.markdown('''
### Entendimento dos Dados - Bivariadas <a name="bivariada"></a>
''', unsafe_allow_html=True)

st.markdown('''
#### Matriz de correlação <a name="correlacao"></a>
''', unsafe_allow_html=True)


st.write((renda
          .iloc[:, 3:]
          .corr(numeric_only=True)
          .tail(n=1)
          ))



st.markdown('''
#### Matriz de dispersão <a name="dispersao"></a>
''', unsafe_allow_html=True)


sns.pairplot(data=renda,
             hue='tipo_renda',
             vars=['qtd_filhos',
                   'idade',
                   'tempo_emprego',
                   'qt_pessoas_residencia',
                   'renda'],
             diag_kind='kde')
st.pyplot(plt)

st.markdown('Ao analisar o *pairplot*, que consiste na matriz de dispersão, é possível identificar alguns *outliers* na variável `renda`, os quais podem afetar o resultado da análise de tendência, apesar de ocorrerem com baixa frequência. Além disso, é observada uma baixa correlação entre praticamente todas as variáveis quantitativas, reforçando os resultados obtidos na matriz de correlação.')

st.markdown('''
##### Clustermap <a name="clustermap"></a>
''', unsafe_allow_html=True)

cmap = sns.diverging_palette(h_neg=100,
                             h_pos=359,
                             as_cmap=True,
                             sep=1,
                             center='light')
ax = sns.clustermap(data=renda.corr(numeric_only=True),
                    figsize=(10, 10),
                    center=0,
                    cmap=cmap)
plt.setp(ax.ax_heatmap.get_xticklabels(), rotation=45)
st.pyplot(plt.gcf())  # Para resolver o aviso PyplotGlobalUseWarning



st.markdown('Com o *clustermap*, é possível reforçar novamente os resultados de baixa correlação com a variável `renda`. Apenas a variável `tempo_emprego` apresenta um índice considerável para análise.')

st.markdown('''
#####  Linha de tendência <a name="tendencia"></a>
''', unsafe_allow_html=True)

plt.figure(figsize=(16, 9))
sns.scatterplot(x='tempo_emprego',
                y='renda',
                hue='tipo_renda',
                size='idade',
                data=renda,
                alpha=0.4)
sns.regplot(x='tempo_emprego',
            y='renda',
            data=renda,
            scatter=False,
            color='.3')
st.pyplot(plt.gcf())  # Para resolver o aviso PyplotGlobalUseWarning


st.markdown('Embora a correlação entre a variável `tempo_emprego` e a variável `renda` não seja tão alta, é possível identificar facilmente a covariância positiva com a inclinação da linha de tendência.')

st.markdown('''
#### Análise das variáveis qualitativas <a name="qualitativas"></a>
''', unsafe_allow_html=True)


with st.expander("Análise de relevância preditiva com variáveis booleanas", expanded=True):
    plt.rc('figure', figsize=(12, 4))
    fig, axes = plt.subplots(nrows=1, ncols=2)
    sns.pointplot(x='posse_de_imovel',
                  y='renda',
                  data=renda,
                  dodge=True,
                  ax=axes[0])
    sns.pointplot(x='posse_de_veiculo',
                  y='renda',
                  data=renda,
                  dodge=True,
                  ax=axes[1])
    st.pyplot(plt)

    st.markdown('Ao comparar os gráficos acima, nota-se que a variável `posse_de_veículo` apresenta maior relevância na predição de renda, evidenciada pela maior distância entre os intervalos de confiança para aqueles que possuem e não possuem veículo, ao contrário da variável `posse_de_imóvel` que não apresenta diferença significativa entre as possíveis condições de posse imobiliária.')



with st.expander("Análise das Variáveis Qualitativas ao Longo do Tempo", expanded=True):
    renda['data_ref'] = pd.to_datetime(arg=renda['data_ref'])
    qualitativas = renda.select_dtypes(include=['object', 'boolean']).columns
    plt.rc('figure', figsize=(16, 4))
    for col in qualitativas:
        fig, axes = plt.subplots(nrows=1, ncols=2)
        fig.subplots_adjust(wspace=.6)
        tick_labels = renda['data_ref'].map(
            lambda x: x.strftime('%b/%Y')).unique()
        # Gráficos de barras empilhadas:
        renda_crosstab = pd.crosstab(index=renda['data_ref'],
                                     columns=renda[col],
                                     normalize='index')
        ax0 = renda_crosstab.plot.bar(stacked=True,
                                      ax=axes[0])
        ax0.set_xticklabels(labels=tick_labels, rotation=45)
        axes[0].legend(bbox_to_anchor=(1, .5), loc=6, title=f"'{col}'")
        # Gráficos de perfis médios no tempo:
        ax1 = sns.pointplot(x='data_ref', y='renda', hue=col,
                            data=renda, dodge=True, ci=95, ax=axes[1])
        ax1.set_xticklabels(labels=tick_labels, rotation=45)
        axes[1].legend(bbox_to_anchor=(1, .5), loc=6, title=f"'{col}'")
        st.pyplot(plt)


st.markdown('''
## Etapa 3 Crisp-DM: Preparação dos Dados<a name="3"></a>
''', unsafe_allow_html=True)

# Removendo a coluna 'data_ref'
renda.drop(columns='data_ref', inplace=True)

# Removendo linhas com valores ausentes
renda.dropna(inplace=True)

# Criando uma tabela para mostrar informações sobre os dados
st.table(pd.DataFrame(index=renda.nunique().index,
                      data={'tipos_dados': renda.dtypes,
                            'qtd_valores': renda.notna().sum(),
                            'qtd_categorias': renda.nunique().values}))


with st.expander("Conversão das Variáveis Categóricas em Variáveis Numéricas (Dummies)", expanded=True):
    # Convertendo variáveis categóricas em variáveis dummy
    renda_dummies = pd.get_dummies(data=renda)
    
    # Exibindo informações sobre o dataframe com dummies
    buffer = io.StringIO()
    renda_dummies.info(buf=buffer)
    st.text(buffer.getvalue())

    # Tabela de correlação das variáveis dummy com a variável alvo 'renda'
    st.table((renda_dummies.corr()['renda']
              .sort_values(ascending=False)
              .to_frame()
              .reset_index()
              .rename(columns={'index': 'Variável', 'renda': 'Correlação'})
              .style.bar(color=['darkred', 'darkgreen'], align='zero')
              ))


st.markdown('''
## Etapa 4 Crisp-DM: Modelagem <a name="4"></a>
''', unsafe_allow_html=True)


st.markdown('Para a modelagem, optou-se pela utilização do algoritmo DecisionTreeRegressor, devido à sua capacidade de lidar com problemas de regressão, como a previsão de renda dos clientes. Além disso, as árvores de decisão são conhecidas por sua interpretabilidade e capacidade de identificar os atributos mais relevantes para a previsão da variável-alvo, tornando-as uma escolha adequada para o projeto.')


st.markdown('''
### Divisão da Base em Treino e Teste <a name="train_test"></a>
''', unsafe_allow_html=True)


X = renda_dummies.drop(columns='renda')
y = renda_dummies['renda']

st.write('Quantidade de linhas e colunas de X:', X.shape)
st.write('Quantidade de linhas de y:', len(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

st.write('Tamanhos dos conjuntos de treino e teste:')
st.write('X_train:', X_train.shape)
st.write('X_test:', X_test.shape)
st.write('y_train:', y_train.shape)
st.write('y_test:', y_test.shape)

st.markdown('''
### Seleção de Hiperparâmetros do Modelo com For Loop <a name="for_loop"></a>
''', unsafe_allow_html=True)



# Importar as bibliotecas necessárias
from sklearn.tree import DecisionTreeRegressor
import pandas as pd

# Criar um DataFrame para armazenar os resultados dos testes
score = pd.DataFrame(columns=['max_depth', 'min_samples_leaf', 'score'])

# Realizar testes para diferentes combinações de parâmetros
for max_depth in range(1, 21):
    for min_samples_leaf in range(1, 31):
        # Instanciar o modelo de árvore de decisão com os parâmetros definidos
        reg_tree = DecisionTreeRegressor(random_state=42,
                                         max_depth=max_depth,
                                         min_samples_leaf=min_samples_leaf)
        # Treinar o modelo com os dados de treino
        reg_tree.fit(X_train, y_train)
        # Avaliar o desempenho do modelo nos dados de teste e armazenar os resultados
        score = pd.concat([score,
                           pd.DataFrame({'max_depth': [max_depth],
                                         'min_samples_leaf': [min_samples_leaf],
                                         'score': [reg_tree.score(X=X_test, y=y_test)]})],
                          ignore_index=True)

# Exibir os resultados ordenados por score
st.dataframe(score.sort_values(by='score', ascending=False))

# Exibir a marcação para a seção de rodar o modelo
st.markdown('''
### Rodando o Modelo <a name="rodando"></a>
''', unsafe_allow_html=True)


from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text

# Instanciar o modelo de árvore de decisão com os parâmetros definidos
reg_tree = DecisionTreeRegressor(random_state=42,
                                 max_depth=8,
                                 min_samples_leaf=4)

# Treinar o modelo e exibir os detalhes do treinamento
details = reg_tree.fit(X_train, y_train)
st.text(details)

# Visualização gráfica da árvore com plot_tree
with st.expander("Visualização Gráfica da Árvore com plot_tree", expanded=True):
    plt.figure(figsize=(18, 9))
    plot_tree(decision_tree=reg_tree,
              feature_names=X.columns.tolist(),  # Converter o Index em uma lista
              filled=True)
    st.pyplot(plt)

# Visualização impressa da árvore
with st.expander("Visualização Impressa da Árvore", expanded=False):
    text_tree_print = export_text(decision_tree=reg_tree,
                                  feature_names=X.columns.tolist())  # Converter o Index em uma lista
    st.text(text_tree_print)



st.markdown('''
## Etapa 5 Crisp-DM: Avaliação dos Resultados <a name="5"></a>
''', unsafe_allow_html=True)

r2_train = reg_tree.score(X=X_train, y=y_train)
r2_test = reg_tree.score(X=X_test, y=y_test)

template = 'O coeficiente de determinação (𝑅²) da árvore com profundidade = {0} para a base de {1} é: {2:.2f}'
st.write(template.format(reg_tree.get_depth(), 'treino', r2_train).replace(".", ","))
st.write(template.format(reg_tree.get_depth(), 'teste', r2_test).replace(".", ","))

renda['renda_predict'] = np.round(reg_tree.predict(X), 2)
st.dataframe(renda[['renda', 'renda_predict']])




st.markdown('''
## Etapa 6 Crisp-DM: Implantação] <a name="6"></a>
''', unsafe_allow_html=True)


### Nessa etapa colocamos em uso o modelo desenvolvido, normalmente implementando o modelo desenvolvido em um motor que toma as decisões com algum nível de automação.



# Título da página
st.title("Simulador de Previsão de Renda")

# Cabeçalho
st.header("Preencha as informações abaixo:")

# Formulário
with st.form("form_simulacao"):
    # Campos do formulário
    st.subheader("Informações Pessoais:")
    sexo = st.radio("Sexo", ('Masculino', 'Feminino'))
    idade = st.slider("Idade", 18, 100)

    st.subheader("Situação Residencial:")
    tipo_residencia = st.selectbox("Tipo de Residência", [
                                   'Casa', 'Apartamento', 'Com os Pais', 'Aluguel', 'Outro'])

    qtd_pessoas_residencia = st.number_input(
        "Número de Pessoas na Residência", 1, 15)

    st.subheader("Situação Profissional:")
    tipo_renda = st.selectbox("Tipo de Renda", [
                              'Sem Renda', 'Empresário', 'Assalariado', 'Servidor Público', 'Pensionista', 'Bolsista'])

    tempo_emprego = st.slider(
        "Tempo de Emprego Atual (em anos)", 0, 50)

    # Botão para submeter o formulário
    submitted = st.form_submit_button("Simular")
    
    if submitted:
        # Preparação dos dados para a previsão
        entrada = pd.DataFrame([{'sexo': sexo,
                                 'posse_de_veiculo': veiculo,
                                 'posse_de_imovel': imovel,
                                 'qtd_filhos': filhos,
                                 'tipo_renda': tiporenda,
                                 'educacao': educacao,
                                 'estado_civil': estadocivil,
                                 'tipo_residencia': residencia,
                                 'idade': idade,
                                 'tempo_emprego': tempoemprego,
                                 'qt_pessoas_residencia': qtdpessoasresidencia}])

        # Realização da previsão e exibição do resultado
        renda_estimada = np.round(reg_tree.predict(entrada).item(), 2)
        st.write(f"Renda estimada: R$ {renda_estimada:.2f}")




'---'

