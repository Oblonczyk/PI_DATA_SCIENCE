# ===============================
# 📦 Importação das Dependências
# ===============================

# 🔢 Manipulação de Dados
import pandas as pd
import numpy as np
import time

# 📊 Visualização de Dados
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# 🤖 Machine Learning - Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    silhouette_score,
    r2_score,
    mean_squared_error
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# 🧠 Aprendizado de Máquina - Balanceamento de Dados
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# 📚 Processamento de Linguagem Natural (PLN)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# ⚡ Big Data com PySpark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.clustering import KMeans as SparkKMeans
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# 📥 Carregar Base de Dados
#df = pd.read_csv(r"C:/Users/bruno/Downloads/PI_DATA_SCIENCE/Analise_BIGDATA_docker/ANALISE_PI/alunos_com_erros.csv")

# 📊 Exploração Inicial
#print(df.head())
#print(df.info())
#print(df.isnull().sum())

# ================================
# 📊 Visualização Inicial com mysql

from sqlalchemy import create_engine
import pandas as pd

# Configuração da conexão
usuario = 'root'
senha = ''
host = 'localhost'
porta = 3306
nome_banco = 'faculdade'

# String de conexão
engine = create_engine(f'mysql+pymysql://{usuario}:{senha}@{host}:{porta}/{nome_banco}')


# Consulta os dados da tabela alunos
query = "SELECT * FROM alunos"

df = pd.read_sql(query, con=engine)

# Conferir os dados
print(df.head())
print(df.info())
print(df.isnull().sum())
# Verificar os tipos de dados
print(df.dtypes)
# Verificar os valores únicos de cada coluna
for col in df.columns:
    print(f"Valores únicos em {col}: {df[col].unique()}")
    

# =================================

# ===============================
# 📌 Pré-processamento (Pandas)
# ===============================

# Remover registros com valores nulos e criar cópia
df_model = df.dropna().copy()

# Label Encoding para colunas categóricas (exceto target)
label_encoders = {}
for c in df_model.select_dtypes(include='object').columns:
    if c != 'risco_evasao':
        le = LabelEncoder()
        df_model.loc[:, c] = le.fit_transform(df_model[c].astype(str))
        label_encoders[c] = le

# Separar features e target
X = df_model.drop(columns=['risco_evasao'])
y = df_model['risco_evasao']

# Codificar target
le_y = LabelEncoder()
y_encoded = le_y.fit_transform(y)

# Normalizar features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# 📊 Modelo de Classificação (Random Forest)
# ===============================

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Relatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=le_y.classes_.astype(str)))

# ===============================
# 📉 Regressão Linear
# ===============================
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import pandas as pd
from sqlalchemy import create_engine

# Conectar ao banco e carregar dados no pandas
engine = create_engine("mysql+pymysql://root:@localhost:3306/faculdade")
df_model = pd.read_sql("SELECT * FROM alunos", con=engine)

if 'desempenho_1' in df_model.columns:
    X_reg = df_model.drop(columns=['desempenho_1'], errors='ignore')
    y_reg = df_model['desempenho_1']

    # Remover colunas de alta cardinalidade
    high_card_cols = ['nome_aluno', 'email', 'professor_1', 'professor_2', 'professor_3', 'professor_4', 'professor_5']
    X_reg = X_reg.drop(columns=high_card_cols, errors='ignore')

    # Transformar variáveis categóricas em numéricas
    X_reg_encoded = pd.get_dummies(X_reg, drop_first=True)

    # Codificar target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_reg.astype(str))

    # Dividir dados
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg_encoded, y_encoded, test_size=0.2, random_state=42
    )

    # Pipeline com Ridge Regression (mais rápido)
    pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('regressor', Ridge())
    ])

    # Treinar pipeline
    pipeline.fit(X_train_r, y_train_r)

    # Fazer predições
    y_pred_r = pipeline.predict(X_test_r)

    # Avaliar o modelo
    print("R²:", r2_score(y_test_r, y_pred_r))
    print("MSE:", mean_squared_error(y_test_r, y_pred_r))
    print("Formato final das features:", X_reg_encoded.shape)

    print("Classes desempenho_1 codificadas:", list(le.classes_))
else:
    print("Coluna desempenho_1 não encontrada na tabela.")
# ===============================
# fim Regressão Linear
# ===============================

# ===============================
# 📈 Clusterização (KMeans + PCA)
# ===============================

# Clusterização
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

print("Silhouette Score:", silhouette_score(X_scaled, clusters))

# PCA para 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_plot = pd.DataFrame({
    'PCA1': X_pca[:, 0],
    'PCA2': X_pca[:, 1],
    'Cluster': clusters
})

# Gráfico interativo
fig = px.scatter(
    df_plot, x='PCA1', y='PCA2',
    color=df_plot['Cluster'].astype(str),
    title='Clusters dos Alunos (KMeans + PCA)',
    color_discrete_sequence=px.colors.qualitative.Set2
)

fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
fig.show()

# ✅ Variáveis úteis: agora com dados das 5 aulas
features = [
    'sexo', 'idade', 'trabalha', 'renda_familiar', 'acompanhamento_medico',
    'tem_filho', 'estado_civil', 'curso', 'status', 'turma',
    'semestre', 'bimestre',
    # Aulas 1 a 5
    'aula_1', 'professor_1', 'notas_1', 'faltas_1',
    'aula_2', 'professor_2', 'notas_2', 'faltas_2',
    'aula_3', 'professor_3', 'notas_3', 'faltas_3',
    'aula_4', 'professor_4', 'notas_4', 'faltas_4',
    'aula_5', 'professor_5', 'notas_5', 'faltas_5'
]
target = 'desempenho_1'  # ou mude para 'risco_evasao' ou 'desempenho_final', se desejar

# 🔍 Eliminar registros com target ausente
df_valid = df.dropna(subset=[target])
X = df_valid[features]
y = df_valid[target]

# ⚙️ Identificar tipos de variáveis
cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64', 'bool']).columns.tolist()

# 🔧 Pipelines de transformação
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
    ('cat', cat_pipeline, cat_cols),
    ('num', num_pipeline, num_cols)
])

# 🧪 Dividir treino/teste com estratificação
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 🤖 Pipeline com SMOTE e Random Forest
model = ImbPipeline(steps=[
    ('preprocessing', preprocessor),
    ('oversample', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
])

# 📈 Treinar e Avaliar
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("📋 Relatório Final (5 Aulas):")
print(classification_report(y_test, y_pred))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.figure(1, figsize=(12,6))
plt.title("Matriz de Confusão - Modelo com 5 Aulas")

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

# ===============================
# 🔧 Iniciar sessão Spark (com Hadoop se quiser)
# ===============================
spark = SparkSession.builder \
    .appName("PrevisaoRiscoEvasao") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
    .config("spark.jars", "C:/caminho/mysql-connector-java-8.0.33.jar") \
    .getOrCreate()

# ===============================
# 📥 Leitura dos dados via JDBC
# ===============================
df = spark.read.format("jdbc").options(
    url="jdbc:mysql://localhost:3306/faculdade",
    driver="com.mysql.cj.jdbc.Driver",
    dbtable="alunos",
    user="root",
    password=""
).load()

df.printSchema()
df.show(5)

# ===============================
# 🧼 Limpeza e conversão
# ===============================
colunas_numericas = ["idade", "renda_familiar", "faltas_1", "notas_1", "faltas_2", "notas_2",
                     "faltas_3", "notas_3", "faltas_4", "notas_4", "faltas_5", "notas_5"]
colunas_categoricas = ["sexo", "trabalha", "acompanhamento_medico", "tem_filho",
                       "estado_civil", "desempenho_1", "desempenho_2",
                       "desempenho_3", "desempenho_4", "desempenho_5", "risco_evasao"]

for coln in colunas_numericas:
    df = df.withColumn(coln, col(coln).cast("double")).na.fill({coln: -1})

for colc in colunas_categoricas:
    df = df.na.fill({colc: "desconhecido"})
    indexer = StringIndexer(inputCol=colc, outputCol=colc + "_idx", handleInvalid="keep")
    df = indexer.fit(df).transform(df)

# ===============================
# 🔄 Vetorização
# ===============================
feature_cols = colunas_numericas + [col + "_idx" for col in colunas_categoricas[:-1]]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df = assembler.transform(df)

# ===============================
# 🌳 Random Forest Classifier
# ===============================
rf = RandomForestClassifier(labelCol="risco_evasao_idx", featuresCol="features")
model = rf.fit(df)
predictions = model.transform(df)

# ===============================
# 📊 Avaliação
# ===============================
evaluator = MulticlassClassificationEvaluator(
    labelCol="risco_evasao_idx", predictionCol="prediction", metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)
print(f"Acurácia do modelo: {accuracy:.4f}")

# ===============================
# 📈 Visualizações
# ===============================
preds_pd = predictions.select("risco_evasao_idx", "prediction").toPandas()

# Gráfico 1: Distribuição Real vs Predito
plt.figure(1, figsize=(10, 5))
sns.countplot(data=preds_pd, x="risco_evasao_idx", color="blue", label="Real", alpha=0.6)
sns.countplot(data=preds_pd, x="prediction", color="red", label="Predito", alpha=0.6)
plt.title("Distribuição das Classes - Real vs Predita")
plt.xlabel("Classe (risco_evasao)")
plt.ylabel("Contagem")
plt.legend()
plt.grid(True)

# Gráfico 2: Matriz de Confusão
conf_mat = confusion_matrix(preds_pd["risco_evasao_idx"], preds_pd["prediction"])
plt.figure(2, figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusão")
plt.xlabel("Predito")
plt.ylabel("Real")

plt.show()

# ===============================
# 📄 Informações adicionais
# ===============================
print("\nInformações do DataFrame Spark:")
print("Colunas disponíveis:", df.columns)
print("Contagem de linhas:", df.count())
print("Contagem de valores nulos por coluna:")
df.select([col(c).isNull().cast("int").alias(c) for c in df.columns]) \
  .agg(*[sum(col(c)).alias(c) for c in df.columns]).show()

print("Distribuição de classes (risco_evasao):")
df.groupBy("risco_evasao").count().show()

print("Distribuição de classes (risco_evasao_idx):")
df.groupBy("risco_evasao_idx").count().show()
print("Distribuição de classes (risco_evasao_idx):")
df.groupBy("risco_evasao_idx").count().show()
print("Distribuição de classes (risco_evasao):")
df.groupBy("risco_evasao").count().show()

# ===============================
# 📈 Visualizações com Pandas, Seaborn e Matplotlib
# ===============================
preds_pd = predictions.select("risco_evasao_idx", "prediction").toPandas()

# Gráfico 1: Real vs Predito
plt.figure(1, figsize=(10, 5))
sns.countplot(data=preds_pd, x="risco_evasao_idx", color="blue", label="Real", alpha=0.6)
sns.countplot(data=preds_pd, x="prediction", color="red", label="Predito", alpha=0.6)
plt.title("Distribuição das Classes - Real vs Predita")
plt.xlabel("Classe (risco_evasao_idx)")
plt.ylabel("Contagem")
plt.legend()
plt.grid(True)

# Gráfico 2: Matriz de Confusão
conf_mat = confusion_matrix(preds_pd["risco_evasao_idx"], preds_pd["prediction"])
plt.figure(2, figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusão")
plt.xlabel("Predito")
plt.ylabel("Real")

plt.show()
# Gráfico 3: Acurácia do Modelo 
labels = ['Acurácia', 'Erro']
sizes = [accuracy, 1 - accuracy]
plt.figure(3, figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#F44336'])
plt.title('Acurácia do Modelo de Evasão')
plt.show()

# ===============================
# 📊 Converter para Pandas e gerar gráfico
# ===============================

# Converter para Pandas
df_pandas = df_spark.toPandas()

# Tratar valores nulos
df_pandas['risco_evasao'] = df_pandas['risco_evasao'].fillna('Desconhecido')

plt.figure(7, figsize=(8, 5))
sns.countplot(data=df_pandas, x='risco_evasao', palette='viridis')
plt.title('Distribuição de Risco de Evasão (Dados via HDFS)')
plt.xlabel('Risco de Evasão')
plt.ylabel('Quantidade')
plt.xticks(rotation=45)
plt.tight_layout()

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(8, figsize=(8,6))
sns.countplot(x='risco_evasao', data=df_model, palette='viridis')
plt.title('Previsões de Risco de Evasão pelo Modelo')
plt.xlabel('Risco Predito')
plt.ylabel('Quantidade')
plt.xticks(rotation=45)
plt.tight_layout()

# ===============================
# 📋 Previsões Spark para análise detalhada
# ===============================

# Mapear as classes numéricas para texto (exemplo)
map_labels = {0: "Baixo", 1: "Médio", 2: "Alto"}

# Converter previsões Spark para Pandas
predictions_pandas = predictions.select("aluno_id", "prediction").toPandas()
predictions_pandas['risco_predito'] = predictions_pandas['prediction'].map(map_labels)

# Ajustar tipos para merge
df_model['aluno_id'] = df_model['aluno_id'].astype(str)
predictions_pandas['aluno_id'] = predictions_pandas['aluno_id'].astype(str)

# Remover coluna risco_predito anterior para evitar conflito
if 'risco_predito' in df_model.columns:
    df_model = df_model.drop(columns=['risco_predito'])

# Merge das previsões Spark no df_model
df_model = df_model.merge(predictions_pandas[['aluno_id', 'risco_predito']], on='aluno_id', how='inner')

print("Colunas após merge:", df_model.columns)
print(df_model.head())

# Filtrar alunos de alto risco
alunos_alto_risco = df_model[df_model['risco_predito'] == 'Alto']
print(alunos_alto_risco[['nome_aluno', 'desempenho_1', 'faltas_1', 'notas_1']].head())

# Gráfico dos 30 primeiros alunos e seu risco predito
plt.figure(9, figsize=(12,6))
sns.countplot(data=df_model.head(30), x='nome_aluno', hue='risco_predito', palette='Set2')
plt.xticks(rotation=90)
plt.title("Risco predito dos 30 primeiros alunos")
plt.xlabel("Nome do Aluno")
plt.ylabel("Contagem")
plt.legend(title='Risco Predito')
plt.tight_layout()
plt.show()

# Gráfico percentual dos riscos preditos
# Calcular os percentuais
risco_counts = df_model['risco_predito'].value_counts(normalize=True) * 100

# Converter para DataFrame para plotar
df_risco_counts = risco_counts.reset_index()
df_risco_counts.columns = ['risco_predito', 'percentual']

# Gráfico interativo
fig = px.bar(
    df_risco_counts,
    x='risco_predito',
    y='percentual',
    color='risco_predito',
    text='percentual',
    color_discrete_sequence=px.colors.qualitative.Set1,
    title='Distribuição percentual dos riscos preditos'
)

fig.update_layout(
    yaxis_title="Porcentagem (%)",
    xaxis_title="Risco Predito",
    showlegend=False
)

fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')

fig.show()

# Variáveis (features) que vai usar para treinar (sem identificadores e target)
features_all = [
    "curso", "status", "turma", "sexo", "idade", "trabalha", "renda_familiar", "acompanhamento_medico",
    "tem_filho", "estado_civil", "semestre", "bimestre",
    "aula_1", "professor_1", "notas_1", "faltas_1", "desempenho_1",
    "aula_2", "professor_2", "notas_2", "faltas_2", "desempenho_2",
    "aula_3", "professor_3", "notas_3", "faltas_3", "desempenho_3",
    "aula_4", "professor_4", "notas_4", "faltas_4", "desempenho_4",
    "aula_5", "professor_5", "notas_5", "faltas_5", "desempenho_5"
]

# Preprocessar seu dataframe (exemplo rápido, ajuste conforme seu caso):
X = df_model[features_all].copy()
y = df_model['risco_evasao']

# Se tem variáveis categóricas, transforme em numérico (LabelEncoder ou get_dummies)
X_encoded = pd.get_dummies(X, drop_first=True)  # codifica as categorias

# Treina modelo RandomForest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_encoded, y)

# Extrai importância das features
importances = rf.feature_importances_

# Nomes das features após o get_dummies
features_encoded_names = X_encoded.columns.tolist()

# Organizar para o gráfico
indices = np.argsort(importances)[::-1]
df_imp = pd.DataFrame({
    'Variável': [features_encoded_names[i] for i in indices],
    'Importância': importances[indices]
})

# Gráfico interativo
fig = px.bar(
    df_imp.head(10),  # mostra top 30 para não ficar poluído
    x='Importância',
    y='Variável',
    orientation='h',
    color='Importância',
    color_continuous_scale='Viridis',
    title='Importância das Variáveis para Predição do Risco de Evasão',
    labels={'Importância':'Importância da Feature', 'Variável':'Feature'}
)

fig.update_layout(yaxis=dict(categoryorder='total ascending'))
fig.show()

# Copiar e preparar dados, garantir que coluna de evasão está presente e limpa
df_plot = df.copy().dropna(subset=['renda_familiar', 'risco_evasao'])

# Criar categorias econômicas baseadas na renda familiar
df_plot['classe_economica'] = pd.cut(
    df_plot['renda_familiar'],
    bins=[-np.inf, 4000, 8000, np.inf],
    labels=['Baixa', 'Média', 'Alta']
)

# Assumindo que 'risco_evasao' é 1 para evadiu e 0 para não evadiu
# Filtrar apenas alunos que evadiram
df_evadidos = df_plot[df_plot['risco_evasao'] == 1]

# Contar número de evadidos por classe econômica
df_contagem = df_evadidos.groupby('classe_economica').size().reset_index(name='quantidade_evadidos')

# Se quiser mostrar barras mesmo para classes sem evadidos, garantimos todas as categorias
todas_classes = ['Baixa', 'Média', 'Alta']
df_contagem = df_contagem.set_index('classe_economica').reindex(todas_classes, fill_value=0).reset_index()

# Gráfico interativo
fig = px.bar(
    df_contagem,
    x='classe_economica',
    y='quantidade_evadidos',
    labels={'classe_economica': 'Classe Econômica', 'quantidade_evadidos': 'Quantidade de Alunos Evadidos'},
    title='Quantidade de Alunos Evadidos por Classe Econômica (Renda Familiar)',
    color='classe_economica',
    color_discrete_map={'Baixa': 'red', 'Média': 'orange', 'Alta': 'green'},
    width=700,
    height=500
)

fig.update_layout(
    yaxis=dict(title='Quantidade de Alunos Evadidos', dtick=1),
    xaxis=dict(title='Classe Econômica'),
    showlegend=False,
    hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial")
)

fig.show()

# Preparar dados: filtrar faltas e evasão
df_faltas = df.copy().dropna(subset=['faltas_1', 'risco_evasao'])

# Filtrar apenas alunos que evadiram
df_evadidos = df_faltas[df_faltas['risco_evasao'] == 1]

# Contar evadidos por número de faltas
df_faltas_count = df_evadidos.groupby('faltas_1').size().reset_index(name='quantidade_evadidos')

# Ordenar por número de faltas para visualização mais clara
df_faltas_count = df_faltas_count.sort_values(by='faltas_1')

# Gráfico interativo
fig = px.bar(
    df_faltas_count,
    x='faltas_1',
    y='quantidade_evadidos',
    labels={'faltas_1': 'Número de Faltas', 'quantidade_evadidos': 'Quantidade de Alunos Evadidos'},
    title='Quantidade de Alunos Evadidos por Número de Faltas',
    width=800,
    height=500
)

fig.update_layout(
    xaxis=dict(dtick=1),  # mostrar cada número de falta no eixo x
    yaxis=dict(dtick=1),
    hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial")
)

fig.show()

# 1. Preparar os dados
df_evasao = df.copy().dropna(subset=['semestre', 'risco_evasao'])

# Filtrar apenas alunos com risco de evasão
df_evasao = df_evasao[df_evasao['risco_evasao'] == 1]

# 2. Contar evasões por semestre
evasao_por_semestre = df_evasao['semestre'].value_counts().reset_index()
evasao_por_semestre.columns = ['semestre', 'total_evasoes']

# 3. Calcular porcentagem do total
total_alunos_por_semestre = df['semestre'].value_counts().reset_index()
total_alunos_por_semestre.columns = ['semestre', 'total_alunos']

dados_plot = evasao_por_semestre.merge(total_alunos_por_semestre, on='semestre')
dados_plot['percentual_evasao'] = (dados_plot['total_evasoes'] / dados_plot['total_alunos'] * 100).round(1)

# 4. Criar gráfico interativo
fig = px.bar(
    dados_plot,
    x='semestre',
    y=['total_evasoes', 'percentual_evasao'],
    barmode='group',
    title='Distribuição de Evasão por Semestre',
    labels={
        'semestre': 'Semestre',
        'value': 'Contagem/Percentual',
        'variable': 'Métrica'
    },
    color_discrete_sequence=['#e74c3c', '#3498db'],
    text_auto=True,
    width=900,
    height=500
)

# 5. Personalizar layout
fig.update_layout(
    xaxis=dict(
        title='Semestre',
        type='category',
        tickmode='linear'
    ),
    yaxis_title="Valor",
    legend_title="Métrica",
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="white",
        font_size=12,
        font_family="Arial"
    )
)

# 6. Renomear legendas
fig.for_each_trace(lambda t: t.update(name='Total de Evasões' if t.name == 'total_evasoes' else 'Percentual de Evasão (%)'))

# 7. Adicionar informações complementares
fig.update_traces(
    textposition='outside',
    hovertemplate=(
        "<b>Semestre %{x}</b><br><br>" +
        "Total de evasões: %{y}<br>" +
        "Percentual: %{customdata[0]}%<extra></extra>"
    ),
    customdata=dados_plot[['percentual_evasao']]
)

fig.show()

# ===============================
# 📊 PCA + RandomForest: Visualização interativa
# ===============================

df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['Cluster'] = clusters
df_pca['Risco_Predito'] = df_model['risco_predito']

fig = px.scatter(
    df_pca, x='PC1', y='PC2', color='Risco_Predito',
    title='Alunos por Risco de Evasão (PCA + RandomForest)',
    labels={'PC1':'Componente Principal 1', 'PC2':'Componente Principal 2'}
)
fig.show()

# --- Exemplo: definir motivo de evasão com base em regras simples ---
# Se tiver a coluna 'motivo_evasao' use ela, senão a gente simula baseada nas variáveis
def definir_motivo(row):
    if row['faltas_1'] > 10:
        return 'Faltas Excessivas'
    elif row['desempenho_1'] < 5:
        return 'Baixo Desempenho'
    elif row['renda_familiar'] < 1500:
        return 'Baixa Renda'
    else:
        return 'Outro'

# Criar coluna de motivo (caso não exista)
if 'motivo_evasao' not in df_model.columns:
    df_model['motivo_evasao'] = df_model.apply(definir_motivo, axis=1)

# --- Contagem dos motivos por risco predito ---
df_motivos = df_model.groupby(['risco_predito', 'motivo_evasao']).size().reset_index(name='quantidade')

# --- Gráfico interativo com Plotly ---
fig = px.bar(
    df_motivos,
    x='motivo_evasao',
    y='quantidade',
    color='risco_predito',
    barmode='group',
    text='quantidade',
    title='Distribuição dos Motivos de Evasão por Categoria de Risco',
    labels={'motivo_evasao': 'Motivo de Evasão', 'quantidade': 'Número de Alunos'}
)

fig.update_layout(
    xaxis_title='Motivo de Evasão',
    yaxis_title='Número de Alunos',
    legend_title='Risco Predito',
    bargap=0.3
)

fig.show()

# 1. Preparar os dados - contar alunos com risco de evasão por curso
risco_por_curso = df[df['risco_evasao'] == 1]['curso'].value_counts().nlargest(5)

# 2. Configurar o estilo do gráfico
plt.figure(10, figsize=(10, 6))
sns.set_style("whitegrid")
paleta = sns.color_palette("Reds_r", len(risco_por_curso))

# 3. Criar o gráfico de barras
ax = sns.barplot(
    x=risco_por_curso.values,
    y=risco_por_curso.index,
    palette=paleta,
    edgecolor='black',
    linewidth=1
)

# 4. Adicionar rótulos e títulos
plt.figure(11, figsize=(8,6))
plt.title('Top 5 Cursos com Mais Alunos em Risco de Evasão', fontsize=16, pad=20)
plt.xlabel('Número de Alunos em Risco', fontsize=12)
plt.ylabel('Curso', fontsize=12)

# 5. Adicionar os valores nas barras
for i, valor in enumerate(risco_por_curso.values):
    ax.text(valor + 0.5, i, f'{valor}', va='center', fontsize=11, color='black')

# 6. Adicionar linha de contexto (total de alunos por curso)
total_por_curso = df['curso'].value_counts()[risco_por_curso.index]
for i, (curso, total) in enumerate(total_por_curso.items()):
    ax.text(risco_por_curso[curso] + 0.5, i-0.2,
            f'({risco_por_curso[curso]/total:.1%} do total)',
            va='center', fontsize=10, color='blue')

# 7. Ajustar layout e mostrar
plt.figure(12, figsize=(12,6))
plt.xlim(0, risco_por_curso.max() * 1.15)
plt.tight_layout()
plt.show()


# Verifica se a coluna existe
if 'risco_predito' in df_model.columns and 'risco_evasao' in df_model.columns:
    # Tabela cruzada com percentual por linha (risco_predito)
    ct = pd.crosstab(df_model['risco_predito'], df_model['risco_evasao'], normalize='index') * 100

    # Ordena os índices se desejar (opcional)
    ct = ct.sort_index()

    plt.figure(12, figsize=(8,6))
    sns.heatmap(ct, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Percentual (%)'})

    plt.title("Percentual de Evasão Real por Categoria de Risco Predito")
    plt.ylabel("Risco Predito")
    plt.xlabel("Evasão Real")
    plt.tight_layout()
    plt.show()
else:
    print("As colunas 'risco_predito' e/ou 'risco_evasao' não estão no DataFrame.")

