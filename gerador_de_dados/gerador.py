import mysql.connector
import random
from faker import Faker
from tqdm import tqdm

# Configuração da conexão com o MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="faculdade"
)
cursor = conn.cursor()

# Inicializa o Faker para geração de dados
faker = Faker("pt_BR")

# Configurações de quantidades
NUM_USUARIOS = 10000
NUM_CURSOS = 10
NUM_TURMAS = 20
NUM_PROFESSORES = 50
NUM_ALUNOS = 9500
NUM_MATERIAS = 30
NUM_AVALIACOES = 100
NUM_NOTAS = 5000
NUM_ALERTAS = 3000
NUM_FREQUENCIA = 10000

# Listas para armazenar os IDs
usuarios_ids = []
cursos_ids = []
turmas_ids = []
professores_ids = []
alunos_ids = []
materias_ids = []
avaliacoes_ids = []

# Inserção de cursos
print("Inserindo cursos...")
for _ in tqdm(range(NUM_CURSOS)):
    nome = faker.word().capitalize() + " " + random.choice(["Engenharia", "Ciência", "Administração", "Tecnologia", "Saúde"])
    cursor.execute("INSERT INTO cursos (nome) VALUES (%s)", (nome,))
    cursos_ids.append(cursor.lastrowid)

# Inserção de turmas
print("Inserindo turmas...")
for _ in tqdm(range(NUM_TURMAS)):
    curso_id = random.choice(cursos_ids)
    nome = faker.word().capitalize() + " " + str(random.randint(1, 10))
    cursor.execute("INSERT INTO turmas (curso_id, nome) VALUES (%s, %s)", (curso_id, nome))
    turmas_ids.append(cursor.lastrowid)

# Inserção de usuários (alunos e professores)
print("Inserindo usuários...")
for _ in tqdm(range(NUM_USUARIOS)):
    nome = faker.name()
    email = faker.unique.email()
    senha_hash = faker.sha256()
    tipo = random.choice(["aluno", "professor", "admin"])
    
    cursor.execute("INSERT INTO usuarios (nome, email, senha_hash, tipo) VALUES (%s, %s, %s, %s)", (nome, email, senha_hash, tipo))
    usuario_id = cursor.lastrowid
    usuarios_ids.append((usuario_id, tipo))

# Separar professores e alunos
professores = [uid for uid, tipo in usuarios_ids if tipo == "professor"][:NUM_PROFESSORES]
alunos = [uid for uid, tipo in usuarios_ids if tipo == "aluno"][:NUM_ALUNOS]

# Inserção de professores
print("Inserindo professores...")
for usuario_id in tqdm(professores):
    nome = faker.name()
    telefone = faker.phone_number()
    especializacao = faker.word().capitalize() + " em " + random.choice(["Matemática", "Computação", "História", "Biologia"])
    
    cursor.execute("INSERT INTO professores (nome, usuario_id, telefone, especializacao) VALUES (%s, %s, %s, %s)", 
                   (nome, usuario_id, telefone, especializacao))
    professores_ids.append(cursor.lastrowid)

# Inserção de alunos
print("Inserindo alunos...")
for usuario_id in tqdm(alunos):
    nome = faker.name()
    data_nascimento = faker.date_of_birth(minimum_age=18, maximum_age=30).strftime('%Y-%m-%d')
    curso_id = random.choice(cursos_ids)
    turma_id = random.choice(turmas_ids)
    
    cursor.execute("INSERT INTO alunos (nome, usuario_id, data_nascimento, curso_id, turma_id) VALUES (%s, %s, %s, %s, %s)", 
                   (nome, usuario_id, data_nascimento, curso_id, turma_id))
    alunos_ids.append(cursor.lastrowid)

# Inserção de situação dos alunos
print("Inserindo situação dos alunos...")
for aluno_id in tqdm(alunos_ids):
    salario_medio = round(random.uniform(500, 5000), 2)
    trabalha = random.choice([True, False])
    cidade = faker.city()
    estado = faker.state_abbr()
    uso_alcool = random.choice([True, False])
    fuma = random.choice([True, False])
    uso_drogas = random.choice([True, False])
    problemas_mentais = faker.sentence() if random.random() < 0.2 else None
    
    cursor.execute("INSERT INTO situacao_aluno (aluno_id, salario_medio, trabalha, cidade, estado, uso_alcool, fuma, uso_drogas, problemas_mentais) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                   (aluno_id, salario_medio, trabalha, cidade, estado, uso_alcool, fuma, uso_drogas, problemas_mentais))

# Inserção de matérias
print("Inserindo matérias...")
for _ in tqdm(range(NUM_MATERIAS)):
    nome = faker.word().capitalize()
    curso_id = random.choice(cursos_ids)
    professor_id = random.choice(professores_ids)
    
    cursor.execute("INSERT INTO materias (nome, curso_id, professor_id) VALUES (%s, %s, %s)", (nome, curso_id, professor_id))
    materias_ids.append(cursor.lastrowid)

# Inserção de avaliações
print("Inserindo avaliações...")
for _ in tqdm(range(NUM_AVALIACOES)):
    materia_id = random.choice(materias_ids)
    titulo = faker.sentence()
    descricao = faker.paragraph()
    data = faker.date_this_year().strftime('%Y-%m-%d')
    
    cursor.execute("INSERT INTO avaliacoes (materia_id, titulo, descricao, data) VALUES (%s, %s, %s, %s)", 
                   (materia_id, titulo, descricao, data))
    avaliacoes_ids.append(cursor.lastrowid)

# Inserção de notas
print("Inserindo notas...")
for _ in tqdm(range(NUM_NOTAS)):
    aluno_id = random.choice(alunos_ids)
    avaliacao_id = random.choice(avaliacoes_ids)
    nota = round(random.uniform(0, 10), 2)
    
    cursor.execute("INSERT INTO notas (aluno_id, avaliacao_id, nota) VALUES (%s, %s, %s)", (aluno_id, avaliacao_id, nota))

# Inserção de alertas
print("Inserindo alertas...")
for _ in tqdm(range(NUM_ALERTAS)):
    aluno_id = random.choice(alunos_ids)
    mensagem = faker.sentence()
    
    cursor.execute("INSERT INTO alertas (aluno_id, mensagem) VALUES (%s, %s)", (aluno_id, mensagem))

# Inserção de frequência
print("Inserindo frequência...")
for _ in tqdm(range(NUM_FREQUENCIA)):
    aluno_id = random.choice(alunos_ids)
    materia_id = random.choice(materias_ids)
    data = faker.date_this_year().strftime('%Y-%m-%d')
    presente = random.choice([True, False])
    
    cursor.execute("INSERT INTO frequencia (aluno_id, materia_id, data, presente) VALUES (%s, %s, %s, %s)", 
                   (aluno_id, materia_id, data, presente))

# Confirmar alterações no banco de dados
conn.commit()
cursor.close()
conn.close()

print("Inserção de dados concluída com sucesso! 🚀")
