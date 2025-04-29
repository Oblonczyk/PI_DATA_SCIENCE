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

# Probabilidade de erro (8%)
ERROR_RATE = 0.08

def introduce_errors(value, max_length=None, valid_choices=None):
    if random.random() < ERROR_RATE:
        error_type = random.choice(["null", "corrupt", "long"])
        if error_type == "null":
            return None
        elif error_type == "corrupt":
            if valid_choices:
                return random.choice(valid_choices)  # valor válido
            return "!!!INVALID_DATA!!!"[:max_length] if max_length else "!!!INVALID_DATA!!!"
        elif error_type == "long" and max_length:
            return "X" * (max_length + 10)
    if isinstance(value, str) and max_length:
        return value[:max_length]
    return value

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
    nome = introduce_errors(faker.word().capitalize() + " " + random.choice(["Engenharia", "Ciência", "Administração", "Tecnologia", "Saúde"]), max_length=50)
    cursor.execute("INSERT INTO cursos (nome) VALUES (%s)", (nome,))
    cursos_ids.append(cursor.lastrowid)

# Inserção de usuários com controle de e-mails únicos
print("Inserindo usuários...")
emails_usados = set()

for _ in tqdm(range(NUM_USUARIOS)):
    while True:
        nome = introduce_errors(faker.name(), max_length=100)
        
        # Geração segura de e-mail, mesmo com erro
        email_base = faker.unique.email()
        email = introduce_errors(email_base, max_length=100)

        # Se o e-mail já foi usado, gera outro
        if email in emails_usados:
            continue
        
        emails_usados.add(email)

        senha_hash = introduce_errors(faker.sha256(), max_length=64)
        tipo = introduce_errors(random.choice(["aluno", "professor", "admin"]), max_length=10, valid_choices=["aluno", "professor", "admin"])
        
        try:
            cursor.execute("INSERT INTO usuarios (nome, email, senha_hash, tipo) VALUES (%s, %s, %s, %s)", (nome, email, senha_hash, tipo))
            usuario_id = cursor.lastrowid
            usuarios_ids.append((usuario_id, tipo))
            break
        except mysql.connector.errors.IntegrityError as e:
            # Em caso de duplicidade inesperada, pula e tenta gerar outro
            continue
# Separar professores e alunos
professores = [uid for uid, tipo in usuarios_ids if tipo == "professor"][:NUM_PROFESSORES]
alunos = [uid for uid, tipo in usuarios_ids if tipo == "aluno"][:NUM_ALUNOS]

# Inserção de professores
print("Inserindo professores...")
for usuario_id in tqdm(professores):
    nome = introduce_errors(faker.name(), max_length=100)
    telefone = introduce_errors(faker.phone_number(), max_length=20)
    especializacao = introduce_errors(faker.word().capitalize() + " em " + random.choice(["Matemática", "Computação", "História", "Biologia"]), max_length=100)
    
    cursor.execute("INSERT INTO professores (nome, usuario_id, telefone, especializacao) VALUES (%s, %s, %s, %s)", 
                   (nome, usuario_id, telefone, especializacao))
    professores_ids.append(cursor.lastrowid)

# Inserção de turmas (agora com cursos)
print("Inserindo turmas...")
for _ in tqdm(range(NUM_TURMAS)):
    curso_id = random.choice(cursos_ids)
    nome = introduce_errors("Turma " + faker.random_uppercase_letter() + str(random.randint(1, 9)), max_length=50)
    cursor.execute("INSERT INTO turmas (curso_id, nome) VALUES (%s, %s)", (curso_id, nome))
    turmas_ids.append(cursor.lastrowid)

# Inserção de alunos
print("Inserindo alunos...")
for usuario_id in tqdm(alunos):
    nome = introduce_errors(faker.name(), max_length=100)
    data_nascimento = introduce_errors(faker.date_of_birth(minimum_age=18, maximum_age=30).strftime('%Y-%m-%d'))
    curso_id = random.choice(cursos_ids)
    turma_id = random.choice(turmas_ids)
    
    cursor.execute("INSERT INTO alunos (nome, usuario_id, data_nascimento, curso_id, turma_id) VALUES (%s, %s, %s, %s, %s)", 
                   (nome, usuario_id, data_nascimento, curso_id, turma_id))
    alunos_ids.append(cursor.lastrowid)

# Inserção de situação dos alunos
print("Inserindo situação dos alunos...")
for aluno_id in tqdm(alunos_ids):
    salario_medio = introduce_errors(round(random.uniform(500, 5000), 2))
    trabalha = introduce_errors(random.choice([True, False]))
    cidade = introduce_errors(faker.city(), max_length=50)
    estado = introduce_errors(faker.state_abbr(), max_length=2)
    uso_alcool = introduce_errors(random.choice([True, False]))
    fuma = introduce_errors(random.choice([True, False]))
    uso_drogas = introduce_errors(random.choice([True, False]))
    problemas_mentais = introduce_errors(faker.sentence(), max_length=200) if random.random() < 0.2 else None
    
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

print("Inserção de dados concluída com erros simulados! 🚀")