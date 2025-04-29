CREATE DATABASE faculdade;
USE faculdade;

-- Tabela de usuários (alunos, professores e administradores)
CREATE TABLE usuarios (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nome VARCHAR(255),
    email VARCHAR(255) UNIQUE,
    senha_hash VARCHAR(255),
    tipo ENUM('aluno', 'professor', 'admin')
);

-- Tabela de cursos
CREATE TABLE cursos (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nome VARCHAR(255)
);

-- Tabela de turmas
CREATE TABLE turmas (
    id INT AUTO_INCREMENT PRIMARY KEY,
    curso_id INT,
    nome VARCHAR(255),
    FOREIGN KEY (curso_id) REFERENCES cursos(id) ON DELETE CASCADE
);

-- Tabela de professores
CREATE TABLE professores (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nome VARCHAR(255),
    usuario_id INT UNIQUE,
    telefone VARCHAR(20),
    especializacao VARCHAR(255),
    FOREIGN KEY (usuario_id) REFERENCES usuarios(id) ON DELETE CASCADE
);

-- Tabela de alunos
CREATE TABLE alunos (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nome VARCHAR(255),
    usuario_id INT UNIQUE,
    data_nascimento DATE,
    curso_id INT,
    turma_id INT,
    FOREIGN KEY (usuario_id) REFERENCES usuarios(id) ON DELETE CASCADE,
    FOREIGN KEY (curso_id) REFERENCES cursos(id) ON DELETE SET NULL,
    FOREIGN KEY (turma_id) REFERENCES turmas(id) ON DELETE SET NULL
);

-- Tabela de situação do aluno
CREATE TABLE situacao_aluno (
    id INT AUTO_INCREMENT PRIMARY KEY,
    aluno_id INT UNIQUE,
    salario_medio DECIMAL(10,2),
    trabalha BOOLEAN,
    cidade VARCHAR(255),
    estado CHAR(2),
    uso_alcool BOOLEAN,
    fuma BOOLEAN,
    uso_drogas BOOLEAN,
    problemas_mentais TEXT,
    FOREIGN KEY (aluno_id) REFERENCES alunos(id) ON DELETE CASCADE
);

-- Tabela de matérias
CREATE TABLE materias (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nome VARCHAR(255),
    curso_id INT,
    professor_id INT,
    FOREIGN KEY (curso_id) REFERENCES cursos(id) ON DELETE CASCADE,
    FOREIGN KEY (professor_id) REFERENCES professores(id) ON DELETE SET NULL
);

-- Tabela de avaliações
CREATE TABLE avaliacoes (
    id INT AUTO_INCREMENT PRIMARY KEY,
    materia_id INT,
    titulo VARCHAR(255),
    descricao TEXT,
    data DATE,
    FOREIGN KEY (materia_id) REFERENCES materias(id) ON DELETE CASCADE
);

-- Tabela de notas
CREATE TABLE notas (
    id INT AUTO_INCREMENT PRIMARY KEY,
    aluno_id INT,
    avaliacao_id INT,
    nota DECIMAL(5,2),
    FOREIGN KEY (aluno_id) REFERENCES alunos(id) ON DELETE CASCADE,
    FOREIGN KEY (avaliacao_id) REFERENCES avaliacoes(id) ON DELETE CASCADE
);

-- Tabela de alertas para alunos
CREATE TABLE alertas (
    id INT AUTO_INCREMENT PRIMARY KEY,
    aluno_id INT,
    mensagem TEXT,
    data TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (aluno_id) REFERENCES alunos(id) ON DELETE CASCADE
);

-- Tabela de histórico acadêmico
CREATE TABLE historico_academico (
    id INT AUTO_INCREMENT PRIMARY KEY,
    aluno_id INT,
    curso_id INT,
    turma_id INT,
    data_inicio DATE,
    data_fim DATE,
    status ENUM('cursando', 'concluído', 'trancado', 'desistente'),
    FOREIGN KEY (aluno_id) REFERENCES alunos(id) ON DELETE CASCADE,
    FOREIGN KEY (curso_id) REFERENCES cursos(id) ON DELETE CASCADE,
    FOREIGN KEY (turma_id) REFERENCES turmas(id) ON DELETE CASCADE
);

-- Tabela de frequência dos alunos
CREATE TABLE frequencia (
    id INT AUTO_INCREMENT PRIMARY KEY,
    aluno_id INT,
    materia_id INT,
    data DATE,
    presente BOOLEAN,
    FOREIGN KEY (aluno_id) REFERENCES alunos(id) ON DELETE CASCADE,
    FOREIGN KEY (materia_id) REFERENCES materias(id) ON DELETE CASCADE
);

-- Verificar se a estrutura foi criada corretamente
SELECT * FROM usuarios;
SELECT * FROM alunos;
SELECT * FROM professores;
SELECT * FROM cursos;
SELECT * FROM turmas;
SELECT * FROM materias;
SELECT * FROM avaliacoes;
SELECT * FROM notas;
SELECT * FROM alertas;
SELECT * FROM historico_academico;
SELECT * FROM frequencia;
SELECT * FROM situacao_aluno;

CREATE VIEW view_tabelas_colunas AS
SELECT
    TABLE_NAME AS tabela,
    COLUMN_NAME AS coluna,
    DATA_TYPE AS tipo,
    CHARACTER_MAXIMUM_LENGTH AS tamanho_maximo,
    IS_NULLABLE AS permite_nulo,
    COLUMN_DEFAULT AS valor_padrao
FROM information_schema.COLUMNS
WHERE TABLE_SCHEMA = 'faculdade';

SELECT * FROM view_tabelas_colunas;
