import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# TRELIÇA PLANA 2D - MÉTODO DOS ELEMENTOS FINITOS
# ============================================================

ESCALA = 300  # fator de escala para visualização da deformada

# Graus de liberdade por nó
GLX = 0
GLY = 1

# ============================================================
# PARÂMETROS DA MALHA
# ============================================================
NEL  = 9   # número de elementos
NGLN = 2   # número de graus de liberdade por nó
NNOE = 2   # número de nós por elemento
NNOS = 6   # número de nós
NVINC = 3  # número de vínculos
NFORC = 2  # número de forças

# Número total de graus de liberdade da estrutura
NGL = NNOS * NGLN

# ============================================================
# PROPRIEDADES DO MATERIAL / SEÇÃO
# ============================================================
A = 0.00146373   # área da seção transversal [m²]
E = 205e9        # módulo de elasticidade [Pa]

# ============================================================
# COORDENADAS NODAIS
# cada linha: [x, y]
# ============================================================
COORD = np.array([
    [0.0, 0.0],   # nó 1
    [0.0, 1.0],   # nó 2
    [0.0, 2.0],   # nó 3
    [0.5, 1.5],   # nó 4
    [1.0, 1.0],   # nó 5
    [0.5, 0.5]    # nó 6
], dtype=float)

# ============================================================
# CONECTIVIDADE DOS ELEMENTOS
# cada linha: [nó inicial, nó final]
# observação: subtrai-se 1 para usar índices do Python
# ============================================================
CONEC = np.array([
    [1, 2],
    [1, 6],
    [2, 3],
    [2, 4],
    [2, 5],
    [2, 6],
    [3, 4],
    [4, 5],
    [5, 6]
], dtype=int) - 1

# ============================================================
# FORÇAS NODAIS
# cada linha: [nó, grau de liberdade, valor]
# ============================================================
FORCAS = np.array([
    [3, GLX,  10000.0],
    [6, GLY, -10000.0]
], dtype=float)

# ============================================================
# VÍNCULOS
# cada linha: [nó, grau de liberdade]
# ============================================================
VINC = np.array([
    [1, GLX],
    [1, GLY],
    [5, GLY]
], dtype=int)

# ============================================================
# DADOS GEOMÉTRICOS DOS ELEMENTOS
# DADOS_ELEM[i,0] = ângulo do elemento i
# DADOS_ELEM[i,1] = comprimento do elemento i
# ============================================================
DADOS_ELEM = np.zeros((NEL, 2), dtype=float)

for e in range(NEL):
    no1 = CONEC[e, 0]
    no2 = CONEC[e, 1]

    dx = COORD[no1, 0] - COORD[no2, 0]
    dy = COORD[no1, 1] - COORD[no2, 1]

    theta = np.arctan2(dy, dx)
    L = np.sqrt(dx**2 + dy**2)

    DADOS_ELEM[e, 0] = theta
    DADOS_ELEM[e, 1] = L

# ============================================================
# INICIALIZAÇÃO DA MATRIZ GLOBAL E DO VETOR GLOBAL
# ============================================================
KG = np.zeros((NGL, NGL), dtype=float)
FG = np.zeros(NGL, dtype=float)

# ============================================================
# MONTAGEM DA MATRIZ DE RIGIDEZ GLOBAL
# ============================================================
for e in range(NEL):
    theta = DADOS_ELEM[e, 0]
    L = DADOS_ELEM[e, 1]

    c = np.cos(theta)
    s = np.sin(theta)

    k = (E * A) / L

    # Matriz de rigidez do elemento de treliça 2D
    KE = k * np.array([
        [ c*c,  c*s, -c*c, -c*s],
        [ c*s,  s*s, -c*s, -s*s],
        [-c*c, -c*s,  c*c,  c*s],
        [-c*s, -s*s,  c*s,  s*s]
    ], dtype=float)

    # Superposição de KE em KG
    for j in range(NNOE):
        for gl_j in range(NGLN):
            col_local = j * NGLN + gl_j
            col_global = CONEC[e, j] * NGLN + gl_j

            for l in range(NNOE):
                for gl_l in range(NGLN):
                    lin_local = l * NGLN + gl_l
                    lin_global = CONEC[e, l] * NGLN + gl_l

                    KG[lin_global, col_global] += KE[lin_local, col_local]

# ============================================================
# MONTAGEM DO VETOR GLOBAL DE FORÇAS
# ============================================================
for i in range(NFORC):
    no = int(FORCAS[i, 0]) - 1
    gl = int(FORCAS[i, 1])
    valor = FORCAS[i, 2]

    indice_global = no * NGLN + gl
    FG[indice_global] += valor

# ============================================================
# APLICAÇÃO DOS VÍNCULOS
# ============================================================
for i in range(NVINC):
    no = VINC[i, 0] - 1
    gl = VINC[i, 1]

    indice_global = no * NGLN + gl

    KG[indice_global, :] = 0.0
    KG[:, indice_global] = 0.0
    KG[indice_global, indice_global] = 1.0
    FG[indice_global] = 0.0

# ============================================================
# SOLUÇÃO DO SISTEMA
# ============================================================
U = np.linalg.solve(KG, FG)   # deslocamentos [m]
U_mm = U * 1e3                # deslocamentos [mm]

print("Deslocamentos nodais [m]:")
print(U)

print("\nDeslocamentos nodais [mm]:")
print(U_mm)

# ============================================================
# CÁLCULO DAS COORDENADAS DEFORMADAS
# ============================================================
XDEF = np.zeros(NNOS)
YDEF = np.zeros(NNOS)

for no in range(NNOS):
    XDEF[no] = COORD[no, 0] + ESCALA * U[NGLN * no]
    YDEF[no] = COORD[no, 1] + ESCALA * U[NGLN * no + 1]

COORD_DEF = np.column_stack((XDEF, YDEF))

# ============================================================
# PLOTAGEM DA ESTRUTURA ORIGINAL E DEFORMADA
# ============================================================
plt.figure(figsize=(8, 6))
ax = plt.gca()
ax.set_aspect('equal')
ax.axis('off')

# Estrutura original
for e in range(NEL):
    no1 = CONEC[e, 0]
    no2 = CONEC[e, 1]

    x = [COORD[no1, 0], COORD[no2, 0]]
    y = [COORD[no1, 1], COORD[no2, 1]]

    plt.plot(x, y, linewidth=4)
    plt.plot(x, y, 'o')

# Estrutura deformada
for e in range(NEL):
    no1 = CONEC[e, 0]
    no2 = CONEC[e, 1]

    xdef = [COORD_DEF[no1, 0], COORD_DEF[no2, 0]]
    ydef = [COORD_DEF[no1, 1], COORD_DEF[no2, 1]]

    plt.plot(xdef, ydef, linewidth=1)
    plt.plot(xdef, ydef, 'o')

plt.title("Treliça original e deformada")
plt.show()