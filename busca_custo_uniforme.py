import heapq

def busca_custo_uniforme(mapa, pos_ini, pos_fin):
    """
    Implementa a Busca de Custo Uniforme para encontrar o menor caminho em uma matriz.

    Args:
        mapa (list of list): A matriz representando o mapa, onde:
                             'I' é o ponto inicial,
                             'T' é o ponto final,
                             '#' são obstáculos,
                             '.' são espaços livres.
        pos_ini (tuple): Coordenada inicial no formato (linha, coluna).
        pos_fin (tuple): Coordenada final no formato (linha, coluna).

    Returns:
        list: Lista de posições do menor caminho do ponto inicial ao final, ou
              uma mensagem indicando que não há caminho.
    """
    # Movimentos possíveis (cima, baixo, esquerda, direita)
    movimentos = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Prioridade inicial: (custo acumulado, posição, caminho percorrido)
    fila_prioridade = [(0, pos_ini, [pos_ini])]
    
    # Conjunto de posições já visitadas
    visitados = set()
    
    while fila_prioridade:
        custo_atual, posicao_atual, caminho = heapq.heappop(fila_prioridade)
        
        # Se alcançarmos o destino, retornar o caminho
        if posicao_atual == pos_fin:
            return caminho
        
        # Marcar a posição como visitada
        if posicao_atual in visitados:
            continue
        visitados.add(posicao_atual)
        
        # Gerar vizinhos válidos
        for movimento in movimentos:
            nova_linha = posicao_atual[0] + movimento[0]
            nova_coluna = posicao_atual[1] + movimento[1]
            nova_posicao = (nova_linha, nova_coluna)
            
            # Verificar se o movimento é válido
            if (
                0 <= nova_linha < len(mapa) and
                0 <= nova_coluna < len(mapa[0]) and
                mapa[nova_linha][nova_coluna] != '#' and
                nova_posicao not in visitados
            ):
                # Adicionar à fila com custo atualizado (consideramos custo = 1 para cada movimento)
                heapq.heappush(fila_prioridade, (custo_atual + 1, nova_posicao, caminho + [nova_posicao]))
    
    # Se não há caminho para o destino
    return "Não há caminho possível."

# Dados fornecidos
mock1 = {
    "map": [
        ['I', '.', '#', '.', '.', '.', '.', '.', '.', '.'],
        ['#', '.', '#', '#', '.', '#', '#', '.', '#', '.'],
        ['.', '.', '.', '.', '.', '.', '#', '.', '.', '.'],
        ['#', '.', '#', '#', '#', '.', '.', '.', '#', '.'],
        ['.', '.', '#', '.', '.', '#', '#', '.', '#', '.'],
        ['.', '#', '.', '.', '.', '.', '#', '.', '#', '.'],
        ['.', '.', '.', '#', '.', '.', '.', '.', '#', 'T'],
        ['.', '#', '#', '.', '#', '#', '.', '#', '#', '.'],
        ['.', '.', '.', '.', '.', '.', '.', '.', '#', '.'],
        ['.', '.', '#', '.', '#', '.', '#', '.', '.', '.'],
    ],
    "pos_ini": (0, 0),
    "pos_fin": (6, 9)
}

# Chamar a função
resultado = busca_custo_uniforme(mock1["map"], mock1["pos_ini"], mock1["pos_fin"])
print(resultado)
