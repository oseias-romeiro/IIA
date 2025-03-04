# -*- coding: utf-8 -*-
"""Trabalho 1.ipynb

# Caça aos tesouros
Trabalho 1 da disciplina de Introdução a Inteligência Artifical - 2024.2, onde será desenvolvido 3 algoritmos de busca para resolver o problema de caça aos tesouros. O objetivo é encontrar o caminho mais curto para se chegar ao tesouro dada as condiçõpes do mapa entregue. Para isso, será implementado os algoritmos de **Busca Uniforme**, **Busca Gulosa** e <b>A*</b>.


## Problema
As seguintes informações são entregues ao programa:
- **Mapa**: matriz com informações do ambiente;
- **Posição inicial**: coordenada no mapa da posição de partida para início das buscas;
- **Posição final**: coordenada da localização do tesouro no mapa.

### Mapa
O mapa segue uma estrutura de matriz e pode conter os seguintes valores:

- `#`: Obstáculo - não pode passar
- `L`: Lama - custo 5
- `.`: Célula vazia - custo 1
- `T`: Tesouro - objetivo final
- `I`: Início - ponto de partida

## Setup
Configurações básicas para o projeto
"""

import heapq

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

costMap = {
    ".": 1,
    "L": 5,
    "T": 1,
    "I": 1
}

movements = [(-1, 0), (1, 0), (0, -1), (0, 1)]

"""## Solução
Abaixo está as funções auxiliares para resolução do problema e em seguida as soluções utilizando cada método de busca escolhida.
"""

manhattanCalc = lambda pos_cur, pos_fin: (abs(pos_cur[0] - pos_fin[0]) + abs(pos_cur[1] - pos_fin[1]))

def getNextPos(pos_cur, mapa, visited):
  next_positions = []
  for movement in movements:
    line = pos_cur[0] + movement[0]
    column = pos_cur[1] + movement[1]
    next_pos = (line, column)

    if (
      0 <= line < len(mapa) and 0 <= column < len(mapa[0]) and
      mapa[line][column] != '#' and
      next_pos not in visited
    ):
      next_positions.append(next_pos)

  return next_positions

"""### Busca de custo uniforme"""

def busca_custo_uniforme(mapa, pos_ini, pos_fin):
    visited = set()

    priorityQ = [(0, pos_ini, [pos_ini])]

    while priorityQ:
        cost, pos_cur, path = heapq.heappop(priorityQ)

        if pos_cur == pos_fin: return path

        visited.add(pos_cur)

        for next_pos in getNextPos(pos_cur, mapa, visited):
          new_cost = cost + costMap[mapa[next_pos[0]][next_pos[1]]]
          item = (new_cost, next_pos, path + [next_pos])
          heapq.heappush(priorityQ, item)

    return "Não há caminho possível."

"""### Busca Gulosa

"""

def busca_gulosa(mapa, pos_ini, pos_fin):
    visited = set()

    priorityQ = [(manhattanCalc(pos_ini, pos_fin), pos_ini, [pos_ini])]

    while priorityQ:
        _, pos_cur, path = heapq.heappop(priorityQ)

        if pos_cur == pos_fin: return path

        visited.add(pos_cur)

        for next_pos in getNextPos(pos_cur, mapa, visited):
          heapq.heappush(priorityQ, (manhattanCalc(next_pos, pos_fin), next_pos, path + [next_pos]))

    return "Não há caminho possível."

"""### Busca A*


"""

def busca_a_estrela(mapa, pos_ini, pos_fin):
  visited = set()

  priorityQ = [(manhattanCalc(pos_ini, pos_fin), 0, pos_ini, [pos_ini])]

  while priorityQ:
    _, cost, pos_cur, path = heapq.heappop(priorityQ)

    if pos_cur == pos_fin: return path

    if pos_cur not in visited: visited.add(pos_cur)

    for next_pos in getNextPos(pos_cur, mapa, visited):
      g = cost + costMap[mapa[next_pos[0]][next_pos[1]]]
      f = g + manhattanCalc(next_pos, pos_fin)
      heapq.heappush(priorityQ, (f, g, next_pos, path + [next_pos]))

  return "Não há caminho possível."

"""## Executando

"""

path_bcu = busca_custo_uniforme(mock1["map"], mock1["pos_ini"], mock1["pos_fin"])
path_gulosa = busca_gulosa(mock1["map"], mock1["pos_ini"], mock1["pos_fin"])
path_a_estrela = busca_a_estrela(mock1["map"], mock1["pos_ini"], mock1["pos_fin"])

print("Caminho do custo uniforme:")
print(path_bcu)

print("Caminho da busca gulosa:")
print(path_gulosa)

print("Caminho da busca A*:")
print(path_a_estrela)