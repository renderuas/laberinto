import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from heapq import heappop, heappush

def generar_laberinto_recursivo(filas, columnas):
    laberinto = np.ones((filas, columnas))
    visitados = np.zeros((filas, columnas), dtype=bool)

    def visitar(y, x):
        visitados[y, x] = True
        laberinto[y, x] = 0
        direcciones = [(0, -2), (0, 2), (-2, 0), (2, 0)]
        np.random.shuffle(direcciones)
        for dx, dy in direcciones:
            nx, ny = x + dx, y + dy
            if nx >= 1 and nx < columnas - 1 and ny >= 1 and ny < filas - 1 and not visitados[ny, nx]:
                laberinto[y + dy // 2, x + dx // 2] = 0
                visitar(ny, nx)

    visitar(1, 1)
    laberinto[0, 1] = 0
    laberinto[-1, -2] = 0
    return laberinto

def manhattan_dist(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

def resolver_laberinto_a_star(laberinto):
    filas, columnas = laberinto.shape
    inicio = (1, 0)
    meta = (columnas - 2, filas - 1)
    frontera = [(0, inicio)]
    costos = {inicio: 0}
    predecesor = {inicio: None}

    while frontera:
        _, actual = heappop(frontera)
        if actual == meta:
            break
        
        x, y = actual
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if nx >= 0 and nx < columnas and ny >= 0 and ny < filas and laberinto[ny, nx] == 0:
                nuevo_costo = costos[actual] + 1
                vecino = (nx, ny)
                if vecino not in costos or nuevo_costo < costos[vecino]:
                    costos[vecino] = nuevo_costo
                    prioridad = nuevo_costo + manhattan_dist(nx, ny, meta[0], meta[1])
                    heappush(frontera, (prioridad, vecino))
                    predecesor[vecino] = actual

    camino = np.zeros_like(laberinto)
    actual = meta
    while actual is not None:
        camino[actual[1], actual[0]] = 1
        actual = predecesor[actual]

    return camino

laberinto_recursivo = generar_laberinto_recursivo(81, 81)
solucion_a_star = resolver_laberinto_a_star(laberinto_recursivo)

plt.figure()
plt.imshow(laberinto_recursivo, cmap="binary")
plt.title('Laberinto (Recursivo)')
plt.axis('off')
plt.savefig('laberinto.png')

plt.figure()
plt.imshow(solucion_a_star, cmap="autumn")
plt.title('Solución (A*)')
plt.axis('off')
plt.savefig('solucion.png')
