import numpy as np
import matplotlib.pyplot as plt
from collections import deque

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

def resolver_laberinto_bfs(laberinto):
    filas, columnas = laberinto.shape
    visitados = np.zeros((filas, columnas), dtype=bool)
    solucion = np.zeros_like(laberinto)
    queue = deque([(1, 0)])

    while queue:
        x, y = queue.popleft()
        if (x, y) == (columnas - 2, filas - 1):
            solucion[y, x] = 1
            break
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if nx >= 0 and nx < columnas and ny >= 0 and ny < filas and not visitados[ny, nx] and laberinto[ny, nx] == 0:
                visitados[ny, nx] = True
                solucion[ny, nx] = 1
                queue.append((nx, ny))

    return solucion

laberinto_recursivo = generar_laberinto_recursivo(81, 81) # Esto marca el tamaño
solucion_bfs = resolver_laberinto_bfs(laberinto_recursivo)

fig_bfs, (ax1_bfs, ax2_bfs) = plt.subplots(1, 2, figsize=(10, 5))
ax1_bfs.imshow(laberinto_recursivo, cmap="binary")
ax1_bfs.set_title('Laberinto (Recursivo)')
ax2_bfs.imshow(solucion_bfs, cmap="autumn")
ax2_bfs.set_title('Solución (BFS)')
#plt.show()
#plt.savefig('laberinto_solucion.png')

# Guardar el laberinto en un archivo
plt.figure() # Crear una nueva figura
plt.imshow(laberinto_recursivo, cmap="binary")
plt.title('Laberinto (Recursivo)')
plt.axis('off') # Ocultar ejes
plt.savefig('laberinto.png')

# Guardar la solución en un archivo
plt.figure() # Crear una nueva figura
plt.imshow(solucion_bfs, cmap="autumn")
plt.title('Solución (BFS)')
plt.axis('off') # Ocultar ejes
plt.savefig('solucion.png')

