import numpy as np
import matplotlib.pyplot as plt

# --- Paso 1: Cargar las imágenes ---
# Los archivos son floats "crudos" (sin header), así que np.fromfile funciona.
# Recuerda: 512 ancho x 140 alto

width = 512 
height = 140

def load_bin(path):
    data = np.fromfile(path, dtype=np.float32)  # ajusta dtype si no es float32
    img = data.reshape(height, width)  # numpy usa (filas, columnas) = (alto, ancho)
    return img

I_atoms = load_bin("2022_06_27_512x140_redMOT_shelving_..._1.bin")
I_light = load_bin("2022_06_27_512x140_redMOT_shelving_..._2.bin")
I_background = load_bin("2022_06_27_512x140_redMOT_shelving_..._3.bin")

# --- Paso 2: Verificar que se ven bien ---
# Antes de calcular OD, siempre es buena idea ver las 3 imágenes crudas
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(I_atoms); axes[0].set_title("atoms")
axes[1].imshow(I_light); axes[1].set_title("light (reference)")
axes[2].imshow(I_background); axes[2].set_title("background/dark")
plt.show()

# Si alguna se ve "rotada" o con ancho/alto invertido,
# es señal de que el reshape(height, width) debería ser reshape(width, height)