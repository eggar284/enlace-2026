# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 14:17:13 2026

@author: samam
"""
# Importar librerias 
import numpy as np 
import matplotlib.pyplot as plt

# Definir medidas
height = 140
width = 512

# Cargar achivos .bin
img_atom = np.fromfile('atom_image.bin',dtype='float64').reshape(height,width)
img_dark = np.fromfile('dark_image.bin',dtype='float64').reshape(height,width)
img_light = np.fromfile('reference_image.bin',dtype='float64').reshape(height,width)

# Establecer numerador y denominador 
numerador = img_light - img_dark
denominador = img_atom - img_dark

# Limitar valores 
denominador_seguro = np.where(denominador > 0, denominador, 1)
numerador_seguro = np.where(numerador > 0, numerador, 1)

# Calcular 0D 
od_raw = np.log(numerador_seguro/denominador_seguro)
od_final = np.clip(od_raw,0,2)

# Graficar OD 
plt.figure(figsize=(7,7))
plt.imshow(od_final, cmap='hot')
plt.colorbar(label='Optical Density (OD)')
plt.title('Optical Density')
plt.show()

# Encontrar centro de masa, en lugar de pixel brillante
# Correcto
total_od = np.sum(od_final)

if total_od > 0:
    y_indice, x_indice = np.indices(od_final.shape)
    cy = np.sum(y_indice * od_final)/total_od
    cx = np.sum(x_indice * od_final)/total_od
    
else:
    cy, cx = height//2, width//2
    
cy, cx = int(round(cy)), int(round(cx))
print(f"Centro de masa: fila{cy}, columna{cx}")
print(f"Valor de OD en el centro: {od_final[cy, cx]:.3f}")

# Tamaño de la ROI (50 vert, 30 hori)
roi_half_h = 70
roi_half_w = 256

# Calcular los bordes
y1 = max(0, cy - roi_half_h)
y2 = min(height, cy + roi_half_h)
x1 = max(0, cx - roi_half_w)
x2 = min(width, cx + roi_half_w)

roi = od_final[y1:y2, x1:x2]

# --- Graficar ROI con estilo profesional (igual a la del profesor) ---
plt.figure(figsize=(12,3.5))                     # Tamaño de la figura
plt.imshow(roi, cmap='hot', vmin=0, vmax=2.0, interpolation='bilinear')  # Escala de color fija de 0 a 2
plt.colorbar(label='Densidad Óptica (OD)')     # Etiqueta de la barra de color
plt.title('Región de Interés (ROI) - Nube Atómica')
plt.xlabel('X (px)')                           # Etiqueta del eje X
plt.ylabel('Y (px)')                           # Etiqueta del eje Y
plt.tight_layout()                             # Ajusta márgenes
plt.show()

# Contar atomos ROI 
# Usar datos de calibracion 
lambda_sonda = 460.8e-9 # Convertido a metros 
sigma_0 = 3 * lambda_sonda**2 / (2*np.pi) # Formula de sec. transversal
pixel_size_camara = 16e-6 # Convertido a metros 
magnificacion = 2
pixel_size_real = pixel_size_camara / magnificacion
area_pixel_real = pixel_size_real**2

suma_od = np.sum(roi)
N_atoms = (area_pixel_real / sigma_0) * suma_od

print(f"Suma de OD en ROI: {suma_od:.3f}")
print(f"Aprox amount of atoms: {N_atoms:.2e}")

