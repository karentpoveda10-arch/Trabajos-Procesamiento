# ============================================================
# DETECCIÓN Y CLASIFICACIÓN AUTOMÁTICA DE MONEDAS COLOMBIANAS
# ============================================================

from skimage import io, color, filters, measure, morphology
from scipy.ndimage import binary_fill_holes
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# === 1. CARGA DE LA IMAGEN ===
img = io.imread("moneda3.jpeg")
plt.figure(), plt.imshow(img), plt.title("Imagen original")

# === 2. PREPROCESAMIENTO ===
Igray = color.rgb2gray(img)              # Convertir a escala de grises
Igray = filters.median(Igray)            # Filtro mediana
plt.figure(), plt.imshow(Igray, cmap="gray"), plt.title("Imagen en gris filtrada")

# === 3. UMBRALIZACIÓN ADAPTATIVA ===
BW = Igray < filters.threshold_local(Igray, block_size=55, offset=0.01)
BW = np.invert(BW)  # monedas blancas sobre fondo negro
plt.figure(), plt.imshow(BW, cmap="gray"), plt.title("Imagen binarizada")

# === 4. OPERACIONES MORFOLÓGICAS ===
BW = morphology.remove_small_objects(BW.astype(bool), 1000)
BW = morphology.binary_closing(BW, morphology.disk(8))
BW = binary_fill_holes(BW)
BW = BW.astype(np.uint8)
plt.figure(), plt.imshow(BW, cmap="gray"), plt.title("Después de operaciones morfológicas")

# === 5. ETIQUETADO DE IMÁGENES ===
labels = measure.label(BW)
props = measure.regionprops(labels, intensity_image=Igray)
print(f"Total de objetos detectados: {len(props)}")

# === 6. FILTRO POR CIRCULARIDAD ===
circularidad_min = 0.80
monedas = []
for prop in props:
    if prop.perimeter > 0:
        circ = 4 * np.pi * prop.area / (prop.perimeter ** 2)
        if circ > circularidad_min:
            monedas.append(prop)

print(f"Monedas detectadas (circulares): {len(monedas)}")

# === 7. CLASIFICACIÓN AUTOMÁTICA CON K-MEANS ===
if len(monedas) > 0:
    areas = np.array([m.area for m in monedas]).reshape(-1, 1)

    # Agrupar en 3 tipos de monedas (200, 500, 1000)
    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10).fit(areas)
    labels_k = kmeans.labels_

    # Ordenar los clusters de menor a mayor tamaño
    cluster_mean = kmeans.cluster_centers_.flatten()
    orden = np.argsort(cluster_mean)
    valores = [200, 500, 1000]  # pesos colombianos

    valor_total = 0
    plt.figure(), plt.imshow(img), plt.title("Monedas clasificadas")
    for i, m in enumerate(monedas):
        C = m.centroid
        idx_val = np.where(orden == labels_k[i])[0][0]
        valor = valores[idx_val]
        valor_total += valor

        plt.text(
            C[1],
            C[0],
            f"${valor}",
            color="red",
            fontsize=14,
            fontweight="bold",
        )

    print("===============================================")
    print(f"Valor total estimado: ${valor_total} pesos colombianos")
    print("===============================================")
    plt.show()
else:
    print("No se detectaron monedas en la imagen.")
