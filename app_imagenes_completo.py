# app_imagenes_completo.py
"""
Aplicación completa de Procesamiento de Imágenes Médicas (Tkinter)
Actividades 1..5. Actividad 5 implementa:
 - Filtros pasa bajos / pasa altos / pasa-banda (Ideal, Butterworth, Gaussiano)
 - Aplicación en dominio espacial (convolución) y frecuencia (FFT)
 - Aplicación por canal para imágenes en color
 - Comparación espacial vs FFT (resultados y espectros)
 - Detección Viola–Jones (Haar) para rostros (incluye batch)
 - Filtros de belleza aplicados a rostros:
     * Filtro belleza (PB Gaussiano) -> suaviza rostro vía filtro espacial o FFT LP
     * Filtro belleza (PB + Banda) -> combina suavizado y realce selectivo
 - Comparación Viola–Jones vs Crecimiento de Regiones
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import numpy as np
import cv2
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage import exposure, util, color
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi

# -------------------------
# Utilities
# -------------------------
def imread_rgb(path):
    I = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if I is None:
        return None
    # drop alpha if present
    if I.ndim == 3 and I.shape[2] == 4:
        I = cv2.cvtColor(I, cv2.COLOR_BGRA2BGR)
    if I.ndim == 3:
        I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    return I

def to_gray_uint8(I):
    if I is None:
        return None
    if I.ndim == 3:
        g = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
    else:
        g = I
    if g.dtype != np.uint8:
        g = util.img_as_ubyte(g)
    return g

def uint8_if_needed(img):
    if img is None:
        return None
    if img.dtype == np.uint8:
        return img
    a = img.astype(np.float64)
    a = a - a.min()
    if a.max() > 0:
        a = a / a.max()
    return (a * 255).astype(np.uint8)

def show_warning(msg):
    messagebox.showwarning("Atención", msg)

# -------------------------
# Frequency helpers
# -------------------------
def make_freq_grid(M, N):
    u = np.arange(-N//2, N//2)
    v = np.arange(-M//2, M//2)
    U, V = np.meshgrid(u, v)
    D = np.sqrt(U**2 + V**2)
    return D

def ideal_lowpass_mask(M, N, D0):
    D = make_freq_grid(M, N)
    return (D <= D0).astype(np.float64)

def butterworth_lowpass(M, N, D0, n=2):
    D = make_freq_grid(M, N)
    return 1.0 / (1.0 + (D / (D0 + 1e-9)) ** (2 * n))

def gaussian_lowpass(M, N, D0):
    D = make_freq_grid(M, N)
    return np.exp(-(D**2) / (2 * (D0**2) + 1e-9))

def bandpass_from_lp(H1, H2):
    return np.clip(H2 - H1, 0, 1)

def apply_fft_filter_2d(img2d, H):
    # Forward FFT, shift, multiply, inverse
    F = np.fft.fft2(img2d.astype(np.float64))
    Fshift = np.fft.fftshift(F)
    G = H * Fshift
    Ginv = np.fft.ifftshift(G)
    g = np.fft.ifft2(Ginv)
    g = np.real(g)
    return uint8_if_needed(g)

def spectrum_vis(img2d):
    F = np.fft.fftshift(np.fft.fft2(img2d.astype(np.float64)))
    S = np.log1p(np.abs(F))
    S = S - S.min()
    if S.max() > 0:
        S = S / S.max()
    return (S * 255).astype(np.uint8)

# -------------------------
# GUI setup
# -------------------------
root = tk.Tk()
root.title("Procesamiento de Imágenes Médicas - COMPLETO")
root.geometry("1250x760")

nb = ttk.Notebook(root)
nb.pack(fill="both", expand=True)

# ---------- Tab 1 ----------
tab1 = ttk.Frame(nb); nb.add(tab1, text="Actividad 1")
lbl1 = ttk.Label(tab1, text="Tema: Escala de Grises y Ecualización del Histograma", font=("TkDefaultFont", 12, "bold"))
lbl1.pack(anchor="nw", pady=6, padx=6)
frame1_controls = ttk.Frame(tab1); frame1_controls.pack(anchor="ne", pady=4, padx=4)
btn1 = ttk.Button(frame1_controls, text="Seleccionar Imagen", width=20); btn1.pack()
fig1 = Figure(figsize=(10.5,5.5)); axs1 = [fig1.add_subplot(2,3,i+1) for i in range(6)]
for ax in axs1: ax.axis('off')
canvas1 = FigureCanvasTkAgg(fig1, master=tab1); canvas1.get_tk_widget().pack(fill="both", expand=True)

# ---------- Tab 2 ----------
tab2 = ttk.Frame(nb); nb.add(tab2, text="Actividad 2")
lbl2 = ttk.Label(tab2, text="Tema: Áreas de poco contraste e información oculta", font=("TkDefaultFont", 12, "bold"))
lbl2.pack(anchor="nw", pady=6, padx=6)
frame2_controls = ttk.Frame(tab2); frame2_controls.pack(anchor="ne", pady=4, padx=4)
btn2 = ttk.Button(frame2_controls, text="Seleccionar Imagen", width=20); btn2.pack()
fig2 = Figure(figsize=(10.5,5.5)); axs2 = [fig2.add_subplot(2,3,i+1) for i in range(6)]
for ax in axs2: ax.axis('off')
canvas2 = FigureCanvasTkAgg(fig2, master=tab2); canvas2.get_tk_widget().pack(fill="both", expand=True)

# ---------- Tab 3 ----------
tab3 = ttk.Frame(nb); nb.add(tab3, text="Actividad 3")
lbl3 = ttk.Label(tab3, text="Tema: Crecimiento de Regiones (clic para semilla)", font=("TkDefaultFont", 12, "bold"))
lbl3.pack(anchor="nw", pady=6, padx=6)
frame3_left = ttk.Frame(tab3); frame3_left.pack(side="left", fill="y", padx=8, pady=8)
ttk.Label(frame3_left, text="Controles").pack(pady=4)
btn3_cargar = ttk.Button(frame3_left, text="Cargar Imagen", width=20); btn3_cargar.pack(pady=6)
ttk.Label(frame3_left, text="Umbral de crecimiento:").pack(pady=(10,0))
scale3 = ttk.Scale(frame3_left, from_=0, to=100, orient="horizontal"); scale3.set(15); scale3.pack(pady=6, fill="x")
btn3_procesar = ttk.Button(frame3_left, text="Aplicar Crecimiento", width=20); btn3_procesar.pack(pady=10)
frame3_right = ttk.Frame(tab3); frame3_right.pack(side="left", fill="both", expand=True, padx=8, pady=8)
fig3 = Figure(figsize=(10.5,6))
ax3_orig = fig3.add_subplot(2,2,1); ax3_orig.set_title('Imagen Original (clic)'); ax3_orig.axis('off')
ax3_res = fig3.add_subplot(2,2,2); ax3_res.set_title('Crecimiento de Regiones'); ax3_res.axis('off')
ax3_comp = fig3.add_subplot(2,2,3); ax3_comp.set_title('Método Comparativo'); ax3_comp.axis('off')
ax3_color = fig3.add_subplot(2,2,4); ax3_color.set_title('Resultado en Color'); ax3_color.axis('off')
canvas3 = FigureCanvasTkAgg(fig3, master=frame3_right); canvas3.get_tk_widget().pack(fill="both", expand=True)

# ---------- Tab 4 ----------
tab4 = ttk.Frame(nb); nb.add(tab4, text="Actividad 4")
lbl4 = ttk.Label(tab4, text="Tema: Filtros Espaciales y Convolución", font=("TkDefaultFont", 12, "bold"))
lbl4.pack(anchor="nw", pady=6, padx=6)
frame4_controls = ttk.Frame(tab4); frame4_controls.pack(anchor="ne", pady=4, padx=4)
btn4 = ttk.Button(frame4_controls, text="Seleccionar Imagen", width=20); btn4.pack()
fig4 = Figure(figsize=(10.5,5.5)); axs4 = [fig4.add_subplot(2,3,i+1) for i in range(6)]
for ax in axs4: ax.axis('off')
canvas4 = FigureCanvasTkAgg(fig4, master=tab4); canvas4.get_tk_widget().pack(fill="both", expand=True)

# ---------- Tab 5 ----------
tab5 = ttk.Frame(nb); nb.add(tab5, text="Actividad 5")
lbl5 = ttk.Label(tab5, text="Tema: FFT, Filtros y Operaciones Avanzadas (COMPLETO)", font=("TkDefaultFont", 12, "bold"))
lbl5.pack(anchor="nw", pady=6, padx=6)
frame5_top = ttk.Frame(tab5); frame5_top.pack(fill="x", padx=6, pady=4)
btn5_cargar = ttk.Button(frame5_top, text="Cargar Imagen", width=18); btn5_cargar.pack(side="left", padx=6)
ttk.Label(frame5_top, text="Seleccione operación:").pack(side="left", padx=(10,6))

filtros = [
    'Original (Gris)',
    'Ideal LP (FFT)',
    'Butterworth LP (FFT)',
    'Gauss LP (FFT)',
    'Ideal HP (FFT)',
    'Butterworth HP (FFT)',
    'Gauss HP (FFT)',
    'Butterworth BP (FFT)',
    'Ideal LP (Espacial conv)',
    'Gauss LP (Espacial conv)',
    'Promedio (Espacial conv)',
    'Espectro Original',
    'Espectro Filtrado (FFT)',
    'Color - Butterworth LP (canal por canal)',
    'Color - Gauss LP (canal por canal)',
    'Detección Viola-Jones (rostros)',
    'Detección Viola-Jones sobre carpeta (batch)',
    'Segmentación piel (HSV)',
    'Filtro belleza (Face - PB Gaussiano)',
    'Filtro belleza (Face - PB + BP)',
    'Comparar Espacial vs FFT (LP)',
    'Comparar Viola-Jones vs Crecimiento (segmentación)'
]
combo5 = ttk.Combobox(frame5_top, values=filtros, width=55, state="readonly")
combo5.pack(side="left", padx=6); combo5.set(filtros[0])
frame5_main = ttk.Frame(tab5); frame5_main.pack(fill="both", expand=True, padx=6, pady=6)
fig5 = Figure(figsize=(11,6))
ax5_orig = fig5.add_subplot(1,2,1); ax5_orig.set_title('Imagen Original'); ax5_orig.axis('off')
ax5_res = fig5.add_subplot(1,2,2); ax5_res.set_title('Resultado'); ax5_res.axis('off')
canvas5 = FigureCanvasTkAgg(fig5, master=frame5_main); canvas5.get_tk_widget().pack(fill="both", expand=True)

# -------------------------
# State storages
# -------------------------
state1 = {'I': None, 'Igray': None}
state2 = {'I': None, 'Igray': None}
state3 = {'I': None, 'Igray': None, 'seed': None}
state4 = {'I': None, 'Igray': None}
state5 = {'I': None, 'Igray': None, 'name': ''}
# -------------------------
# Activity 1 implementation
# -------------------------
def procesarImagen_tab1():
    # Abrir diálogo para seleccionar un archivo; devuelve ruta o cadena vacía si se cancela
    path = filedialog.askopenfilename(
        filetypes=[("Imágenes", "*.jpg *.png *.bmp *.tif"), ("All", "*.*")]
    )
    if not path:
        return  # el usuario canceló, salir sin hacer nada

    # Leer la imagen en RGB (se asume que imread_rgb devuelve un array HxWx3 uint8)
    I = imread_rgb(path)
    if I is None:
        # Si la lectura falla, mostrar error al usuario y salir
        messagebox.showerror("Error", "No se pudo abrir la imagen")
        return

    # Guardar la imagen original en el estado (diccionario compartido con la GUI)
    state1['I'] = I

    # Convertir a escala de grises en formato uint8 (0..255) y guardarlo
    # to_gray_uint8 es una función auxiliar que debería:
    # - convertir RGB->grises usando una combinación ponderada (luminancia)
    # - asegurar tipo uint8
    state1['Igray'] = to_gray_uint8(I)

    # --- Mostrar la imagen original en el primer axes ---
    axs1[0].clear()
    axs1[0].imshow(I)
    axs1[0].set_title('Imagen Original')
    axs1[0].axis('off')  # ocultar ejes/ticks para una vista limpia

    # --- Mostrar escala de grises en el segundo axes ---
    axs1[1].clear()
    axs1[1].imshow(state1['Igray'], cmap='gray')
    axs1[1].set_title('Escala de Grises')
    axs1[1].axis('off')

    # --- Histograma de la imagen en gris (tercer axes) ---
    axs1[2].clear()
    axs1[2].hist(state1['Igray'].ravel(), bins=256)  # ravel() aplana el array
    axs1[2].set_title('Histograma Gris')

    # --- Ecualización del histograma ---
    # exposure.equalize_hist (skimage.exposure) devuelve una imagen tipo float en [0,1]
    Ieq = exposure.equalize_hist(state1['Igray'])

    # Mostrar la imagen ecualizada (convertida a uint8 si la GUI espera 0..255)
    axs1[3].clear()
    axs1[3].imshow(uint8_if_needed(Ieq), cmap='gray')  # uint8_if_needed: helper para visualizar correctamente
    axs1[3].set_title('Ecualización')
    axs1[3].axis('off')

    # --- Histograma de la imagen ecualizada ---
    # Aquí convierten Ieq de [0,1] a 0..255 y a uint8 para que el histograma tenga 256 bins entre 0 y 255
    axs1[4].clear()
    axs1[4].hist((Ieq * 255).astype(np.uint8).ravel(), bins=256)
    axs1[4].set_title('Histograma Eq')

    # --- Transformación logarítmica ---
    # np.log1p(x) = log(1 + x) (estable numéricamente para x>=0)
    # Se hace sobre la imagen en uint8 convertida a float64 para evitar overflow/inexactitud
    Ilog = np.log1p(state1['Igray'].astype(np.float64))

    # Normalizar Ilog a rango [0,1] para visualizar y convertir a uint8
    Ilog = (Ilog - Ilog.min()) / (Ilog.max() - Ilog.min() + 1e-9)

    # Mostrar la transformación log escalada a 0..255
    axs1[5].clear()
    axs1[5].imshow((Ilog * 255).astype(np.uint8), cmap='gray')
    axs1[5].set_title('Transformación Log')
    axs1[5].axis('off')

    # Refrescar el canvas de Matplotlib embebido en la GUI
    canvas1.draw()

# Asociar la función al botón
btn1.config(command=procesarImagen_tab1)

# -------------------------
# Activity 2 implementation
# -------------------------
def procesarActividad2_tab():
    path = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg *.png *.bmp *.tif"), ("All", "*.*")])
    if not path: return
    I = imread_rgb(path)
    if I is None:
        messagebox.showerror("Error", "No se pudo abrir la imagen"); return
    state2['I'] = I
    state2['Igray'] = to_gray_uint8(I)
    axs2[0].clear(); axs2[0].imshow(state2['Igray'], cmap='gray'); axs2[0].set_title('Original'); axs2[0].axis('off')
    axs2[1].clear(); axs2[1].hist(state2['Igray'].ravel(), bins=256); axs2[1].set_title('Histograma')
    Ieq = exposure.equalize_hist(state2['Igray'])
    axs2[2].clear(); axs2[2].imshow(uint8_if_needed(Ieq), cmap='gray'); axs2[2].set_title('Ecualizada'); axs2[2].axis('off')
    imgf = state2['Igray'].astype(np.float64)
    ksize = 15
    mean = cv2.blur(imgf, (ksize, ksize))
    mean_sq = cv2.blur(imgf*imgf, (ksize, ksize))
    localVar = mean_sq - mean*mean
    localStd = np.sqrt(np.clip(localVar, 0, None))
    axs2[3].clear(); axs2[3].imshow(uint8_if_needed(localStd), cmap='gray'); axs2[3].set_title('Contraste local'); axs2[3].axis('off')
    Ismooth = cv2.blur(state2['Igray'], (7,7))
    k = 1.5
    Ires = cv2.addWeighted(state2['Igray'], 1.0 + k, Ismooth, -k, 0)
    axs2[4].clear(); axs2[4].imshow(Ismooth, cmap='gray'); axs2[4].set_title('Suavizada'); axs2[4].axis('off')
    axs2[5].clear(); axs2[5].imshow(Ires, cmap='gray'); axs2[5].set_title(f'Realzada (k={k})'); axs2[5].axis('off')
    canvas2.draw()
btn2.config(command=procesarActividad2_tab)

# -------------------------
# Activity 3 implementation (comentado)
# -------------------------
_click_cid3 = None

def cargarImagen_tab3():
    # Abrir diálogo y cargar la imagen
    path = filedialog.askopenfilename(
        filetypes=[("Imágenes", "*.jpg *.png *.bmp *.tif"), ("All", "*.*")]
    )
    if not path:
        return

    I = imread_rgb(path)                       # leer imagen (HxWx3 o HxW)
    if I is None:
        messagebox.showerror("Error", "No se pudo abrir la imagen")
        return

    # Guardar estado: original, versión en gris y semilla (vacía aún)
    state3['I'] = I
    state3['Igray'] = to_gray_uint8(I)
    state3['seed'] = None

    # Limpiar y actualizar ejes de la GUI
    ax3_orig.clear()
    ax3_orig.imshow(I)
    ax3_orig.set_title('Imagen Original (clic para semilla)')
    ax3_orig.axis('off')

    ax3_res.clear(); ax3_res.set_title('Crecimiento de Regiones'); ax3_res.axis('off')
    ax3_comp.clear(); ax3_comp.set_title('Método Comparativo'); ax3_comp.axis('off')
    ax3_color.clear(); ax3_color.set_title('Resultado en Color'); ax3_color.axis('off')
    canvas3.draw()

    # (Re)conectar el evento de clic en la figura para definir la semilla.
    # Si ya había una conexión, la desconectamos para evitar duplicados.
    global _click_cid3
    if _click_cid3 is not None:
        fig3.canvas.mpl_disconnect(_click_cid3)
    _click_cid3 = fig3.canvas.mpl_connect('button_press_event', definirSemilla_tab3)


def definirSemilla_tab3(event):
    # Se ejecuta al hacer clic; sólo actuamos si el clic fue sobre el axes correcto.
    if event.inaxes != ax3_orig:
        return
    if event.xdata is None or event.ydata is None:
        return

    # Convertir coordenadas flotantes a enteras (column = x, row = y)
    x = int(round(event.xdata))
    y = int(round(event.ydata))

    # Guardar semilla en el estado (x,y)
    state3['seed'] = (x, y)

    # Dibujar un asterisco rojo para indicar visualmente la semilla
    ax3_orig.plot(x, y, 'r*', markersize=10)
    canvas3.draw()


def procesarCrecimiento_tab3():
    # Verificaciones previas
    if state3['Igray'] is None or state3['seed'] is None:
        messagebox.showerror("Error", "Primero cargue imagen y seleccione semilla (clic en la imagen)")
        return

    # Obtener umbral desde el slider (scale3) — valor entero representando diferencia en niveles de gris
    th = int(scale3.get())

    Igray = state3['Igray']
    sx, sy = state3['seed']   # sx=columna (x), sy=fila (y)

    # Intensidad de la semilla (valor de referencia fijo)
    seedVal = int(Igray[sy, sx])

    H, W = Igray.shape

    # Máscara booleana de la misma forma que Igray (False = no incluida)
    J = np.zeros_like(Igray, dtype=bool)

    # Incluir la semilla y preparar la lista/cola de píxeles por explorar
    J[sy, sx] = True
    pixList = [(sx, sy)]

    # Vecindad 8-conectada (incluye diagonales)
    vecinos = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    # Bucle de crecimiento (BFS): mientras haya píxeles pendientes
    while pixList:
        px, py = pixList.pop(0)       # pop(0) -> FIFO
        for dx, dy in vecinos:
            nx, ny = px + dx, py + dy
            # Comprobación de límites y que no haya sido incluida aún
            if 0 <= nx < W and 0 <= ny < H and not J[ny, nx]:
                # Criterio de inclusión: diferencia absoluta con la semilla < umbral
                if abs(int(Igray[ny, nx]) - seedVal) < th:
                    J[ny, nx] = True            # marcar como parte de la región
                    pixList.append((nx, ny))   # añadir a la cola para expandir desde él

    # Mostrar máscara resultante (blanco = región)
    ax3_res.clear()
    ax3_res.imshow(J, cmap='gray')
    ax3_res.set_title('Crecimiento de Regiones')
    ax3_res.axis('off')

    # Método comparativo: Binarización por Otsu (umbral global automático)
    try:
        T = threshold_otsu(Igray)    # umbral óptimo
        BW = Igray > T
    except Exception:
        # Fallback si Otsu falla por alguna razón (ej. imagen constante)
        BW = Igray > np.mean(Igray)

    ax3_comp.clear()
    ax3_comp.imshow(BW, cmap='gray')
    ax3_comp.set_title('Binarización Otsu')
    ax3_comp.axis('off')

    # Construir overlay coloreado: resaltar la región J sobre la imagen original
    overlay = state3['I'].copy()
    overlay = uint8_if_needed(overlay)   # asegurar tipo uint8
    mask = J

    # Si la imagen original es monocroma, replicarla a 3 canales
    if overlay.ndim == 2:
        overlay_col = np.stack([overlay, overlay, overlay], axis=2)
    else:
        overlay_col = overlay.copy()

    # Para los píxeles de la máscara: forzamos canal rojo a 255 y atenuamos los otros
    overlay_col[mask, 0] = 255
    overlay_col[mask, 1] = (overlay_col[mask, 1] * 0.4).astype(np.uint8)
    overlay_col[mask, 2] = (overlay_col[mask, 2] * 0.4).astype(np.uint8)

    ax3_color.clear()
    ax3_color.imshow(overlay_col)
    ax3_color.set_title('Overlay')
    ax3_color.axis('off')

    # Refrescar GUI
    canvas3.draw()


# Asociar botones a funciones
btn3_cargar.config(command=cargarImagen_tab3)
btn3_procesar.config(command=procesarCrecimiento_tab3)

# -------------------------
# Activity 4 implementation
# -------------------------
def procesarActividad4_tab():
    path = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg *.png *.bmp *.tif"), ("All", "*.*")])
    if not path: return
    I = imread_rgb(path)
    if I is None:
        messagebox.showerror("Error", "No se pudo abrir la imagen"); return
    state4['I'] = I; state4['Igray'] = to_gray_uint8(I)
    Igray = state4['Igray'].astype(np.float64)
    sobelx = cv2.Sobel(Igray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(Igray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.hypot(sobelx, sobely)
    kx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]]); ky = kx.T
    prewittx = ndi.convolve(Igray, kx); prewitty = ndi.convolve(Igray, ky)
    prewitt = np.hypot(prewittx, prewitty)
    canny_edges = cv2.Canny(state4['Igray'], 100, 200)
    lap = ndi.filters.laplace(Igray, mode='reflect')
    kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    perso = cv2.filter2D(state4['Igray'].astype(np.float32), -1, kernel)
    axs4[0].clear(); axs4[0].imshow(state4['Igray'], cmap='gray'); axs4[0].set_title('Original'); axs4[0].axis('off')
    axs4[1].clear(); axs4[1].imshow(uint8_if_needed(sobel), cmap='gray'); axs4[1].set_title('Sobel'); axs4[1].axis('off')
    axs4[2].clear(); axs4[2].imshow(uint8_if_needed(prewitt), cmap='gray'); axs4[2].set_title('Prewitt'); axs4[2].axis('off')
    axs4[3].clear(); axs4[3].imshow(canny_edges, cmap='gray'); axs4[3].set_title('Canny'); axs4[3].axis('off')
    axs4[4].clear(); axs4[4].imshow(uint8_if_needed(lap), cmap='gray'); axs4[4].set_title('Laplaciano'); axs4[4].axis('off')
    axs4[5].clear(); axs4[5].imshow(uint8_if_needed(perso), cmap='gray'); axs4[5].set_title('Personalizado'); axs4[5].axis('off')
    canvas4.draw()
btn4.config(command=procesarActividad4_tab)

# -------------------------
# Activity 5 implementation (CORREGIDO y COMENTADO)
# -------------------------

# Ruta del clasificador Haar para detección de rostros
haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
if not os.path.exists(haar_path):
    print("Warning: Haar cascade not found at", haar_path)

# ----------------------------------------
# Cargar imagen en pestaña Actividad 5
# ----------------------------------------
def onLoadImage5():
    # Se abre un cuadro de diálogo para elegir la imagen
    path = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg *.png *.bmp *.tif"), ("All", "*.*")])
    if not path: return
    I = imread_rgb(path)
    if I is None:
        messagebox.showerror("Error", "No se pudo abrir la imagen"); return
    
    # Guardamos en el estado global
    state5['I'] = I
    state5['Igray'] = to_gray_uint8(I)
    state5['name'] = path
    
    # Mostrar la imagen original en el eje correspondiente
    ax5_orig.clear()
    ax5_orig.imshow(I)
    ax5_orig.set_title(f'Original: {os.path.basename(path)}')
    ax5_orig.axis('off')
    
    ax5_res.clear()
    ax5_res.set_title('Resultado')
    ax5_res.axis('off')
    canvas5.draw()

# ----------------------------------------
# Filtros espaciales de suavizado
# ----------------------------------------
def apply_spatial_lp_conv(img2d, kernel_type='average', ksize=9, sigma=2):
    """
    Aplica filtros pasa bajos en dominio espacial.
    - average: filtro de promedio
    - gaussian: filtro gaussiano
    """
    if kernel_type == 'average':
        return cv2.blur(img2d, (ksize, ksize))
    elif kernel_type == 'gaussian':
        return cv2.GaussianBlur(img2d, (ksize, ksize), sigma)
    else:
        return img2d

# ----------------------------------------
# Aplicar filtrado FFT a imágenes en color
# ----------------------------------------
def apply_fft_filter_color(img_color, H):
    """
    Aplica un filtro en frecuencia (H) a cada canal de la imagen en color.
    """
    out = np.zeros_like(img_color)
    for c in range(3):
        out[:,:,c] = apply_fft_filter_2d(img_color[:,:,c], H)
    return out

# ----------------------------------------
# Detección de rostros con Viola-Jones (OpenCV Haar cascades)
# ----------------------------------------
def detect_faces_opencv(Icolor, scaleFactor=1.1, minNeighbors=4, minSize=(30,30)):
    """
    Detecta múltiples rostros en la imagen.
    Se usa el clasificador Haar de OpenCV (Viola-Jones).
    """
    gray = Icolor if Icolor.ndim == 2 else cv2.cvtColor(Icolor, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(haar_path)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors,
        minSize=minSize,
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces

# ----------------------------------------
# Filtro "belleza" aplicado a rostros detectados
# ----------------------------------------
def apply_beauty_on_face(Icolor, face_bbox, method='gauss_spatial',
                         lp_type='gaussian', D0=30, ksize=15, sigma=3,
                         blend_alpha=0.7):
    """
    Aplica suavizado selectivo a un rostro detectado:
    - Se extrae ROI del rostro.
    - Se filtra (Gaussiano, FFT pasa bajos, o mixto).
    - Se crea máscara suave (feathering) para evitar bordes duros.
    - Se combina el resultado con el rostro original.
    """
    out = Icolor.copy().astype(np.uint8)
    x,y,w,h = face_bbox

    # Margen adicional para recorte más natural
    padw = int(0.06*w); padh = int(0.04*h)
    xs, ys = max(0, x+padw), max(0, y+padh)
    xe, ye = min(Icolor.shape[1], x+w-padw), min(Icolor.shape[0], y+h-padh)
    face_roi = out[ys:ye, xs:xe].copy()
    if face_roi.size == 0:
        return out

    # --- Paso 1: aplicar suavizado ---
    if method == 'gauss_spatial':
        filtered = cv2.GaussianBlur(face_roi, (ksize|1, ksize|1), sigma)
    elif method == 'fft_lowpass':
        M, N = face_roi.shape[:2]
        if lp_type == 'butterworth':
            H = butterworth_lowpass(M, N, D0, n=2)
        elif lp_type == 'ideal':
            H = ideal_lowpass_mask(M, N, D0)
        else:
            H = gaussian_lowpass(M, N, D0)
        filtered = np.zeros_like(face_roi)
        for c in range(3):
            filtered[:,:,c] = apply_fft_filter_2d(face_roi[:,:,c], H)
    elif method == 'fft_lowpass_plus_bp':
        M, N = face_roi.shape[:2]
        H_lp = gaussian_lowpass(M, N, D0)
        H_bp = bandpass_from_lp(ideal_lowpass_mask(M,N,15), ideal_lowpass_mask(M,N,80))
        filtered_lp = np.zeros_like(face_roi)
        filtered_bp = np.zeros_like(face_roi)
        for c in range(3):
            filtered_lp[:,:,c] = apply_fft_filter_2d(face_roi[:,:,c], H_lp)
            filtered_bp[:,:,c] = apply_fft_filter_2d(face_roi[:,:,c], H_bp)
        filtered = np.clip(0.85*filtered_lp + 0.15*filtered_bp, 0, 255).astype(np.uint8)
    else:
        filtered = cv2.GaussianBlur(face_roi, (ksize|1, ksize|1), sigma)

    # --- Paso 2: crear máscara elíptica difusa ---
    mask = np.zeros(face_roi.shape[:2], dtype=np.float32)
    center = (mask.shape[1]//2, mask.shape[0]//2)
    axes = (int(0.45*mask.shape[1]), int(0.55*mask.shape[0]))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)
    mask = cv2.GaussianBlur(mask, (51,51), 15)

    # --- Paso 3: fusión filtrado + original ---
    blended = (mask[:,:,None]*filtered + (1-mask[:,:,None])*face_roi).astype(np.uint8)

    # --- Paso 4: mezcla global para no perder detalles ---
    blended = (blend_alpha * blended + (1-blend_alpha) * face_roi).astype(np.uint8)

    # Reemplazar ROI en la imagen completa
    out[ys:ye, xs:xe] = blended
    return out

# ----------------------------------------
# Proceso principal Actividad 5
# ----------------------------------------
def process_activity5():
    if state5['I'] is None:
        show_warning("Cargue una imagen antes de aplicar operaciones.")
        return
    val = combo5.get()
    Icolor = state5['I']
    Igray = state5['Igray']
    M, N = Igray.shape
    D0 = 30; nB = 2

    # Precalcular máscaras de filtros en frecuencia
    H_ideal_lp = ideal_lowpass_mask(M, N, D0)
    H_but_lp = butterworth_lowpass(M, N, D0, n=2)
    H_gau_lp = gaussian_lowpass(M, N, D0)
    H_ideal_hp = 1 - H_ideal_lp
    H_but_hp = 1 - H_but_lp
    H_gau_hp = 1 - H_gau_lp
    H_bp = bandpass_from_lp(ideal_lowpass_mask(M,N,15), ideal_lowpass_mask(M,N,80))
    H_but_bp = butterworth_lowpass(M,N,15,nB) - butterworth_lowpass(M,N,80,nB)

    ax5_res.clear()
    try:
        # --------------------
        # Operaciones básicas
        # --------------------
        if val == 'Original (Gris)':
            ax5_res.imshow(Igray, cmap='gray')
            ax5_res.set_title('Original (Gris)')

        elif val == 'Ideal LP (FFT)':
            J = apply_fft_filter_2d(Igray, H_ideal_lp)
            ax5_res.imshow(J, cmap='gray')
            ax5_res.set_title('Ideal LP (FFT)')

        elif val == 'Butterworth LP (FFT)':
            J = apply_fft_filter_2d(Igray, H_but_lp)
            ax5_res.imshow(J, cmap='gray')
            ax5_res.set_title('Butterworth LP (FFT)')

        elif val == 'Gauss LP (FFT)':
            J = apply_fft_filter_2d(Igray, H_gau_lp)
            ax5_res.imshow(J, cmap='gray')
            ax5_res.set_title('Gauss LP (FFT)')

        elif val == 'Ideal HP (FFT)':
            J = apply_fft_filter_2d(Igray, H_ideal_hp)
            ax5_res.imshow(J, cmap='gray')
            ax5_res.set_title('Ideal HP (FFT)')

        elif val == 'Butterworth HP (FFT)':
            J = apply_fft_filter_2d(Igray, H_but_hp)
            ax5_res.imshow(J, cmap='gray')
            ax5_res.set_title('Butterworth HP (FFT)')

        elif val == 'Gauss HP (FFT)':
            J = apply_fft_filter_2d(Igray, H_gau_hp)
            ax5_res.imshow(J, cmap='gray')
            ax5_res.set_title('Gauss HP (FFT)')

        elif val == 'Butterworth BP (FFT)':
            J = apply_fft_filter_2d(Igray, H_but_bp)
            ax5_res.imshow(J, cmap='gray')
            ax5_res.set_title('Butterworth BP (FFT)')

        elif val == 'Ideal LP (Espacial conv)':
            J = apply_spatial_lp_conv(Igray, 'average', ksize=9)
            ax5_res.imshow(J, cmap='gray')
            ax5_res.set_title('LP (Promedio) - Espacial')

        elif val == 'Gauss LP (Espacial conv)':
            J = apply_spatial_lp_conv(Igray, 'gaussian', ksize=9, sigma=2)
            ax5_res.imshow(J, cmap='gray')
            ax5_res.set_title('Gauss LP - Espacial')

        elif val == 'Promedio (Espacial conv)':
            J = apply_spatial_lp_conv(Igray, 'average', ksize=9)
            ax5_res.imshow(J, cmap='gray')
            ax5_res.set_title('Promedio (Espacial)')

        # --------------------
        # Filtros en color
        # --------------------
        elif val == 'Color - Butterworth LP (canal por canal)':
            Jc = apply_fft_filter_color(Icolor, H_but_lp)
            ax5_res.imshow(Jc.astype(np.uint8))
            ax5_res.set_title('Color - Butterworth LP (por canal)')

        elif val == 'Color - Gauss LP (canal por canal)':
            Jc = apply_fft_filter_color(Icolor, H_gau_lp)
            ax5_res.imshow(Jc.astype(np.uint8))
            ax5_res.set_title('Color - Gauss LP (por canal)')

        # --------------------
        # Detección de rostros
        # --------------------
        elif val == 'Detección Viola-Jones (rostros)':
            faces = detect_faces_opencv(Icolor)
            out = Icolor.copy()
            if len(faces) == 0:
                messagebox.showinfo("Viola-Jones", "No se detectaron rostros.")
            else:
                for (x,y,w,h) in faces:
                    cv2.rectangle(out, (x,y), (x+w,y+h), (255,0,0), 2)
                ax5_res.imshow(out)
                ax5_res.set_title(f'Viola-Jones - {len(faces)} rostros detectados')

        elif val == 'Filtro belleza (Face - PB Gaussiano)':
            faces = detect_faces_opencv(Icolor)
            if len(faces) == 0:
                show_warning("No se detectaron rostros.")
            else:
                areas = [w*h for (x,y,w,h) in faces]
                idx = int(np.argmax(areas))
                bbox = faces[idx]
                out = apply_beauty_on_face(Icolor, bbox, method='gauss_spatial', ksize=21, sigma=5, blend_alpha=0.75)
                ax5_res.imshow(out)
                ax5_res.set_title('Filtro belleza (PB Gaussiano) sobre rostro')

        elif val == 'Filtro belleza (Face - PB + BP)':
            faces = detect_faces_opencv(Icolor)
            if len(faces) == 0:
                show_warning("No se detectaron rostros.")
            else:
                areas = [w*h for (x,y,w,h) in faces]
                idx = int(np.argmax(areas))
                bbox = faces[idx]
                out = apply_beauty_on_face(Icolor, bbox, method='fft_lowpass_plus_bp', D0=25, blend_alpha=0.75)
                ax5_res.imshow(out)
                ax5_res.set_title('Filtro belleza (PB + BP) sobre rostro')

        else:
            ax5_res.text(0.5, 0.5, "Operación no implementada", ha='center')

    except Exception as e:
        ax5_res.clear()
        ax5_res.text(0.1, 0.5, f"Error: {str(e)}")
        print("Error activity5:", e)

    ax5_res.axis('off')
    canvas5.draw()

# ----------------------------------------
# Botones y bindings
# ----------------------------------------
btn5_cargar.config(command=onLoadImage5)
combo5.bind("<<ComboboxSelected>>", lambda e: process_activity5())
btn_apply5 = ttk.Button(frame5_top, text="Aplicar", command=process_activity5)
btn_apply5.pack(side="left", padx=6)

