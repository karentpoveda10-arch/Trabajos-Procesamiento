# app_imagenes_completo.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import cv2
from PIL import Image, ImageTk
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage import exposure, color, util
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi

# ---------------------------
# Helpers: conversion & utils
# ---------------------------
def imread_rgb(path):
    I = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if I is None:
        return None
    # If has alpha, drop it
    if I.ndim == 3 and I.shape[2] == 4:
        I = cv2.cvtColor(I, cv2.COLOR_BGRA2BGR)
    if I.ndim == 3:
        I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    return I

def to_gray_uint8(I):
    if I.ndim == 3:
        g = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
    else:
        g = I
    if g.dtype != np.uint8:
        g = util.img_as_ubyte(g)
    return g

def normalize_display(img):
    # returns float image 0..1 or uint8 depending on caller; for imshow consistent display
    if img is None: return None
    if img.dtype == np.uint8:
        return img
    arr = img.astype(np.float64)
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    return (arr * 255).astype(np.uint8)

def uint8_if_needed(img):
    if img.dtype == np.uint8:
        return img
    img = img.astype(np.float64)
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()
    return (img * 255).astype(np.uint8)

# -------------
# FFT filters
# -------------
def make_freq_grids(M, N):
    u = np.arange(-N//2, N//2)
    v = np.arange(-M//2, M//2)
    U, V = np.meshgrid(u, v)
    D = np.sqrt(U**2 + V**2)
    return U, V, D

def apply_fft_filter(img2d, H):
    # img2d: 2D array
    F = np.fft.fft2(img2d.astype(np.float64))
    Fshift = np.fft.fftshift(F)
    G = H * Fshift
    Ginv = np.fft.ifftshift(G)
    g = np.fft.ifft2(Ginv)
    g = np.real(g)
    return uint8_if_needed(g)

# -------------------------
# GUI: Main window & tabs
# -------------------------
root = tk.Tk()
root.title("Procesamiento de Imágenes Médicas - COMPLETO")
root.geometry("1250x750")

nb = ttk.Notebook(root)
nb.pack(fill="both", expand=True)

# ---------- Tab 1 ----------
tab1 = ttk.Frame(nb); nb.add(tab1, text="Actividad 1")
lbl1 = ttk.Label(tab1, text="Tema: Escala de Grises y Ecualización del Histograma", font=("TkDefaultFont", 12, "bold"))
lbl1.pack(anchor="nw", pady=6, padx=6)

frame1_controls = ttk.Frame(tab1)
frame1_controls.pack(anchor="ne", pady=4, padx=4)
btn1 = ttk.Button(frame1_controls, text="Seleccionar Imagen", width=20)
btn1.pack()

fig1 = Figure(figsize=(10.5,5.5))
axs1 = [fig1.add_subplot(2,3,i+1) for i in range(6)]
for ax in axs1: ax.axis('off')
canvas1 = FigureCanvasTkAgg(fig1, master=tab1)
canvas1.get_tk_widget().pack(fill="both", expand=True)

# ---------- Tab 2 ----------
tab2 = ttk.Frame(nb); nb.add(tab2, text="Actividad 2")
lbl2 = ttk.Label(tab2, text="Tema: Áreas de poco contraste e información oculta", font=("TkDefaultFont", 12, "bold"))
lbl2.pack(anchor="nw", pady=6, padx=6)
frame2_controls = ttk.Frame(tab2); frame2_controls.pack(anchor="ne", pady=4, padx=4)
btn2 = ttk.Button(frame2_controls, text="Seleccionar Imagen", width=20); btn2.pack()

fig2 = Figure(figsize=(10.5,5.5))
axs2 = [fig2.add_subplot(2,3,i+1) for i in range(6)]
for ax in axs2: ax.axis('off')
canvas2 = FigureCanvasTkAgg(fig2, master=tab2)
canvas2.get_tk_widget().pack(fill="both", expand=True)

# ---------- Tab 3 ----------
tab3 = ttk.Frame(nb); nb.add(tab3, text="Actividad 3")
lbl3 = ttk.Label(tab3, text="Tema: Crecimiento de Regiones (clic para semilla)", font=("TkDefaultFont", 12, "bold"))
lbl3.pack(anchor="nw", pady=6, padx=6)

frame3_left = ttk.Frame(tab3); frame3_left.pack(side="left", fill="y", padx=8, pady=8)
ttk.Label(frame3_left, text="Controles").pack(pady=4)
btn3_cargar = ttk.Button(frame3_left, text="Cargar Imagen", width=20); btn3_cargar.pack(pady=6)
ttk.Label(frame3_left, text="Umbral de crecimiento:").pack(pady=(10,0))
scale3 = ttk.Scale(frame3_left, from_=0, to=50, orient="horizontal")
scale3.set(15)
scale3.pack(pady=6, fill="x")
btn3_procesar = ttk.Button(frame3_left, text="Aplicar Crecimiento", width=20); btn3_procesar.pack(pady=10)

frame3_right = ttk.Frame(tab3); frame3_right.pack(side="left", fill="both", expand=True, padx=8, pady=8)
fig3 = Figure(figsize=(10.5,6))
ax3_orig = fig3.add_subplot(2,2,1); ax3_orig.set_title('Imagen Original (clic)'); ax3_orig.axis('off')
ax3_res = fig3.add_subplot(2,2,2); ax3_res.set_title('Crecimiento de Regiones'); ax3_res.axis('off')
ax3_comp = fig3.add_subplot(2,2,3); ax3_comp.set_title('Método Comparativo'); ax3_comp.axis('off')
ax3_color = fig3.add_subplot(2,2,4); ax3_color.set_title('Resultado en Color'); ax3_color.axis('off')
canvas3 = FigureCanvasTkAgg(fig3, master=frame3_right)
canvas3.get_tk_widget().pack(fill="both", expand=True)

# ---------- Tab 4 ----------
tab4 = ttk.Frame(nb); nb.add(tab4, text="Actividad 4")
lbl4 = ttk.Label(tab4, text="Tema: Filtros Espaciales y Convolución", font=("TkDefaultFont", 12, "bold"))
lbl4.pack(anchor="nw", pady=6, padx=6)
frame4_controls = ttk.Frame(tab4); frame4_controls.pack(anchor="ne", pady=4, padx=4)
btn4 = ttk.Button(frame4_controls, text="Seleccionar Imagen", width=20); btn4.pack()

fig4 = Figure(figsize=(10.5,5.5))
axs4 = [fig4.add_subplot(2,3,i+1) for i in range(6)]
for ax in axs4: ax.axis('off')
canvas4 = FigureCanvasTkAgg(fig4, master=tab4)
canvas4.get_tk_widget().pack(fill="both", expand=True)

# ---------- Tab 5 ----------
tab5 = ttk.Frame(nb); nb.add(tab5, text="Actividad 5")
lbl5 = ttk.Label(tab5, text="Tema: FFT, Filtros y Operaciones Avanzadas", font=("TkDefaultFont", 12, "bold"))
lbl5.pack(anchor="nw", pady=6, padx=6)

frame5_top = ttk.Frame(tab5); frame5_top.pack(fill="x", padx=6, pady=4)
btn5_cargar = ttk.Button(frame5_top, text="Cargar Imagen", width=18); btn5_cargar.pack(side="left", padx=6)
ttk.Label(frame5_top, text="Seleccione operación:").pack(side="left", padx=(10,6))

filtros = [
    'Original (Gris)',
    'Butterworth LP (FFT)',
    'Gauss LP (FFT)',
    'Butterworth HP (FFT)',
    'Butterworth BP (FFT)',
    'Filtro Espacial: Promedio',
    'Filtro Espacial: Gaussiano',
    'Espectro Original',
    'Espectro Butterworth LP',
    'Color - Butterworth LP',
    'Color - Gauss LP',
    'Detección Viola-Jones (rostros)',
    'Segmentación piel (HSV)',
    'Filtro belleza (Gaussian Blur)',
    'Filtro belleza (Bilateral)',
    'Filtro belleza (Mediana+Unsharp)'
]
combo5 = ttk.Combobox(frame5_top, values=filtros, width=45, state="readonly")
combo5.pack(side="left", padx=6)
combo5.set(filtros[0])

frame5_main = ttk.Frame(tab5); frame5_main.pack(fill="both", expand=True, padx=6, pady=6)
fig5 = Figure(figsize=(11,6))
ax5_orig = fig5.add_subplot(1,2,1); ax5_orig.set_title('Imagen Original'); ax5_orig.axis('off')
ax5_res = fig5.add_subplot(1,2,2); ax5_res.set_title('Resultado'); ax5_res.axis('off')
canvas5 = FigureCanvasTkAgg(fig5, master=frame5_main)
canvas5.get_tk_widget().pack(fill="both", expand=True)

# ---------------------------
# State storages for tabs
# ---------------------------
state1 = {'I': None, 'Igray': None}
state2 = {'I': None, 'Igray': None}
state3 = {'I': None, 'Igray': None, 'seed': None}
state4 = {'I': None, 'Igray': None}
state5 = {'I': None, 'Igray': None, 'name': ''}

# ---------------------------
# Activity 1 implementation
# ---------------------------
def procesarImagen_tab1():
    path = filedialog.askopenfilename(filetypes=[("Imágenes","*.jpg *.png *.bmp *.tif"),("All","*.*")])
    if not path: return
    I = imread_rgb(path)
    if I is None:
        messagebox.showerror("Error","No se pudo abrir la imagen")
        return
    state1['I'] = I
    state1['Igray'] = to_gray_uint8(I)

    axs1[0].clear(); axs1[0].imshow(I); axs1[0].set_title('Imagen Original'); axs1[0].axis('off')
    axs1[1].clear(); axs1[1].imshow(state1['Igray'], cmap='gray'); axs1[1].set_title('Escala de Grises'); axs1[1].axis('off')
    axs1[2].clear(); axs1[2].hist(state1['Igray'].ravel(), bins=256); axs1[2].set_title('Histograma Gris')
    Ieq = exposure.equalize_hist(state1['Igray'])
    axs1[3].clear(); axs1[3].imshow(uint8_if_needed(Ieq), cmap='gray'); axs1[3].set_title('Ecualización'); axs1[3].axis('off')
    axs1[4].clear(); axs1[4].hist((Ieq*255).astype(np.uint8).ravel(), bins=256); axs1[4].set_title('Histograma Eq')
    Ilog = np.log1p(state1['Igray'].astype(np.float64))
    Ilog = (Ilog - Ilog.min())/(Ilog.max()-Ilog.min()+1e-9)
    axs1[5].clear(); axs1[5].imshow((Ilog*255).astype(np.uint8), cmap='gray'); axs1[5].set_title('Transformación Log'); axs1[5].axis('off')

    canvas1.draw()

btn1.config(command=procesarImagen_tab1)

# ---------------------------
# Activity 2 implementation
# ---------------------------
def procesarActividad2_tab():
    path = filedialog.askopenfilename(filetypes=[("Imágenes","*.jpg *.png *.bmp *.tif"),("All","*.*")])
    if not path: return
    I = imread_rgb(path)
    if I is None:
        messagebox.showerror("Error","No se pudo abrir la imagen"); return
    state2['I'] = I
    state2['Igray'] = to_gray_uint8(I)

    axs2[0].clear(); axs2[0].imshow(state2['Igray'], cmap='gray'); axs2[0].set_title('Original'); axs2[0].axis('off')
    axs2[1].clear(); axs2[1].hist(state2['Igray'].ravel(), bins=256); axs2[1].set_title('Histograma')
    Ieq = exposure.equalize_hist(state2['Igray'])
    axs2[2].clear(); axs2[2].imshow(uint8_if_needed(Ieq), cmap='gray'); axs2[2].set_title('Ecualizada'); axs2[2].axis('off')

    # local std via cv2: compute local variance via filter
    imgf = state2['Igray'].astype(np.float64)
    ksize = 15
    mean = cv2.blur(imgf, (ksize,ksize))
    mean_sq = cv2.blur(imgf*imgf, (ksize,ksize))
    localVar = mean_sq - mean*mean
    localStd = np.sqrt(np.clip(localVar,0,None))
    # normalize for display
    localStd_disp = uint8_if_needed(localStd)
    axs2[3].clear(); axs2[3].imshow(localStd_disp, cmap='gray'); axs2[3].set_title('Contraste local'); axs2[3].axis('off')

    # Unsharp-like: Ires = I + k*(I - Ismooth)
    Ismooth = cv2.blur(state2['Igray'], (7,7))
    k = 1.5
    Ires = cv2.addWeighted(state2['Igray'], 1.0 + k, Ismooth, -k, 0)
    axs2[4].clear(); axs2[4].imshow(Ismooth, cmap='gray'); axs2[4].set_title('Suavizada'); axs2[4].axis('off')
    axs2[5].clear(); axs2[5].imshow(Ires, cmap='gray'); axs2[5].set_title(f'Realzada (k={k})'); axs2[5].axis('off')

    canvas2.draw()

btn2.config(command=procesarActividad2_tab)

# ---------------------------
# Activity 3 implementation (seed click)
# ---------------------------
_click_cid3 = None

def cargarImagen_tab3():
    path = filedialog.askopenfilename(filetypes=[("Imágenes","*.jpg *.png *.bmp *.tif"),("All","*.*")])
    if not path: return
    I = imread_rgb(path)
    if I is None:
        messagebox.showerror("Error","No se pudo abrir la imagen"); return
    state3['I'] = I
    state3['Igray'] = to_gray_uint8(I)
    state3['seed'] = None

    ax3_orig.clear(); ax3_orig.imshow(I); ax3_orig.set_title('Imagen Original (clic para semilla)'); ax3_orig.axis('off')
    ax3_res.clear(); ax3_res.set_title('Crecimiento de Regiones'); ax3_res.axis('off')
    ax3_comp.clear(); ax3_comp.set_title('Método Comparativo'); ax3_comp.axis('off')
    ax3_color.clear(); ax3_color.set_title('Resultado en Color'); ax3_color.axis('off')
    canvas3.draw()

    global _click_cid3
    if _click_cid3 is not None:
        fig3.canvas.mpl_disconnect(_click_cid3)
    _click_cid3 = fig3.canvas.mpl_connect('button_press_event', definirSemilla_tab3)

def definirSemilla_tab3(event):
    # event.xdata, event.ydata in pixel coords
    if event.inaxes != ax3_orig: return
    if event.xdata is None or event.ydata is None: return
    x = int(round(event.xdata)); y = int(round(event.ydata))
    state3['seed'] = (x, y)
    ax3_orig.plot(x, y, 'r*', markersize=10)
    canvas3.draw()

def procesarCrecimiento_tab3():
    if state3['Igray'] is None or state3['seed'] is None:
        messagebox.showerror("Error", "Primero cargue imagen y seleccione semilla (clic en la imagen)")
        return
    th = int(scale3.get())
    Igray = state3['Igray']
    sx, sy = state3['seed']  # note: x=col, y=row
    # BFS region growing using seed intensity
    seedVal = int(Igray[sy, sx])
    H, W = Igray.shape
    J = np.zeros_like(Igray, dtype=bool)
    J[sy, sx] = True
    pixList = [(sx, sy)]
    vecinos = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    while pixList:
        px, py = pixList.pop(0)
        for dx, dy in vecinos:
            nx, ny = px + dx, py + dy
            if nx>=0 and ny>=0 and nx<W and ny<H and not J[ny, nx]:
                if abs(int(Igray[ny, nx]) - seedVal) < th:
                    J[ny, nx] = True
                    pixList.append((nx, ny))
    ax3_res.clear(); ax3_res.imshow(J, cmap='gray'); ax3_res.set_title('Crecimiento de Regiones'); ax3_res.axis('off')

    # Otsu comparativo
    try:
        T = threshold_otsu(Igray)
        BW = Igray > T
    except Exception:
        BW = Igray > np.mean(Igray)
    ax3_comp.clear(); ax3_comp.imshow(BW, cmap='gray'); ax3_comp.set_title('Binarización Otsu'); ax3_comp.axis('off')

    # Overlay color
    overlay = state3['I'].copy()
    if overlay.dtype != np.uint8:
        overlay = uint8_if_needed(overlay)
    # color overlay: red where J True
    mask = J
    if overlay.ndim==2:
        overlay_col = np.stack([overlay, overlay, overlay], axis=2)
    else:
        overlay_col = overlay.copy()
    overlay_col[mask, 0] = 255  # red channel
    overlay_col[mask, 1] = (overlay_col[mask,1]*0.4).astype(np.uint8)
    overlay_col[mask, 2] = (overlay_col[mask,2]*0.4).astype(np.uint8)
    ax3_color.clear(); ax3_color.imshow(overlay_col); ax3_color.set_title('Overlay'); ax3_color.axis('off')

    canvas3.draw()

btn3_cargar.config(command=cargarImagen_tab3)
btn3_procesar.config(command=procesarCrecimiento_tab3)

# ---------------------------
# Activity 4 implementation
# ---------------------------
def procesarActividad4_tab():
    path = filedialog.askopenfilename(filetypes=[("Imágenes","*.jpg *.png *.bmp *.tif"),("All","*.*")])
    if not path: return
    I = imread_rgb(path)
    if I is None:
        messagebox.showerror("Error","No se pudo abrir la imagen"); return
    state4['I'] = I
    state4['Igray'] = to_gray_uint8(I)

    Igray = state4['Igray'].astype(np.float64)
    # Sobel, Prewitt (approx with cv2 or numpy), Canny
    sobelx = cv2.Sobel(Igray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(Igray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.hypot(sobelx, sobely)
    # Prewitt via convolution kernels
    kx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
    ky = kx.T
    prewittx = ndi.convolve(Igray, kx)
    prewitty = ndi.convolve(Igray, ky)
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

# ---------------------------
# Activity 5 implementation (full)
# ---------------------------
def onLoadImage5():
    path = filedialog.askopenfilename(filetypes=[("Imágenes","*.jpg *.png *.bmp *.tif"),("All","*.*")])
    if not path: return
    I = imread_rgb(path)
    if I is None:
        messagebox.showerror("Error","No se pudo abrir la imagen"); return
    state5['I'] = I
    state5['Igray'] = to_gray_uint8(I)
    state5['name'] = path
    ax5_orig.clear(); ax5_orig.imshow(I); ax5_orig.set_title(f'Original: {path.split("/")[-1]}'); ax5_orig.axis('off')
    ax5_res.clear(); ax5_res.set_title('Resultado'); ax5_res.axis('off')
    canvas5.draw()

def imgauss_filter(img, ksize=9, sigma=2):
    return cv2.GaussianBlur(img, (ksize,ksize), sigma)

def unsharp_mask(img, radius=1, amount=0.6):
    # img uint8 or float
    if img.dtype != np.uint8:
        base = uint8_if_needed(img)
    else:
        base = img.copy()
    blur = cv2.GaussianBlur(base, (0,0), radius)
    sharpened = cv2.addWeighted(base, 1+amount, blur, -amount, 0)
    return sharpened

def onSelectFiltro5_event(event=None):
    if state5['I'] is None:
        messagebox.showwarning("Atención", "Primero cargue una imagen con 'Cargar Imagen'")
        return
    val = combo5.get()
    Icolor = state5['I']
    Igray = state5['Igray']
    M, N = Igray.shape
    U, V, D = make_freq_grids(M, N)
    D0 = 30
    nB = 2
    H_LP = 1.0 / (1.0 + (D / (D0 + 1e-9)) ** (2 * nB))
    H_GLP = np.exp(-(D ** 2) / (2 * (D0 ** 2)))
    H_HP = 1 - H_LP
    # BP as difference between two Butterworth LP with different cutoffs
    H_BP = (1.0 / (1.0 + (D / 15.0) ** (2 * nB))) - (1.0 / (1.0 + (D / 80.0) ** (2 * nB)))

    ax5_res.clear()
    # --- Cases:
    if val == 'Original (Gris)':
        ax5_res.imshow(Igray, cmap='gray'); ax5_res.set_title('Original (Gris)')
    elif val == 'Butterworth LP (FFT)':
        J = apply_fft_filter(Igray, H_LP)
        ax5_res.imshow(J, cmap='gray'); ax5_res.set_title('Butterworth LP (FFT)')
    elif val == 'Gauss LP (FFT)':
        J = apply_fft_filter(Igray, H_GLP)
        ax5_res.imshow(J, cmap='gray'); ax5_res.set_title('Gauss LP (FFT)')
    elif val == 'Butterworth HP (FFT)':
        J = apply_fft_filter(Igray, H_HP)
        ax5_res.imshow(J, cmap='gray'); ax5_res.set_title('Butterworth HP (FFT)')
    elif val == 'Butterworth BP (FFT)':
        J = apply_fft_filter(Igray, H_BP)
        ax5_res.imshow(J, cmap='gray'); ax5_res.set_title('Butterworth BP (FFT)')
    elif val == 'Filtro Espacial: Promedio':
        J = cv2.blur(Igray, (9,9))
        ax5_res.imshow(J, cmap='gray'); ax5_res.set_title('Filtro Promedio')
    elif val == 'Filtro Espacial: Gaussiano':
        J = cv2.GaussianBlur(Igray, (9,9), 2)
        ax5_res.imshow(J, cmap='gray'); ax5_res.set_title('Filtro Gaussiano')
    elif val == 'Espectro Original':
        F = np.fft.fftshift(np.fft.fft2(Igray.astype(np.float64)))
        S = np.log1p(np.abs(F))
        ax5_res.imshow(S, cmap='gray'); ax5_res.set_title('Espectro Original'); ax5_res.axis('off')
    elif val == 'Espectro Butterworth LP':
        J = apply_fft_filter(Igray, H_LP)
        F = np.fft.fftshift(np.fft.fft2(J.astype(np.float64)))
        S = np.log1p(np.abs(F))
        ax5_res.imshow(S, cmap='gray'); ax5_res.set_title('Espectro Butterworth LP'); ax5_res.axis('off')
    elif val == 'Color - Butterworth LP':
        J = np.zeros_like(Icolor)
        for c in range(3):
            J[:,:,c] = apply_fft_filter(Icolor[:,:,c], H_LP)
        ax5_res.imshow(J.astype(np.uint8)); ax5_res.set_title('Color - Butterworth LP')
    elif val == 'Color - Gauss LP':
        J = np.zeros_like(Icolor)
        for c in range(3):
            J[:,:,c] = apply_fft_filter(Icolor[:,:,c], H_GLP)
        ax5_res.imshow(J.astype(np.uint8)); ax5_res.set_title('Color - Gauss LP')
    elif val == 'Detección Viola-Jones (rostros)':
        # Use OpenCV Haar cascade (comes with OpenCV install)
        gray = Icolor if Icolor.ndim==2 else cv2.cvtColor(Icolor, cv2.COLOR_RGB2GRAY)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(50,50))
        out = Icolor.copy()
        if len(faces)==0:
            messagebox.showinfo("Viola-Jones", "No se detectaron rostros")
        else:
            # choose largest face if multiple (as MATLAB code did)
            areas = [w*h for (x,y,w,h) in faces]
            idx = int(np.argmax(areas))
            x,y,w,h = faces[idx]
            cv2.rectangle(out, (x,y), (x+w,y+h), (255,0,0), 2)
            # put label
            cv2.putText(out, "Rostro", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
        ax5_res.imshow(out); ax5_res.set_title('Detección Viola-Jones')
    elif val == 'Segmentación piel (HSV)':
        # heuristic thresholds similar to MATLAB code
        hsv = cv2.cvtColor(Icolor, cv2.COLOR_RGB2HSV).astype(np.float32)
        h = hsv[:,:,0] / 180.0  # normalize H 0..1
        s = hsv[:,:,1] / 255.0
        v = hsv[:,:,2] / 255.0
        mask = (h < 0.15) & (s > 0.2) & (v > 0.35)
        mask = ndi.binary_fill_holes(mask)
        # remove small objects: use morphology
        mask = ndi.binary_opening(mask, structure=np.ones((5,5)))
        mask = ndi.binary_closing(mask, structure=np.ones((5,5)))
        mask = mask.astype(np.uint8)*255
        ax5_res.imshow(mask, cmap='gray'); ax5_res.set_title('Segmentación piel (HSV)')
    elif val == 'Filtro belleza (Gaussian Blur)':
        Idd = Icolor.astype(np.float32)/255.0
        J = cv2.GaussianBlur(Idd, (0,0), 3)
        mix = 0.6*J + 0.4*Idd
        ax5_res.imshow(uint8_if_needed(mix)); ax5_res.set_title('Filtro belleza (Gauss)')
    elif val == 'Filtro belleza (Bilateral)':
        try:
            J = cv2.bilateralFilter(Icolor, d=9, sigmaColor=75, sigmaSpace=75)
            ax5_res.imshow(J); ax5_res.set_title('Filtro belleza (Bilateral)')
        except Exception as e:
            ax5_res.imshow(Icolor); ax5_res.set_title('Bilateral no disponible; imagen original')
    elif val == 'Filtro belleza (Mediana+Unsharp)':
        J = Icolor.copy()
        for c in range(3):
            ch = cv2.medianBlur(Icolor[:,:,c], 5)
            # imsharpen equivalent: unsharp mask
            ch_sharp = unsharp_mask(ch, radius=1, amount=0.6)
            J[:,:,c] = ch_sharp
        ax5_res.imshow(J.astype(np.uint8)); ax5_res.set_title('Filtro belleza (Mediana+Unsharp)')
    else:
        ax5_res.text(0.5,0.5,"Operación no implementada", ha='center')

    ax5_res.axis('off')
    canvas5.draw()

btn5_cargar.config(command=onLoadImage5)
combo5.bind("<<ComboboxSelected>>", lambda e: onSelectFiltro5_event())

# allow clicking a small button to apply current combo selection
def apply_current_filter():
    onSelectFiltro5_event()
btn_apply5 = ttk.Button(frame5_top, text="Aplicar", command=apply_current_filter)
btn_apply5.pack(side="left", padx=6)

# ---------------------------
# Start GUI loop
# ---------------------------
# Make layout responsive
root.update_idletasks()
root.minsize(1000,700)

# Show instructions label at bottom
status = ttk.Label(root, text="Instrucciones: cargue imágenes en cada pestaña. En Actividad 3: haga clic en la imagen para elegir la semilla, luego 'Aplicar Crecimiento'.")
status.pack(side="bottom", fill="x")

root.mainloop()
