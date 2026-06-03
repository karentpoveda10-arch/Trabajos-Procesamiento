#!/usr/bin/env python3
"""
Proyecto Imágenes Médicas - versión Python (Tkinter + OpenCV + Pillow + skimage)

Funcionalidad equivalente al script MATLAB que me diste:
- Pestañas: Formatos, Resolución, Color, Operaciones, Comparaciones
- Cargar imagen(es), mostrar propiedades de archivo
- Redimensionar y mostrar PSNR/SSIM
- Convertir color: YMC (simulado), HSV, LAB
- Operaciones: suma, resta, multiplicación, división, AND, OR, XOR
- Umbralización con slider
- Comparaciones entre 2 imágenes: diferencia + PSNR + SSIM
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
import os

# ---------- Helpers ----------
def cv2_to_pil(cv_img):
    """Convert BGR (cv2) image to PIL Image"""
    if cv_img is None:
        return None
    if len(cv_img.shape) == 2:
        mode = "L"
        img = Image.fromarray(cv_img)
    else:
        img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    return img

def pil_to_tk(pil_img, maxsize=(480, 480)):
    """Return ImageTk.PhotoImage resized to fit maxsize while keeping aspect."""
    if pil_img is None:
        return None
    pil_img.thumbnail(maxsize, Image.ANTIALIAS)
    return ImageTk.PhotoImage(pil_img)

def safe_imread(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    # Some formats may have alpha; convert to BGR
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        # remove alpha
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def compute_psnr_ssim(a, b):
    """Assumes a,b are grayscale or color arrays same size (uint8)."""
    try:
        psnr_val = sk_psnr(a, b, data_range=255)
    except Exception:
        psnr_val = float('nan')
    try:
        if a.ndim == 3:
            # sk_ssim for color: use multichannel=True
            ssim_val = sk_ssim(a, b, data_range=255, multichannel=True)
        else:
            ssim_val = sk_ssim(a, b, data_range=255)
    except Exception:
        ssim_val = float('nan')
    return psnr_val, ssim_val

# ---------- App ----------
class ProyectoImagenesApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Procesador de Imágenes Médicas (Python)")
        self.root.geometry("1100x650")

        # Variables que contienen imágenes en formato OpenCV (BGR)
        self.img_loaded = None
        self.img2_loaded = None
        self.imgC1 = None
        self.imgC2 = None
        self.gray_img = None  # for thresholding display

        # Widgets references for image previews (PhotoImage objects must be saved)
        self.preview_photo_1 = None
        self.preview_photo_2 = None
        self.preview_photo_3 = None
        self.preview_photo_4 = None

        # Create notebook (tabs)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True)

        # Create frames for tabs with padding so widgets don't overlap
        self.tab1 = ttk.Frame(self.notebook, padding=8)
        self.tab2 = ttk.Frame(self.notebook, padding=8)
        self.tab3 = ttk.Frame(self.notebook, padding=8)
        self.tab4 = ttk.Frame(self.notebook, padding=8)
        self.tab5 = ttk.Frame(self.notebook, padding=8)

        self.notebook.add(self.tab1, text='Formatos')
        self.notebook.add(self.tab2, text='Resolución')
        self.notebook.add(self.tab3, text='Color')
        self.notebook.add(self.tab4, text='Operaciones')
        self.notebook.add(self.tab5, text='Comparaciones')

        # Build each tab
        self.build_tab_formatos()
        self.build_tab_resolucion()
        self.build_tab_color()
        self.build_tab_operaciones()
        self.build_tab_comparaciones()

    # ---------------- TAB 1: FORMATOS ----------------
    def build_tab_formatos(self):
        frame = self.tab1

        # Left: preview canvas (use Label)
        left_frame = ttk.Frame(frame, width=520)
        left_frame.pack(side='left', fill='both', expand=False)

        btn_load = ttk.Button(left_frame, text="Cargar Imagen", command=self.cargar_imagen)
        btn_load.pack(anchor='nw', pady=(0,8))

        self.label_preview1 = ttk.Label(left_frame)
        self.label_preview1.pack(fill='both', expand=True)

        # Right: info box
        right_frame = ttk.Frame(frame)
        right_frame.pack(side='right', fill='both', expand=True)

        ttk.Label(right_frame, text="Propiedades del archivo:", font=('Segoe UI', 10, 'bold')).pack(anchor='nw')
        self.listbox_info = tk.Listbox(right_frame, height=20)
        self.listbox_info.pack(fill='both', expand=True, padx=5, pady=5)

    def cargar_imagen(self):
        path = filedialog.askopenfilename(filetypes=[("Imagen", "*.jpg;*.png;*.bmp;*.tiff;*.jpeg;*.dicom"), ("Todos", "*.*")])
        if not path:
            return
        img = safe_imread(path)
        if img is None:
            messagebox.showerror("Error", "No se pudo leer la imagen.")
            return
        self.img_loaded = img.copy()
        self.gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pil = cv2_to_pil(img)
        self.preview_photo_1 = pil_to_tk(pil, maxsize=(520,520))
        self.label_preview1.configure(image=self.preview_photo_1)
        self.preview_photo_1 = pil_to_tk(pil, maxsize=(520,520))
        self.label_preview1.configure(image=self.preview_photo_1)
        self.label_preview1.image = self.preview_photo_1   # <--- ESTA LÍNEA ES CLAVE

        # File info
        try:
            info = os.stat(path)
            filesize_kb = info.st_size/1024.0
        except Exception:
            filesize_kb = None
        h, w = img.shape[:2]
        # Color type
        color_type = 'RGB' if img.ndim == 3 else 'Grayscale'
        props = [
            f"Formato (ext): {os.path.splitext(path)[1].lstrip('.')}",
            f"Tamaño: {w} x {h}",
            f"Tipo Color: {color_type}"
        ]
        # Bit depth (assume 8)
        props.append("Profundidad: 8 (por canal, asunción)")
        # Resolution not available from cv2 (same as MATLAB's imfinfo XResolution)
        props.append("Resolución: No disponible (no embedida)")
        if filesize_kb is not None:
            props.append(f"Tamaño archivo: {filesize_kb:.2f} KB")
        # Set listbox
        self.listbox_info.delete(0, tk.END)
        for p in props:
            self.listbox_info.insert(tk.END, p)

    # ---------------- TAB 2: RESOLUCION ----------------
    def build_tab_resolucion(self):
        frame = self.tab2

        controls_frame = ttk.Frame(frame)
        controls_frame.pack(side='top', fill='x', pady=(0,8))

        ttk.Label(controls_frame, text="Selecciona resolución:").pack(side='left', padx=(6,6))
        self.res_combo = ttk.Combobox(controls_frame, values=["128x128", "256x256", "512x512"], state='readonly')
        self.res_combo.current(1)  # default 256x256
        self.res_combo.pack(side='left')

        btn_resize = ttk.Button(controls_frame, text="Redimensionar", command=self.redimensionar)
        btn_resize.pack(side='left', padx=8)

        # Main preview area: two image frames
        preview_frame = ttk.Frame(frame)
        preview_frame.pack(fill='both', expand=True)

        left = ttk.LabelFrame(preview_frame, text="Original")
        left.pack(side='left', fill='both', expand=True, padx=8, pady=8)

        right = ttk.LabelFrame(preview_frame, text="Redimensionada")
        right.pack(side='right', fill='both', expand=True, padx=8, pady=8)

        self.label_res_orig = ttk.Label(left)
        self.label_res_orig.pack(fill='both', expand=True, padx=4, pady=4)

        self.label_res_new = ttk.Label(right)
        self.label_res_new.pack(fill='both', expand=True, padx=4, pady=4)

        # Metrics box below
        metrics_frame = ttk.Frame(frame)
        metrics_frame.pack(side='bottom', fill='x', padx=8, pady=4)
        self.listbox_metrics = tk.Listbox(metrics_frame, height=4)
        self.listbox_metrics.pack(fill='x', expand=True)

    def redimensionar(self):
        if self.img_loaded is None:
            messagebox.showwarning("Aviso", "Primero cargue una imagen en la pestaña 'Formatos'.")
            return
        sel = self.res_combo.get()
        size = int(sel.split('x')[0])
        # original preview
        pil_orig = cv2_to_pil(self.img_loaded)
        self.preview_photo_2 = pil_to_tk(pil_orig, maxsize=(480,480))
        self.label_res_orig.configure(image=self.preview_photo_2)
        self.preview_photo_1 = pil_to_tk(pil, maxsize=(520,520))
        self.label_preview1.configure(image=self.preview_photo_1)
        self.label_preview1.image = self.preview_photo_1   # <--- ESTA LÍNEA ES CLAVE


        resized = cv2.resize(self.img_loaded, (size, size), interpolation=cv2.INTER_AREA)
        pil_res = cv2_to_pil(resized)
        self.preview_photo_3 = pil_to_tk(pil_res, maxsize=(480,480))
        self.label_res_new.configure(image=self.preview_photo_3)

        # Compute PSNR and SSIM between resized and original resized to that size
        orig_resized = cv2.resize(self.img_loaded, (size, size), interpolation=cv2.INTER_AREA)
        psnr_val, ssim_val = compute_psnr_ssim(orig_resized, resized)
        self.listbox_metrics.delete(0, tk.END)
        self.listbox_metrics.insert(tk.END, f"Tamaño elegido: {size}x{size}")
        self.listbox_metrics.insert(tk.END, f"PSNR: {psnr_val:.4f}")
        self.listbox_metrics.insert(tk.END, f"SSIM: {ssim_val:.4f}")

    # ---------------- TAB 3: COLOR ----------------
    def build_tab_color(self):
        frame = self.tab3

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(side='top', fill='x', pady=(0,8))

        ttk.Button(btn_frame, text="Convertir a YMC", command=self.convertir_ymc).pack(side='left', padx=6)
        ttk.Button(btn_frame, text="HSV", command=self.convertir_hsv).pack(side='left', padx=6)
        ttk.Button(btn_frame, text="LAB", command=self.convertir_lab).pack(side='left', padx=6)

        # Layout: original at top-left, three channels below
        top_frame = ttk.Frame(frame)
        top_frame.pack(fill='both', expand=False)
        bottom_frame = ttk.Frame(frame)
        bottom_frame.pack(fill='both', expand=True)

        left_top = ttk.LabelFrame(top_frame, text="Original")
        left_top.pack(side='left', fill='both', expand=True, padx=8, pady=8)
        self.label_color_orig = ttk.Label(left_top)
        self.label_color_orig.pack(fill='both', expand=True)

        # Channels as three small frames
        chan_frame = ttk.Frame(bottom_frame)
        chan_frame.pack(fill='both', expand=True, padx=8, pady=8)

        self.label_chan1 = ttk.LabelFrame(chan_frame, text="Canal 1")
        self.label_chan1.pack(side='left', fill='both', expand=True, padx=6)
        self.preview_chan1 = ttk.Label(self.label_chan1)
        self.preview_chan1.pack(fill='both', expand=True, padx=4, pady=4)

        self.label_chan2 = ttk.LabelFrame(chan_frame, text="Canal 2")
        self.label_chan2.pack(side='left', fill='both', expand=True, padx=6)
        self.preview_chan2 = ttk.Label(self.label_chan2)
        self.preview_chan2.pack(fill='both', expand=True, padx=4, pady=4)

        self.label_chan3 = ttk.LabelFrame(chan_frame, text="Canal 3")
        self.label_chan3.pack(side='left', fill='both', expand=True, padx=6)
        self.preview_chan3 = ttk.Label(self.label_chan3)
        self.preview_chan3.pack(fill='both', expand=True, padx=4, pady=4)

    def convertir_ymc(self):
        if self.img_loaded is None:
            messagebox.showwarning("Aviso", "Primero cargue una imagen en la pestaña 'Formatos'.")
            return
        img = self.img_loaded.astype(np.float32) / 255.0  # normalized RGB
        # Simular Y M C como 1 - channel (no es exactamente Y/M/C real, pero sigue la idea)
        Y = 1.0 - img[:, :, 2]  # using BGR: index 2 is R -> treat as Y (approx)
        M = 1.0 - img[:, :, 1]  # G
        C = 1.0 - img[:, :, 0]  # B

        # Display original
        self.preview_photo_4 = pil_to_tk(cv2_to_pil(self.img_loaded), maxsize=(340,240))
        self.label_color_orig.configure(image=self.preview_photo_4)

        # Each channel scale to 0-255 for display
        def show_channel(ch, widget):
            ch_img = (np.clip(ch, 0, 1) * 255).astype(np.uint8)
            if ch_img.ndim == 2:
                pil = Image.fromarray(ch_img)
            else:
                pil = Image.fromarray(cv2.cvtColor(ch_img, cv2.COLOR_BGR2RGB))
            ph = pil_to_tk(pil, maxsize=(300,300))
            widget.configure(image=ph)
            # keep reference
            if widget is self.preview_chan1:
                self.preview_chan1.img = ph
            if widget is self.preview_chan2:
                self.preview_chan2.img = ph
            if widget is self.preview_chan3:
                self.preview_chan3.img = ph

        show_channel(Y, self.preview_chan1)
        show_channel(M, self.preview_chan2)
        show_channel(C, self.preview_chan3)

    def convertir_hsv(self):
        if self.img_loaded is None:
            messagebox.showwarning("Aviso", "Primero cargue una imagen en la pestaña 'Formatos'.")
            return
        hsv = cv2.cvtColor(self.img_loaded, cv2.COLOR_BGR2HSV)
        H = hsv[:, :, 0]
        S = hsv[:, :, 1]
        V = hsv[:, :, 2]
        # Show original
        self.preview_photo_4 = pil_to_tk(cv2_to_pil(self.img_loaded), maxsize=(340,240))
        self.label_color_orig.configure(image=self.preview_photo_4)

        def show_gray(arr, widget, title=None):
            pil = Image.fromarray(arr)
            ph = pil_to_tk(pil, maxsize=(300,300))
            widget.configure(image=ph)
            widget.img = ph

        show_gray(H, self.preview_chan1)
        show_gray(S, self.preview_chan2)
        show_gray(V, self.preview_chan3)

    def convertir_lab(self):
        if self.img_loaded is None:
            messagebox.showwarning("Aviso", "Primero cargue una imagen en la pestaña 'Formatos'.")
            return
        # Convert to LAB using OpenCV (BGR->LAB)
        lab = cv2.cvtColor(self.img_loaded, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0]
        A = lab[:, :, 1]
        B = lab[:, :, 2]
        # Show original
        self.preview_photo_4 = pil_to_tk(cv2_to_pil(self.img_loaded), maxsize=(340,240))
        self.label_color_orig.configure(image=self.preview_photo_4)

        def show_gray(arr, widget):
            pil = Image.fromarray(arr)
            ph = pil_to_tk(pil, maxsize=(300,300))
            widget.configure(image=ph)
            widget.img = ph

        show_gray(L, self.preview_chan1)
        show_gray(A, self.preview_chan2)
        show_gray(B, self.preview_chan3)

    # ---------------- TAB 4: OPERACIONES ----------------
    def build_tab_operaciones(self):
        frame = self.tab4

        top_frame = ttk.Frame(frame)
        top_frame.pack(side='top', fill='x', pady=(0,6))

        ttk.Button(top_frame, text="Cargar 2da Imagen", command=self.cargar_imagen2).pack(side='left', padx=6)
        ttk.Button(top_frame, text="Suma", command=self.sumar_imagenes).pack(side='left', padx=6)
        ttk.Button(top_frame, text="Resta", command=self.restar_imagenes).pack(side='left', padx=6)
        ttk.Button(top_frame, text="Multiplicación", command=self.multiplicar_imagenes).pack(side='left', padx=6)
        ttk.Button(top_frame, text="División", command=self.dividir_imagenes).pack(side='left', padx=6)
        ttk.Button(top_frame, text="AND", command=self.and_imagenes).pack(side='left', padx=6)
        ttk.Button(top_frame, text="OR", command=self.or_imagenes).pack(side='left', padx=6)
        ttk.Button(top_frame, text="XOR", command=self.xor_imagenes).pack(side='left', padx=6)
        ttk.Button(top_frame, text="Umbralización", command=self.umbralizar_imagen).pack(side='left', padx=6)

        # Slider area
        slider_frame = ttk.Frame(frame)
        slider_frame.pack(fill='x', pady=(4,8), padx=6)
        ttk.Label(slider_frame, text="Umbral:").pack(side='left')
        self.slider = ttk.Scale(slider_frame, from_=0.0, to=1.0, orient='horizontal', command=self.on_slider_change)
        self.slider.set(0.5)
        self.slider.pack(side='left', fill='x', expand=True, padx=6)
        self.slider_label = ttk.Label(slider_frame, text="Umbral = 0.50")
        self.slider_label.pack(side='left', padx=6)
        # Start hidden
        self.slider.pack_forget()
        self.slider_label.pack_forget()

        # Result preview
        result_frame = ttk.LabelFrame(frame, text="Resultado")
        result_frame.pack(fill='both', expand=True, padx=8, pady=8)
        self.label_result = ttk.Label(result_frame)
        self.label_result.pack(fill='both', expand=True)

    def cargar_imagen2(self):
        path = filedialog.askopenfilename(filetypes=[("Imagen", "*.jpg;*.png;*.bmp;*.tiff;*.jpeg"),("Todos","*.*")])
        if not path:
            return
        img = safe_imread(path)
        if img is None:
            messagebox.showerror("Error", "No se pudo leer la imagen.")
            return
        self.img2_loaded = img.copy()
        messagebox.showinfo("Info", "Segunda imagen cargada correctamente (usa las operaciones).")

    def ensure_same_size(self, a, b):
        # Resize b to a's size
        if a is None or b is None:
            return None
        if a.shape[:2] == b.shape[:2]:
            return b
        return cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)

    def show_result_cv(self, cv_img):
        if cv_img is None:
            return
        # convert to displayable PIL and set label
        pil = cv2_to_pil(cv_img)
        ph = pil_to_tk(pil, maxsize=(900,380))
        self.label_result.configure(image=ph)
        self.label_result.image = ph

    def sumar_imagenes(self):
        if self.img_loaded is None or self.img2_loaded is None:
            messagebox.showwarning("Aviso", "Cargue la imagen 1 (Formatos) y la imagen 2 (Operaciones).")
            return
        b = self.ensure_same_size(self.img_loaded, self.img2_loaded)
        res = cv2.add(self.img_loaded, b)  # saturating add
        self.show_result_cv(res)

    def restar_imagenes(self):
        if self.img_loaded is None or self.img2_loaded is None:
            messagebox.showwarning("Aviso", "Cargue la imagen 1 (Formatos) y la imagen 2 (Operaciones).")
            return
        b = self.ensure_same_size(self.img_loaded, self.img2_loaded)
        res = cv2.subtract(self.img_loaded, b)
        self.show_result_cv(res)

    def multiplicar_imagenes(self):
        if self.img_loaded is None or self.img2_loaded is None:
            messagebox.showwarning("Aviso", "Cargue la imagen 1 (Formatos) y la imagen 2 (Operaciones).")
            return
        b = self.ensure_same_size(self.img_loaded, self.img2_loaded).astype(np.float32)/255.0
        a = self.img_loaded.astype(np.float32)/255.0
        resf = a * b
        res = np.clip((resf*255.0), 0, 255).astype(np.uint8)
        self.show_result_cv(res)

    def dividir_imagenes(self):
        if self.img_loaded is None or self.img2_loaded is None:
            messagebox.showwarning("Aviso", "Cargue la imagen 1 (Formatos) y la imagen 2 (Operaciones).")
            return
        b = self.ensure_same_size(self.img_loaded, self.img2_loaded).astype(np.float32)/255.0
        a = self.img_loaded.astype(np.float32)/255.0
        # avoid division by zero
        b = np.where(b == 0, 1e-6, b)
        resf = a / b
        res = np.clip((resf*255.0), 0, 255).astype(np.uint8)
        self.show_result_cv(res)

    def and_imagenes(self):
        if self.img_loaded is None or self.img2_loaded is None:
            messagebox.showwarning("Aviso", "Cargue las dos imágenes primero.")
            return
        a = cv2.cvtColor(self.img_loaded, cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(self.ensure_same_size(self.img_loaded, self.img2_loaded), cv2.COLOR_BGR2GRAY)
        # threshold both at 0.5
        _, ta = cv2.threshold(a, 127, 255, cv2.THRESH_BINARY)
        _, tb = cv2.threshold(b, 127, 255, cv2.THRESH_BINARY)
        res = cv2.bitwise_and(ta, tb)
        self.show_result_cv(res)

    def or_imagenes(self):
        if self.img_loaded is None or self.img2_loaded is None:
            messagebox.showwarning("Aviso", "Cargue las dos imágenes primero.")
            return
        a = cv2.cvtColor(self.img_loaded, cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(self.ensure_same_size(self.img_loaded, self.img2_loaded), cv2.COLOR_BGR2GRAY)
        _, ta = cv2.threshold(a, 127, 255, cv2.THRESH_BINARY)
        _, tb = cv2.threshold(b, 127, 255, cv2.THRESH_BINARY)
        res = cv2.bitwise_or(ta, tb)
        self.show_result_cv(res)

    def xor_imagenes(self):
        if self.img_loaded is None or self.img2_loaded is None:
            messagebox.showwarning("Aviso", "Cargue las dos imágenes primero.")
            return
        a = cv2.cvtColor(self.img_loaded, cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(self.ensure_same_size(self.img_loaded, self.img2_loaded), cv2.COLOR_BGR2GRAY)
        _, ta = cv2.threshold(a, 127, 255, cv2.THRESH_BINARY)
        _, tb = cv2.threshold(b, 127, 255, cv2.THRESH_BINARY)
        res = cv2.bitwise_xor(ta, tb)
        self.show_result_cv(res)

    def umbralizar_imagen(self):
        if self.img_loaded is None:
            messagebox.showwarning("Aviso", "Primero cargue una imagen en la pestaña 'Formatos'.")
            return
        # Show slider and label
        self.slider.pack(side='left', fill='x', expand=True, padx=6)
        self.slider_label.pack(side='left', padx=6)
        self.slider.set(0.5)
        self.on_slider_change(None)

    def on_slider_change(self, _):
        if self.img_loaded is None:
            return
        thresh = float(self.slider.get())
        self.slider_label.config(text=f"Umbral = {thresh:.2f}")
        gray = cv2.cvtColor(self.img_loaded, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold((gray*1.0).astype(np.uint8), int(thresh*255), 255, cv2.THRESH_BINARY)
        self.gray_img = gray
        self.show_result_cv(bw)

    # ---------------- TAB 5: COMPARACIONES ----------------
    def build_tab_comparaciones(self):
        frame = self.tab5

        top_controls = ttk.Frame(frame)
        top_controls.pack(side='top', fill='x', pady=(0,8))

        ttk.Button(top_controls, text="Cargar Imagen 1", command=self.cargar_c1).pack(side='left', padx=6)
        ttk.Button(top_controls, text="Cargar Imagen 2", command=self.cargar_c2).pack(side='left', padx=6)
        ttk.Button(top_controls, text="Comparar", command=self.comparar_imagenes).pack(side='left', padx=6)

        preview_frame = ttk.Frame(frame)
        preview_frame.pack(fill='both', expand=True)

        left = ttk.LabelFrame(preview_frame, text="Imagen 1")
        left.pack(side='left', fill='both', expand=True, padx=8, pady=8)
        self.label_cmp_1 = ttk.Label(left)
        self.label_cmp_1.pack(fill='both', expand=True)

        right = ttk.LabelFrame(preview_frame, text="Imagen 2")
        right.pack(side='left', fill='both', expand=True, padx=8, pady=8)
        self.label_cmp_2 = ttk.Label(right)
        self.label_cmp_2.pack(fill='both', expand=True)

        bottom = ttk.LabelFrame(frame, text="Diferencia")
        bottom.pack(fill='x', padx=8, pady=8)
        self.label_cmp_diff = ttk.Label(bottom)
        self.label_cmp_diff.pack(fill='both', expand=True)

        # Stats box on the right
        stats_frame = ttk.Frame(frame)
        stats_frame.pack(side='right', fill='y', padx=8)
        ttk.Label(stats_frame, text="Métricas", font=('Segoe UI', 10, 'bold')).pack()
        self.listbox_stats = tk.Listbox(stats_frame, width=30, height=10)
        self.listbox_stats.pack(fill='y', pady=4)

    def cargar_c1(self):
        path = filedialog.askopenfilename(filetypes=[("Imagen", "*.jpg;*.png;*.bmp;*.tiff;*.jpeg"),("Todos","*.*")])
        if not path:
            return
        img = safe_imread(path)
        if img is None:
            messagebox.showerror("Error", "No se pudo leer la imagen 1.")
            return
        self.imgC1 = img.copy()
        ph = pil_to_tk(cv2_to_pil(self.imgC1), maxsize=(400,400))
        self.label_cmp_1.configure(image=ph)
        self.label_cmp_1.image = ph

    def cargar_c2(self):
        path = filedialog.askopenfilename(filetypes=[("Imagen", "*.jpg;*.png;*.bmp;*.tiff;*.jpeg"),("Todos","*.*")])
        if not path:
            return
        img = safe_imread(path)
        if img is None:
            messagebox.showerror("Error", "No se pudo leer la imagen 2.")
            return
        self.imgC2 = img.copy()
        ph = pil_to_tk(cv2_to_pil(self.imgC2), maxsize=(400,400))
        self.label_cmp_2.configure(image=ph)
        self.label_cmp_2.image = ph

    def comparar_imagenes(self):
        if self.imgC1 is None or self.imgC2 is None:
            messagebox.showwarning("Aviso", "Cargue ambas imágenes para comparar.")
            return
        # Resize to 256x256 as in MATLAB
        img1r = cv2.resize(self.imgC1, (256, 256), interpolation=cv2.INTER_AREA)
        img2r = cv2.resize(self.imgC2, (256, 256), interpolation=cv2.INTER_AREA)
        # Convert to grayscale for diff display
        g1 = cv2.cvtColor(img1r, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(img2r, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(g1, g2)
        ph = pil_to_tk(Image.fromarray(diff), maxsize=(900,200))
        self.label_cmp_diff.configure(image=ph)
        self.label_cmp_diff.image = ph

        # PSNR and SSIM (use color arrays for more accuracy if desired)
        psnr_val, ssim_val = compute_psnr_ssim(img1r, img2r)
        self.listbox_stats.delete(0, tk.END)
        self.listbox_stats.insert(tk.END, f"PSNR: {psnr_val:.4f}")
        self.listbox_stats.insert(tk.END, f"SSIM: {ssim_val:.4f}")
        self.listbox_stats.insert(tk.END, f"Tamaño Img1: {self.imgC1.shape[1]}x{self.imgC1.shape[0]}")
        self.listbox_stats.insert(tk.END, f"Tamaño Img2: {self.imgC2.shape[1]}x{self.imgC2.shape[0]}")

# ---------- Run ----------
if __name__ == "__main__":
    root = tk.Tk()
    app = ProyectoImagenesApp(root)
    root.mainloop()
