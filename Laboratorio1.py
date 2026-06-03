# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 23:46:44 2025

@author: karen
"""

"""
image_toolkit.py
Aplicación Tkinter para:
- Carga en jpg/png/bmp/tiff
- Mostrar propiedades (size, canales, depth estimada)
- Redimensionar, calcular PSNR/SSIM
- Convertir RGB->CMY (y mostrar canales)
- Operaciones básicas: suma, resta, multiplicación por constante, umbral
- Comparar dos imágenes: mapa de diferencia y estadísticas

Requisitos:
pip install numpy Pillow opencv-python scikit-image matplotlib
"""

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk, ImageOps
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from io import BytesIO

# ---------- Helpers ----------
def pil_to_tk(img_pil, maxsize=(400,400)):
    img = img_pil.copy()
    img.thumbnail(maxsize, Image.ANTIALIAS)
    return ImageTk.PhotoImage(img)

def open_image(path):
    pil = Image.open(path).convert('RGB')
    return pil

def get_properties(pil_img, path=None):
    w,h = pil_img.size
    mode = pil_img.mode
    # Depth estimate: bits per channel if info available, else assume 8
    depth = pil_img.bits if hasattr(pil_img, "bits") else 8
    dpi = pil_img.info.get('dpi', (72,72))
    props = {
        'width': w, 'height': h, 'mode': mode, 'depth': depth, 'dpi': dpi, 'path': path
    }
    return props

def pil_to_ndarray(pil):
    return np.array(pil).astype(np.uint8)

def ndarray_to_pil(arr):
    return Image.fromarray(arr.astype(np.uint8))

# ---------- Color conversions ----------
def rgb_to_cmy(rgb_arr):
    # input 0-255 uint8
    norm = rgb_arr.astype(np.float32)/255.0
    cmy = 1.0 - norm  # CMY in 0..1
    return (cmy * 255).astype(np.uint8)

def rgb_to_y_channel(rgb_arr):
    # luma
    arr = rgb_arr.astype(np.float32)/255.0
    y = 0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]
    return (y*255).astype(np.uint8)

def rgb_to_hsv_pil(pil):
    return pil.convert('HSV')

def rgb_to_lab_ndarray(rgb_arr):
    # OpenCV expects BGR
    bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return lab

# ---------- Metrics ----------
def compute_psnr(img_ref_arr, img_test_arr):
    # ensure same shape and range 0..255
    return psnr(img_ref_arr, img_test_arr, data_range=255)

def compute_ssim(img_ref_arr, img_test_arr):
    # compute per-channel and mean
    if img_ref_arr.ndim==3:
        svals = []
        for ch in range(img_ref_arr.shape[2]):
            s, _ = ssim(img_ref_arr[:,:,ch], img_test_arr[:,:,ch], full=True, data_range=255)
            svals.append(s)
        return float(np.mean(svals))
    else:
        s, _ = ssim(img_ref_arr, img_test_arr, full=True, data_range=255)
        return float(s)

# ---------- Basic ops ----------
def add_images(a, b):
    return np.clip(a.astype(np.int32) + b.astype(np.int32), 0, 255).astype(np.uint8)

def sub_images(a, b):
    return np.clip(a.astype(np.int32) - b.astype(np.int32), 0, 255).astype(np.uint8)

def mul_const(a, k):
    return np.clip(a.astype(np.float32) * k, 0, 255).astype(np.uint8)

def threshold_image(a_gray, thresh):
    out = (a_gray >= thresh).astype(np.uint8)*255
    return out

# ---------- GUI ----------
class ImageToolkitApp:
    def __init__(self, root):
        self.root = root
        root.title("Image Toolkit - Actividades")
        self.left_frame = ttk.Frame(root, padding=6)
        self.left_frame.grid(row=0,column=0, sticky='nsew')
        self.right_frame = ttk.Frame(root, padding=6)
        self.right_frame.grid(row=0,column=1, sticky='nsew')

        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        # State
        self.pil_img = None
        self.img_arr = None
        self.compare_img = None
        self.compare_arr = None

        # Controls
        ttk.Button(self.left_frame, text="Cargar imagen", command=self.load_image).pack(fill='x')
        ttk.Button(self.left_frame, text="Cargar imagen 2 (para comparar)", command=self.load_compare_image).pack(fill='x')
        self.props_text = tk.Text(self.left_frame, height=6)
        self.props_text.pack(fill='x', pady=4)

        # Resize controls
        resize_box = ttk.LabelFrame(self.left_frame, text="Resolución / Redimensionar")
        resize_box.pack(fill='x', pady=4)
        ttk.Label(resize_box, text="Ancho:").grid(row=0,column=0)
        self.width_entry = ttk.Entry(resize_box, width=8)
        self.width_entry.grid(row=0,column=1)
        ttk.Label(resize_box, text="Alto:").grid(row=0,column=2)
        self.height_entry = ttk.Entry(resize_box, width=8)
        self.height_entry.grid(row=0,column=3)
        ttk.Button(resize_box, text="Aplicar", command=self.apply_resize).grid(row=0,column=4, padx=4)
        ttk.Button(resize_box, text="Evaluar PSNR/SSIM vs original", command=self.evaluate_resize).grid(row=1,column=0, columnspan=5, pady=4)

        # Color conversion
        color_box = ttk.LabelFrame(self.left_frame, text="Conversiones de color")
        color_box.pack(fill='x', pady=4)
        ttk.Button(color_box, text="Mostrar canales CMY", command=self.show_cmy).pack(fill='x')
        ttk.Button(color_box, text="Mostrar Y (luma)", command=self.show_y).pack(fill='x')
        ttk.Button(color_box, text="Mostrar HSV y LAB", command=self.show_hsv_lab).pack(fill='x')

        # Basic ops
        ops_box = ttk.LabelFrame(self.left_frame, text="Operaciones básicas")
        ops_box.pack(fill='x', pady=4)
        ttk.Button(ops_box, text="Suma con imagen 2", command=self.op_add).pack(fill='x')
        ttk.Button(ops_box, text="Resta con imagen 2", command=self.op_sub).pack(fill='x')
        ttk.Label(ops_box, text="Multiplicar por (k):").pack()
        self.k_entry = ttk.Entry(ops_box, width=8)
        self.k_entry.insert(0, "1.2")
        self.k_entry.pack()
        ttk.Button(ops_box, text="Aplicar multiplicación", command=self.op_mul).pack(fill='x')
        ttk.Label(ops_box, text="Umbralización (0-255):").pack()
        self.t_entry = ttk.Entry(ops_box, width=8)
        self.t_entry.insert(0, "128")
        self.t_entry.pack()
        ttk.Button(ops_box, text="Aplicar umbral (canal Y)", command=self.op_threshold).pack(fill='x')

        # Compare / extra
        extra_box = ttk.LabelFrame(self.left_frame, text="Comparar (Punto extra)")
        extra_box.pack(fill='x', pady=4)
        ttk.Button(extra_box, text="Comparar imagen y imagen2", command=self.compare_two_images).pack(fill='x')
        self.compare_text = tk.Text(extra_box, height=6)
        self.compare_text.pack(fill='x')

        # Canvas & image displays on right
        self.canvas_top = ttk.Frame(self.right_frame)
        self.canvas_top.pack(fill='both', expand=True)
        self.display_label = ttk.Label(self.canvas_top)
        self.display_label.pack()

        # small preview for image2
        self.display_compare_label = ttk.Label(self.canvas_top)
        self.display_compare_label.pack()

    # ---------- UI actions ----------
    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images","*.jpg *.jpeg *.png *.bmp *.tiff")])
        if not path:
            return
        self.pil_img = open_image(path)
        self.img_arr = pil_to_ndarray(self.pil_img)
        props = get_properties(self.pil_img, path)
        self.show_properties(props)
        self.show_pil(self.pil_img, target='main')

    def load_compare_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images","*.jpg *.jpeg *.png *.bmp *.tiff")])
        if not path:
            return
        pil = open_image(path)
        self.compare_img = pil
        self.compare_arr = pil_to_ndarray(pil)
        self.show_pil(pil, target='compare')

    def show_properties(self, props):
        self.props_text.delete('1.0','end')
        self.props_text.insert('end', f"Ruta: {props['path']}\n")
        self.props_text.insert('end', f"Tamaño: {props['width']} x {props['height']} px\n")
        self.props_text.insert('end', f"Modo: {props['mode']}\n")
        self.props_text.insert('end', f"Profundidad estimada por canal: {props['depth']} bits\n")
        self.props_text.insert('end', f"DPI: {props['dpi']}\n")

        self.width_entry.delete(0, 'end'); self.width_entry.insert(0, str(props['width']))
        self.height_entry.delete(0, 'end'); self.height_entry.insert(0, str(props['height']))

    def show_pil(self, pil_img, target='main'):
        tkimg = pil_to_tk(pil_img, maxsize=(600,400))
        if target=='main':
            self.display_label.configure(image=tkimg)
            self.display_label.image = tkimg
        else:
            self.display_compare_label.configure(image=tkimg)
            self.display_compare_label.image = tkimg

    def apply_resize(self):
        if self.pil_img is None:
            messagebox.showwarning("Aviso", "Carga primero una imagen.")
            return
        try:
            w = int(self.width_entry.get()); h = int(self.height_entry.get())
        except:
            messagebox.showerror("Error","Ancho/Alto inválidos")
            return
        resized = self.pil_img.resize((w,h), Image.LANCZOS)
        self.show_pil(resized, target='main')
        # update state (but keep original too)
        self.img_arr = pil_to_ndarray(resized)
        self.pil_img = resized
        self.show_properties(get_properties(self.pil_img))

    def evaluate_resize(self):
        # compare current displayed image with original if compare image exists use it
        if self.compare_arr is not None:
            ref = self.compare_arr
            test = self.img_arr
            # if sizes differ: resize ref to test size
            if ref.shape != test.shape:
                ref_rs = cv2.resize(ref, (test.shape[1], test.shape[0]), interpolation=cv2.INTER_AREA)
            else:
                ref_rs = ref
        else:
            messagebox.showinfo("Info", "Para evaluar PSNR/SSIM, carga una imagen 2 como referencia (imagen original).")
            return
        val_psnr = compute_psnr(ref_rs, test)
        val_ssim = compute_ssim(ref_rs, test)
        messagebox.showinfo("Resultados", f"PSNR: {val_psnr:.2f} dB\nSSIM: {val_ssim:.4f}")

    def show_cmy(self):
        if self.img_arr is None:
            messagebox.showwarning("Aviso", "Carga una imagen primero.")
            return
        cmy = rgb_to_cmy(self.img_arr)  # uint8
        # show each channel
        C = cmy[:,:,0]; M=cmy[:,:,1]; Y=cmy[:,:,2]
        fig, axs = plt.subplots(1,4, figsize=(12,4))
        axs[0].imshow(self.img_arr); axs[0].set_title('RGB (original)'); axs[0].axis('off')
        axs[1].imshow(C, cmap='gray'); axs[1].set_title('C (Cian)'); axs[1].axis('off')
        axs[2].imshow(M, cmap='gray'); axs[2].set_title('M (Magenta)'); axs[2].axis('off')
        axs[3].imshow(Y, cmap='gray'); axs[3].set_title('Y (Yellow)'); axs[3].axis('off')
        plt.tight_layout(); plt.show()

    def show_y(self):
        if self.img_arr is None:
            messagebox.showwarning("Aviso","Carga imagen")
            return
        y = rgb_to_y_channel(self.img_arr)
        pil = Image.fromarray(y)
        self.show_pil(pil, target='main')

    def show_hsv_lab(self):
        if self.pil_img is None:
            messagebox.showwarning("Aviso","Carga imagen")
            return
        hsv = rgb_to_hsv_pil(self.pil_img)
        lab_arr = rgb_to_lab_ndarray(self.img_arr)
        L = lab_arr[:,:,0]
        fig, axs = plt.subplots(1,3, figsize=(12,4))
        axs[0].imshow(self.pil_img); axs[0].set_title('RGB'); axs[0].axis('off')
        axs[1].imshow(hsv); axs[1].set_title('HSV'); axs[1].axis('off')
        axs[2].imshow(L, cmap='gray'); axs[2].set_title('LAB - L channel'); axs[2].axis('off')
        plt.tight_layout(); plt.show()

    # ops that use image2
    def op_add(self):
        if self.img_arr is None or self.compare_arr is None:
            messagebox.showwarning("Aviso", "Carga ambas imágenes")
            return
        a = self.img_arr; b = cv2.resize(self.compare_arr, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
        out = add_images(a,b)
        self.show_pil(ndarray_to_pil(out), target='main')

    def op_sub(self):
        if self.img_arr is None or self.compare_arr is None:
            messagebox.showwarning("Aviso", "Carga ambas imágenes")
            return
        a = self.img_arr; b = cv2.resize(self.compare_arr, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
        out = sub_images(a,b)
        self.show_pil(ndarray_to_pil(out), target='main')

    def op_mul(self):
        if self.img_arr is None:
            messagebox.showwarning("Aviso", "Carga imagen")
            return
        try:
            k = float(self.k_entry.get())
        except:
            messagebox.showerror("Error", "k inválido")
            return
        out = mul_const(self.img_arr, k)
        self.show_pil(ndarray_to_pil(out), target='main')

    def op_threshold(self):
        if self.img_arr is None:
            messagebox.showwarning("Aviso", "Carga imagen")
            return
        try:
            t = int(self.t_entry.get())
        except:
            messagebox.showerror("Error", "Umbral inválido")
            return
        y = rgb_to_y_channel(self.img_arr)
        out = threshold_image(y, t)
        self.show_pil(Image.fromarray(out), target='main')

    def compare_two_images(self):
        if self.img_arr is None or self.compare_arr is None:
            messagebox.showwarning("Aviso", "Carga ambas imágenes para comparar")
            return
        a = self.img_arr
        b = cv2.resize(self.compare_arr, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
        val_psnr = compute_psnr(b,a)
        val_ssim = compute_ssim(b,a)
        diff = np.abs(a.astype(int)-b.astype(int)).astype(np.uint8)
        # show results
        self.compare_text.delete('1.0','end')
        self.compare_text.insert('end', f"PSNR: {val_psnr:.2f} dB\nSSIM: {val_ssim:.4f}\n")
        # show difference map
        # combine channels of diff into a heatmap (sum)
        diff_gray = np.clip(np.mean(diff, axis=2), 0, 255).astype(np.uint8)
        fig, axs = plt.subplots(1,3, figsize=(12,4))
        axs[0].imshow(a); axs[0].set_title('Imagen A'); axs[0].axis('off')
        axs[1].imshow(b); axs[1].set_title('Imagen B (referencia)'); axs[1].axis('off')
        axs[2].imshow(diff_gray, cmap='hot'); axs[2].set_title('Mapa diferencia (promedio canales)'); axs[2].axis('off')
        plt.tight_layout(); plt.show()

# ---------- Run ----------
def main():
    root = tk.Tk()
    app = ImageToolkitApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
