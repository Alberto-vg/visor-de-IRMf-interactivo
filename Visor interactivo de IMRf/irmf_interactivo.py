import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# Cargar volumen NIfTI
img = nib.load('sub-01_T1w.nii')
vol = img.get_fdata()

# tomar el primer volumen (t=0)
if vol.ndim == 4:
    vol = vol[..., 0]

# Asegurar tipo float y reemplazar posibles NaN/Inf
vol = np.asarray(vol, dtype=np.float32)
vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)

# Tamaños e índices centrales
sx, sy, sz = vol.shape  # (X, Y, Z)
mx = sx // 2
my = sy // 2
mz = sz // 2

# Utilidades de imagen
def normalizar(im):
    vmin = np.percentile(im, 2.0)
    vmax = np.percentile(im, 98.0)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.min(im)), float(np.max(im))
    if vmax <= vmin:
        return np.zeros_like(im, dtype=np.float32)
    im = (im - vmin) / (vmax - vmin)
    im = np.clip(im, 0.0, 1.0).astype(np.float32)
    return im

def cortes_rotados(volumen, x_idx, y_idx, z_idx):
    """
    Devuelve (sagital_v, coronal_v, axial_v) normalizados y rotados
    para visualización consistente.
    - Sagital: X fijo = x_idx -> (Y, Z)   -> rot90
    - Coronal: Y fijo = y_idx -> (X, Z)   -> rot90
    - Axial:   Z fijo = z_idx -> (X, Y)   -> rot90
    """
    sag = volumen[x_idx, :, :]    # (sy, sz)
    cor = volumen[:, y_idx, :]    # (sx, sz)
    axi = volumen[:, :, z_idx]    # (sx, sy)

    sag_v = normalizar(np.rot90(sag))
    cor_v = normalizar(np.rot90(cor))
    axi_v = normalizar(np.rot90(axi))
    return sag_v, cor_v, axi_v

# Preparar vistas
x_idx = mx
y_idx = my
z_idx = mz

sag_v, cor_v, axi_v = cortes_rotados(vol, x_idx, y_idx, z_idx)

fig = plt.figure(figsize=(10, 6))
gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.4], height_ratios=[1, 1],
                      wspace=0.08, hspace=0.08)

ax_sag = fig.add_subplot(gs[:, 0])   # izquierda (2 filas)
ax_axi = fig.add_subplot(gs[0, 1])   # arriba derecha
ax_cor = fig.add_subplot(gs[1, 1])   # abajo derecha

im_sag = ax_sag.imshow(sag_v, origin='lower', aspect='equal')
ax_sag.set_title("Sagital (navegación)\nMueve el cursor aquí", fontsize=10)
ax_sag.set_xticks([]); ax_sag.set_yticks([])

im_axi = ax_axi.imshow(axi_v, origin='lower', aspect='equal')
ax_axi.set_title(f"Axial (z={z_idx})", fontsize=10)
ax_axi.set_xticks([]); ax_axi.set_yticks([])

im_cor = ax_cor.imshow(cor_v, origin='lower', aspect='equal')
ax_cor.set_title(f"Coronal (y={y_idx})", fontsize=10)
ax_cor.set_xticks([]); ax_cor.set_yticks([])

# Cruz en sagital (coordenadas pantalla ~ (y, z) tras rotación)
lh = ax_sag.axhline(y_idx, linewidth=0.8)
lv = ax_sag.axvline(z_idx, linewidth=0.8)

# Estado mutable para evitar problemas de scope
state = {'y_idx': y_idx, 'z_idx': z_idx}

# Interacción con cursor
def on_move(event):
    if event.inaxes is not ax_sag:
        return
    if event.xdata is None or event.ydata is None:
        return

    # En sag_v: columna ~ z, fila ~ y
    z_new = int(round(event.xdata))
    y_new = int(round(event.ydata))

    # Limitar a rangos válidos
    if z_new < 0: z_new = 0
    if z_new > sz - 1: z_new = sz - 1
    if y_new < 0: y_new = 0
    if y_new > sy - 1: y_new = sy - 1

    if z_new == state['z_idx'] and y_new == state['y_idx']:
        return

    state['z_idx'] = z_new
    state['y_idx'] = y_new

    # Recalcular cortes
    sag_v2, cor_v2, axi_v2 = cortes_rotados(vol, x_idx, state['y_idx'], state['z_idx'])

    # Actualizar imágenes
    im_sag.set_data(sag_v2)
    im_cor.set_data(cor_v2)
    im_axi.set_data(axi_v2)

    # Actualizar títulos e indicadores
    ax_axi.set_title(f"Axial (z={state['z_idx']})", fontsize=10)
    ax_cor.set_title(f"Coronal (y={state['y_idx']})", fontsize=10)
    lh.set_ydata([state['y_idx'], state['y_idx']])
    lv.set_xdata([state['z_idx'], state['z_idx']])

    fig.canvas.draw_idle()

fig.canvas.mpl_connect('motion_notify_event', on_move)

plt.show()

