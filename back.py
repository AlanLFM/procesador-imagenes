# back.py

import os
import io
import base64
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from skimage.filters import roberts, threshold_multiotsu
from skimage import img_as_ubyte
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory  # <-- send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# -------------------------------------------------------------------
# (CAMBIO) Configurar Flask para servir archivos estáticos de esta carpeta
# -------------------------------------------------------------------
#
#    static_folder='.'       → indica: “Los archivos estáticos se buscan en la carpeta actual (donde está back.py)”
#    static_url_path=''      → hace que, al solicitar /estilos.css, Flask lo sirva directamente desde ./estilos.css
#
app = Flask(
    __name__,
    static_folder='.',       # carpeta actual como origen de archivos estáticos (main.html, estilos.css)
    static_url_path=''       # la URL base para estáticos es la raíz; de modo que /estilos.css carga ./estilos.css
)
CORS(app)

# -------------------------------------------------------------------
# Configuración de carpetas para subir y guardar resultados en disco
# -------------------------------------------------------------------
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# -------------------------------------------------------------------
# Funciones auxiliares (misma lógica que antes)
# -------------------------------------------------------------------
def agregar_sal_pimienta(img, cantidad):
    salida = np.copy(img)
    num_pix = int(cantidad * img.size)
    coords = [np.random.randint(0, i-1, num_pix) for i in img.shape]
    salida[coords[0], coords[1]] = 255
    coords = [np.random.randint(0, i-1, num_pix) for i in img.shape]
    salida[coords[0], coords[1]] = 0
    return salida

def agregar_gaussiano(img, media, sigma):
    media = np.mean(img)
    gauss = np.random.normal(media, sigma, img.shape).astype(np.uint8)
    salida = cv2.add(img, gauss)
    return salida

def ecualizacion_log_hiperbolica(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    r_min = np.min(img)
    r_max = np.max(img)
    hist, bins = np.histogram(img.flatten(), bins=256, range=[0, 256], density=True)
    cdf = hist.cumsum()
    cdf_normalizada = cdf * ((r_max - r_min) / (1 if r_max == r_min else (r_max - r_min)))
    transformacion = np.floor(r_min + (r_max - r_min) * cdf_normalizada).astype(np.uint8)
    img_ecualizada = transformacion[img]
    return img_ecualizada

def aplicar_roberts(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imagen_float = np.float32(img)
    kernel_roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    kernel_roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    roberts_x = cv2.filter2D(imagen_float, -1, kernel_roberts_x)
    roberts_y = cv2.filter2D(imagen_float, -1, kernel_roberts_y)
    roberts_img = cv2.sqrt(cv2.addWeighted(cv2.pow(roberts_x, 2.0), 1.0,
                                           cv2.pow(roberts_y, 2.0), 1.0, 0.0))
    roberts_img = np.clip(roberts_img, 0, 255)
    return np.uint8(roberts_img)

def aplicar_canny(img, low_threshold=50, high_threshold=150):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    suavizada = cv2.GaussianBlur(img, (5, 5), 0)
    bordes = cv2.Canny(suavizada, low_threshold, high_threshold)
    return bordes

def aplicar_multiumbral(img, classes=3):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    umbrales = threshold_multiotsu(img, classes=classes)
    segmentos = np.digitize(img, bins=umbrales)
    segmentos = np.uint8(255 * segmentos / np.max(segmentos))
    return segmentos

def extraer_canales_rgb(img):
    if len(img.shape) == 3:
        b, g, r = cv2.split(img)
        cero = np.zeros_like(b)
        canal_r = cv2.merge([cero, cero, r])
        canal_g = cv2.merge([cero, g, cero])
        canal_b = cv2.merge([b, cero, cero])
        canal_r = cv2.cvtColor(canal_r, cv2.COLOR_BGR2RGB)
        canal_g = cv2.cvtColor(canal_g, cv2.COLOR_BGR2RGB)
        canal_b = cv2.cvtColor(canal_b, cv2.COLOR_BGR2RGB)
        return canal_r, canal_g, canal_b
    else:
        return img, img, img

def calcular_histograma(img):
    if len(img.shape) == 3:
        b, g, r = cv2.split(img)
        hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
        hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist_gray = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
        return hist_r, hist_g, hist_b, hist_gray
    else:
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        return hist

def operaciones_aritmeticas(img, escalar):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    suma = cv2.add(img, escalar)
    resta = cv2.subtract(img, escalar)
    multiplicacion = cv2.multiply(img, escalar)
    return suma, resta, multiplicacion

def operaciones_logicas(img1, img2):
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    h, w = img1.shape
    img2 = cv2.resize(img2, (w, h))
    and_img = cv2.bitwise_and(img1, img2)
    or_img = cv2.bitwise_or(img1, img2)
    xor_img = cv2.bitwise_xor(img1, img2)
    return and_img, or_img, xor_img

def aplicar_filtros(img, filtro_tipo):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if filtro_tipo == 'gaussiano':
        return cv2.GaussianBlur(img, (5, 5), 1)
    elif filtro_tipo == 'promedio':
        return cv2.blur(img, (5, 5))
    elif filtro_tipo == 'mediana':
        return cv2.medianBlur(img, 5)
    else:
        return img

def imagen_a_base64(img):
    if len(img.shape) == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img
    pil_img = Image.fromarray(img_rgb)
    img_buffer = io.BytesIO()
    pil_img.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_base64}"

def figura_a_base64(fig):
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{img_base64}"

# -------------------------------------------------------------------
# 3. Ruta raíz: servir main.html
# -------------------------------------------------------------------
#
# Gracias a static_folder='.' y static_url_path='', cuando alguien visite '/'
# Flask buscará primero un archivo estático llamado 'index.html' o 'main.html'.
# Para mayor claridad, lo devolvemos explícitamente:
#
@app.route('/', methods=['GET'])
def index():
    # Devuelve main.html desde la carpeta actual
    return send_from_directory('.', 'main.html')

# -------------------------------------------------------------------
# 4. Rutas de la API (sin cambios)
# -------------------------------------------------------------------
@app.route('/upload', methods=['POST'])
def upload_image():
    """Subir imagen"""
    if 'image' not in request.files:
        return jsonify({'error': 'No se encontró archivo de imagen'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó archivo'}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img = cv2.imread(filepath)
        if img is None:
            return jsonify({'error': 'No se pudo cargar la imagen'}), 400

        img_base64 = imagen_a_base64(img)

        return jsonify({
            'success': True,
            'filename': filename,
            'image': img_base64,
            'shape': img.shape
        })

@app.route('/process', methods=['POST'])
def process_image():
    """Procesar imagen según el tipo de operación"""
    data = request.get_json()

    if not data or 'filename' not in data or 'operation' not in data:
        return jsonify({'error': 'Datos incompletos'}), 400

    filename = data['filename']
    operation = data['operation']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if not os.path.exists(filepath):
        return jsonify({'error': 'Archivo no encontrado'}), 404

    img = cv2.imread(filepath)
    if img is None:
        return jsonify({'error': 'No se pudo cargar la imagen'}), 400

    try:
        result = {}

        if operation == 'histograma':
            if len(img.shape) == 3:
                hist_r, hist_g, hist_b, hist_gray = calcular_histograma(img)
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(hist_r, color='red', label='Rojo')
                ax.plot(hist_g, color='green', label='Verde')
                ax.plot(hist_b, color='blue', label='Azul')
                ax.plot(hist_gray, color='black', linestyle='--', label='Luminancia')
                ax.set_title('Histograma RGB')
                ax.set_xlabel('Intensidad de píxel')
                ax.set_ylabel('Frecuencia')
                ax.legend()
                result['histogram'] = figura_a_base64(fig)
            else:
                hist = calcular_histograma(img)
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(hist, color='black')
                ax.set_title('Histograma')
                ax.set_xlabel('Intensidad de píxel')
                ax.set_ylabel('Frecuencia')
                result['histogram'] = figura_a_base64(fig)

        elif operation == 'canales_rgb':
            canal_r, canal_g, canal_b = extraer_canales_rgb(img)
            result['canal_rojo'] = imagen_a_base64(canal_r)
            result['canal_verde'] = imagen_a_base64(canal_g)
            result['canal_azul'] = imagen_a_base64(canal_b)

        elif operation == 'roberts':
            img_roberts = aplicar_roberts(img)
            result['image'] = imagen_a_base64(img_roberts)

        elif operation == 'canny':
            low_thresh = data.get('low_threshold', 50)
            high_thresh = data.get('high_threshold', 150)
            img_canny = aplicar_canny(img, low_thresh, high_thresh)
            result['image'] = imagen_a_base64(img_canny)

        elif operation == 'multiumbral':
            classes = data.get('classes', 3)
            img_multi = aplicar_multiumbral(img, classes)
            result['image'] = imagen_a_base64(img_multi)

        elif operation == 'aritmeticas':
            escalar = data.get('escalar', 50)
            suma, resta, mult = operaciones_aritmeticas(img, escalar)
            result['suma'] = imagen_a_base64(suma)
            result['resta'] = imagen_a_base64(resta)
            result['multiplicacion'] = imagen_a_base64(mult)

        elif operation == 'logicas':
            filename2 = data.get('filename2')
            if not filename2:
                return jsonify({'error': 'Se necesita una segunda imagen para operaciones lógicas'}), 400

            filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
            img2 = cv2.imread(filepath2)
            and_img, or_img, xor_img = operaciones_logicas(img, img2)
            result['and'] = imagen_a_base64(and_img)
            result['or'] = imagen_a_base64(or_img)
            result['xor'] = imagen_a_base64(xor_img)

        elif operation == 'ruido_sal_pimienta':
            cantidad = data.get('cantidad', 0.05)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            img_ruido = agregar_sal_pimienta(img_gray, cantidad)
            result['image'] = imagen_a_base64(img_ruido)

        elif operation == 'ruido_gaussiano':
            sigma = data.get('sigma', 25)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            img_ruido = agregar_gaussiano(img_gray, 0, sigma)
            result['image'] = imagen_a_base64(img_ruido)

        elif operation == 'filtros':
            filtro_tipo = data.get('filtro_tipo', 'gaussiano')
            img_filtrada = aplicar_filtros(img, filtro_tipo)
            result['image'] = imagen_a_base64(img_filtrada)

        elif operation == 'ecualizacion':
            img_eq = ecualizacion_log_hiperbolica(img)
            result['image'] = imagen_a_base64(img_eq)

        else:
            return jsonify({'error': 'Operación no válida'}), 400

        return jsonify({
            'success': True,
            'result': result
        })

    except Exception as e:
        return jsonify({'error': f'Error procesando imagen: {str(e)}'}), 500

# -------------------------------------------------------------------
# 5. Ejecución local (dev) y en producción (Render con gunicorn)
# -------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
