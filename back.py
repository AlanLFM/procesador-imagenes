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
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# -------------------------------------------------------------------
# Configurar Flask para servir archivos estáticos
# -------------------------------------------------------------------
app = Flask(
    __name__,
    static_folder='front',
    static_url_path='/static'
)
CORS(app)

# Configuración de carpetas
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# -------------------------------------------------------------------
# NUEVA FUNCIÓN: Segmentación de autos (adaptada de Ideal.py)
# -------------------------------------------------------------------
def segment_cars_classical(img):
    """
    Segmentación clásica de autos usando OpenCV y procesamiento de contornos.
    Usa vecindad 4 y 8 según sea más apropiado para cada operación morfológica.
    
    Args:
        img: imagen BGR de OpenCV
    
    Returns:
        dict con las imágenes de cada etapa y el resultado final
    """
    # 1. Convertir a RGB para visualización
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 2. Preprocesamiento
    # 2.1. Convertir a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2.2. Aplicar filtro Gaussiano para reducir ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Detección de bordes con Canny
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
    # 4. Operaciones morfológicas con vecindad específica
    # Kernel de vecindad 4 (cruz) - mejor para operaciones de cierre conservadoras
    kernel_4 = np.array([[0, 1, 0],
                         [1, 1, 1],
                         [0, 1, 0]], dtype=np.uint8)
    
    # Kernel de vecindad 8 (cuadrado 3x3) - mejor para dilatación más agresiva
    kernel_8 = np.ones((3, 3), dtype=np.uint8)
    
    # 4.1. Usar vecindad 4 para cierre morfológico (más conservador)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_4)
    
    # 4.2. Usar vecindad 8 para dilatación (más expansivo)
    dilated = cv2.dilate(closed, kernel_8, iterations=2)
    
    # 5. Encontrar contornos en la etapa dilatada
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 6. Filtrar contornos por características de autos
    car_contours = []
    min_area = 1000      # Área mínima
    max_area = 50000     # Área máxima
    min_aspect_ratio = 1.2  # Relación de aspecto mínima (ancho/alto)
    max_aspect_ratio = 4.0  # Relación de aspecto máxima
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / float(h)
            if min_aspect_ratio < aspect_ratio < max_aspect_ratio:
                # Calcular solidez (área del contorno / área del rectángulo envolvente)
                solidity = area / float(w * h)
                # Los autos tienden a tener una solidez moderada
                if 0.3 < solidity < 0.8:
                    car_contours.append(contour)
    
    # 7. Visualización de resultados: dibujar contornos y bounding boxes
    result_img = img_rgb.copy()
    for i, contour in enumerate(car_contours):
        # Dibujar contorno en verde
        cv2.drawContours(result_img, [contour], -1, (0, 255, 0), 2)
        # Dibujar bounding box en rojo
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Etiquetar
        cv2.putText(result_img, f'Auto {i+1}', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    return {
        'original': img_rgb,
        'gray': gray,
        'blurred': blurred,
        'edges': edges,
        'closed': closed,
        'dilated': dilated,
        'result': result_img,
        'car_count': len(car_contours),
        'contours': car_contours
    }

def create_segmentation_process_visualization(segmentation_data):
    """
    Crear una visualización completa del proceso de segmentación
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    
    # 1. Imagen original en RGB
    axes[0].imshow(segmentation_data['original'])
    axes[0].set_title("Original (RGB)")
    axes[0].axis("off")
    
    # 2. Escala de grises
    axes[1].imshow(segmentation_data['gray'], cmap="gray")
    axes[1].set_title("Escala de grises")
    axes[1].axis("off")
    
    # 3. Imagen desenfocada (GaussianBlur)
    axes[2].imshow(segmentation_data['blurred'], cmap="gray")
    axes[2].set_title("GaussianBlur")
    axes[2].axis("off")
    
    # 4. Bordes (Canny)
    axes[3].imshow(segmentation_data['edges'], cmap="gray")
    axes[3].set_title("Canny (bordes)")
    axes[3].axis("off")
    
    # 5. Cerrado morfológico con vecindad 4
    axes[4].imshow(segmentation_data['closed'], cmap="gray")
    axes[4].set_title("Cierre (Vecindad 4)")
    axes[4].axis("off")
    
    # 6. Dilatación con vecindad 8
    axes[5].imshow(segmentation_data['dilated'], cmap="gray")
    axes[5].set_title("Dilatación (Vecindad 8)")
    axes[5].axis("off")
    
    # 7. Resultado final con contornos y bounding boxes
    axes[6].imshow(segmentation_data['result'])
    axes[6].set_title(f"Resultado final\n({segmentation_data['car_count']} autos detectados)")
    axes[6].axis("off")
    
    # 8. Kernels utilizados
    kernel_4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    kernel_8 = np.ones((3, 3), dtype=np.uint8)
    
    axes[7].text(0.1, 0.8, "Kernels utilizados:", fontsize=12, weight='bold', transform=axes[7].transAxes)
    axes[7].text(0.1, 0.6, "Vecindad 4 (Cierre):\n[[0,1,0]\n [1,1,1]\n [0,1,0]]", 
                fontsize=10, transform=axes[7].transAxes, family='monospace')
    axes[7].text(0.1, 0.2, "Vecindad 8 (Dilatación):\n[[1,1,1]\n [1,1,1]\n [1,1,1]]", 
                fontsize=10, transform=axes[7].transAxes, family='monospace')
    axes[7].axis("off")
    
    plt.tight_layout()
    return fig

# -------------------------------------------------------------------
# Funciones auxiliares existentes (manteniendo las mismas)
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
# Rutas de la aplicación
# -------------------------------------------------------------------
@app.route('/', methods=['GET'])
def index():
    return send_from_directory('front', 'main.html')

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

        # NUEVA OPERACIÓN: Segmentación de autos
        if operation == 'deteccion_vehiculos':
            segmentation_data = segment_cars_classical(img)
            
            # Crear visualización del proceso completo
            process_fig = create_segmentation_process_visualization(segmentation_data)
            
            result['result'] = imagen_a_base64(segmentation_data['result'])
            result['process_visualization'] = figura_a_base64(process_fig)
            result['car_count'] = segmentation_data['car_count']
            result['gray'] = imagen_a_base64(segmentation_data['gray'])
            result['blurred'] = imagen_a_base64(segmentation_data['blurred'])
            result['edges'] = imagen_a_base64(segmentation_data['edges'])
            result['closed'] = imagen_a_base64(segmentation_data['closed'])
            result['dilated'] = imagen_a_base64(segmentation_data['dilated'])

        elif operation == 'histograma':
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
