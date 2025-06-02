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
import warnings
warnings.filterwarnings('ignore')

# Configurar Flask
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

# Configuración para optimizar memoria
plt.rcParams['figure.max_open_warning'] = 0
matplotlib.rcParams['font.size'] = 8

# -------------------------------------------------------------------
# FUNCIONES DE DETECCIÓN DE VEHÍCULOS (desde m5_ecualizacion.py)
# -------------------------------------------------------------------

def roberts_filter(image):
    """Implementación del filtro Roberts para detección de bordes"""
    roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    
    grad_x = cv2.filter2D(image, cv2.CV_32F, roberts_x)
    grad_y = cv2.filter2D(image, cv2.CV_32F, roberts_y)
    
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    
    return magnitude

def ecualizacion_log_hiperbolica_avanzada(img):
    """Ecualización log-hiperbólica optimizada"""
    r_min = np.min(img)
    r_max = np.max(img)
    
    if r_max == r_min:
        return img.copy()
    
    hist, bins = np.histogram(img.flatten(), bins=256, range=[0,256], density=True)
    cdf = hist.cumsum()
    
    cdf_normalizada = cdf * ((r_max - r_min) / (r_max - r_min))
    transformacion = np.floor(r_min + (r_max - r_min) * cdf_normalizada).astype(np.uint8)
    
    img_ecualizada = transformacion[img]
    
    # Limpiar memoria
    del hist, cdf, cdf_normalizada, transformacion
    gc.collect()
    
    return img_ecualizada

def multi_otsu_segmentation(image, n_classes=4):
    """Segmentación usando Multi-Otsu thresholding"""
    thresholds = threshold_multiotsu(image, classes=n_classes)
    segmented = np.digitize(image, bins=thresholds)
    return segmented, thresholds

def segment_cars_advanced_processing(image_path):
    """Versión optimizada de detección de vehículos"""
    
    # 1. Cargar y procesar imagen inicial
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("No se pudo cargar la imagen")
    
    # Redimensionar si es muy grande para ahorrar memoria
    height, width = img.shape[:2]
    if width > 1024 or height > 1024:
        scale = min(1024/width, 1024/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height))
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Procesamiento paso a paso con limpieza de memoria
    img_ecualizada = ecualizacion_log_hiperbolica_avanzada(original_gray)
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_contrast = clahe.apply(img_ecualizada)
    
    # Corrección gamma
    def adjust_gamma(image, gamma=1.1):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    gamma_corrected = adjust_gamma(enhanced_contrast, gamma=1.1)
    
    # 3. Blur (solo uno para ahorrar memoria)
    blur_img = cv2.GaussianBlur(gamma_corrected, (7, 7), 1.5)
    
    # 4. Filtro Roberts
    roberts_edges = roberts_filter(blur_img)
    
    # 5. Multi-Otsu con menos clases
    segmented_img, thresholds = multi_otsu_segmentation(blur_img, n_classes=3)
    
    # 6. Crear máscara principal
    if len(thresholds) > 1:
        vehicle_mask = ((segmented_img == 1) | (segmented_img == 2)).astype(np.uint8) * 255
    else:
        vehicle_mask = segmented_img.astype(np.uint8)
    
    # 7. Combinar con Roberts
    combined_roberts = cv2.bitwise_and(vehicle_mask, roberts_edges)
    
    # 8. Operaciones morfológicas simplificadas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morphology_result = cv2.morphologyEx(combined_roberts, cv2.MORPH_CLOSE, kernel, iterations=2)
    morphology_result = cv2.dilate(morphology_result, kernel, iterations=1)
    
    # 9. Encontrar contornos
    contours, _ = cv2.findContours(morphology_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 10. Filtrado de contornos optimizado
    car_contours = []
    car_info = []
    
    img_area = img.shape[0] * img.shape[1]
    min_area = img_area * 0.002  # Área mínima más grande
    max_area = img_area * 0.08   # Área máxima más pequeña
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Filtros básicos más estrictos
            if (1.2 < aspect_ratio < 3.5 and
                y > img.shape[0] * 0.4 and  # Solo parte inferior
                w > 20 and h > 15):  # Tamaño mínimo
                
                moments = cv2.moments(contour)
                if moments['m00'] != 0:
                    cx = int(moments['m10'] / moments['m00'])
                    cy = int(moments['m01'] / moments['m00'])
                else:
                    cx, cy = x + w//2, y + h//2
                
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                extent = area / (w * h) if (w * h) > 0 else 0
                
                if solidity > 0.3 and extent > 0.2:
                    car_contours.append(contour)
                    car_info.append({
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'solidity': solidity,
                        'extent': extent,
                        'center': (cx, cy),
                        'bbox': (x, y, w, h)
                    })
    
    # 11. Eliminar duplicados simplificado
    final_contours = []
    final_info = []
    
    for i, (contour, info) in enumerate(zip(car_contours, car_info)):
        is_duplicate = False
        x1, y1, w1, h1 = info['bbox']
        
        for existing_info in final_info:
            x2, y2, w2, h2 = existing_info['bbox']
            
            # Verificar solapamiento simple
            if (abs(x1 - x2) < max(w1, w2) * 0.5 and 
                abs(y1 - y2) < max(h1, h2) * 0.5):
                if info['area'] <= existing_info['area']:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            final_contours.append(contour)
            final_info.append(info)
    
    # 12. Crear imagen resultado
    result_img = img_rgb.copy()
    
    for i, (contour, info) in enumerate(zip(final_contours, final_info)):
        cv2.drawContours(result_img, [contour], -1, (0, 255, 0), 2)
        
        x, y, w, h = info['bbox']
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        label = f'Auto {i+1}'
        cv2.putText(result_img, label, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cx, cy = info['center']
        cv2.circle(result_img, (cx, cy), 3, (0, 0, 255), -1)
    
    # Limpiar memoria antes de retornar
    del img, vehicle_mask, combined_roberts, contours
    gc.collect()
    return {
        'result_image': result_img,
        'car_contours': final_contours,
        'car_info': final_info,
        'processing_steps': {
            'original': original_gray,
            'ecualizacion_log': img_ecualizada,
            'enhanced_contrast': enhanced_contrast,
            'gamma_corrected': gamma_corrected,
            'roberts_edges': roberts_edges,
            'segmented': segmented_img,
            'morphology_result': morphology_result
        },
        'thresholds': thresholds
    }
    
    for i, (contour, info) in enumerate(zip(car_contours, car_info)):
        is_duplicate = False
        x1, y1, w1, h1 = info['bbox']
        
        for j, existing_info in enumerate(final_info):
            x2, y2, w2, h2 = existing_info['bbox']
            
            x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            intersection = x_overlap * y_overlap
            union = w1 * h1 + w2 * h2 - intersection
            
            if union > 0:
                iou = intersection / union
                if iou > 0.3:
                    if info['area'] <= existing_info['area']:
                        is_duplicate = True
                        break
                    else:
                        final_contours[j] = contour
                        final_info[j] = info
                        is_duplicate = True
                        break
        
        if not is_duplicate:
            final_contours.append(contour)
            final_info.append(info)
    
    # 12. Crear imagen resultado
    result_img = img_rgb.copy()
    
    for i, (contour, info) in enumerate(zip(final_contours, final_info)):
        cv2.drawContours(result_img, [contour], -1, (0, 255, 0), 2)
        
        x, y, w, h = info['bbox']
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        label = f'Auto {i+1}'
        cv2.putText(result_img, label, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cx, cy = info['center']
        cv2.circle(result_img, (cx, cy), 3, (0, 0, 255), -1)
    
    return {
        'result_image': result_img,
        'car_contours': final_contours,
        'car_info': final_info,
        'processing_steps': {
            'original': original_gray,
            'ecualizacion_log': img_ecualizada,
            'enhanced_contrast': enhanced_contrast,
            'gamma_corrected': gamma_corrected,
            'blur_light': blur_light,
            'blur_medium': blur_medium,
            'blur_heavy': blur_heavy,
            'roberts_edges': roberts_edges,
            'segmented': segmented_img,
            'masks': masks,
            'combined_roberts': combined_roberts,
            'morphology_result': dilated
        },
        'thresholds': thresholds
    }

def generar_visualizacion_completa(results):
    """Versión optimizada de visualización - más pequeña y eficiente"""
    steps = results['processing_steps']
    
    # Crear figura más pequeña
    fig = plt.figure(figsize=(12, 8), dpi=80)  # Reducido de 20x16 a 12x8
    
    # Solo mostrar los pasos más importantes
    plots_config = [
        ('original', 'Original', 'gray'),
        ('ecualizacion_log', 'Equalizado', 'gray'),
        ('roberts_edges', 'Bordes Roberts', 'gray'),
        ('segmented', 'Segmentación', 'tab10'),
        ('morphology_result', 'Morfología', 'gray'),
    ]
    
    # 2x3 grid en lugar de 4x4
    for i, (step_key, title, cmap) in enumerate(plots_config):
        if step_key in steps and i < 5:
            plt.subplot(2, 3, i + 1)
            plt.title(title, fontsize=9)
            
            # Redimensionar imagen si es muy grande
            img = steps[step_key]
            if img.shape[0] > 400 or img.shape[1] > 400:
                scale = min(400/img.shape[1], 400/img.shape[0])
                new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
                if len(img.shape) == 2:
                    img = cv2.resize(img, new_size)
                else:
                    img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
            
            plt.imshow(img, cmap=cmap)
            plt.axis('off')
    
    # Resultado final
    plt.subplot(2, 3, 6)
    plt.title('Resultado Final', fontsize=9)
    result_img = results['result_image']
    if result_img.shape[0] > 400 or result_img.shape[1] > 400:
        scale = min(400/result_img.shape[1], 400/result_img.shape[0])
        new_size = (int(result_img.shape[1] * scale), int(result_img.shape[0] * scale))
        result_img = cv2.resize(result_img, new_size, interpolation=cv2.INTER_AREA)
    
    plt.imshow(result_img)
    plt.axis('off')
    
    plt.tight_layout()
    return fig

# -------------------------------------------------------------------
# FUNCIONES AUXILIARES EXISTENTES
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
    """Versión optimizada con compresión"""
    if len(img.shape) == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img
    
    # Reducir calidad para ahorrar memoria
    pil_img = Image.fromarray(img_rgb)
    
    # Redimensionar si es muy grande
    if pil_img.size[0] > 800 or pil_img.size[1] > 800:
        pil_img.thumbnail((800, 800), Image.Resampling.LANCZOS)
    
    img_buffer = io.BytesIO()
    pil_img.save(img_buffer, format='PNG', optimize=True)
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    
    # Limpiar
    img_buffer.close()
    del pil_img, img_buffer
    gc.collect()
    
    return f"data:image/png;base64,{img_base64}"

def figura_a_base64(fig):
    """Versión optimizada para figuras"""
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', bbox_inches='tight', 
                dpi=80, optimize=True)  
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    
    # Limpiar memoria
    img_buffer.close()
    plt.close(fig)
    del fig
    gc.collect()
    
    return f"data:image/png;base64,{img_base64}"

# -------------------------------------------------------------------
# RUTAS DE LA API
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
    """Versión optimizada del procesamiento"""
    data = request.get_json()

    if not data or 'filename' not in data or 'operation' not in data:
        return jsonify({'error': 'Datos incompletos'}), 400

    filename = data['filename']
    operation = data['operation']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if not os.path.exists(filepath):
        return jsonify({'error': 'Archivo no encontrado'}), 404

    try:
        result = {}

        if operation == 'deteccion_vehiculos':
            # Aplicar detección optimizada
            vehicle_results = segment_cars_optimized(filepath)
            
            # Generar visualización optimizada
            fig_completa = generar_visualizacion_optimizada(vehicle_results)
            
            # Convertir a base64
            result['imagen_final'] = imagen_a_base64(vehicle_results['result_image'])
            result['visualizacion_completa'] = figura_a_base64(fig_completa)
            
            # Información básica
            result['num_vehiculos'] = len(vehicle_results['car_info'])
            result['umbrales_otsu'] = [float(t) for t in vehicle_results['thresholds']]
            
            # Solo información esencial de vehículos
            vehiculos_info = []
            for i, info in enumerate(vehicle_results['car_info']):
                vehiculos_info.append({
                    'id': i + 1,
                    'area': int(info['area']),
                    'center': info['center'],
                    'bbox': info['bbox']
                })
            result['vehiculos'] = vehiculos_info
            
            # Solo pasos esenciales
            steps = vehicle_results['processing_steps']
            result['pasos_procesamiento'] = {
                'original': imagen_a_base64(steps['original']),
                'roberts_edges': imagen_a_base64(steps['roberts_edges']),
                'morphology_result': imagen_a_base64(steps['morphology_result'])
            }
            
            # Limpiar memoria
            del vehicle_results, steps
            gc.collect()
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
