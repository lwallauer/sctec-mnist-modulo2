import sys
import os
import time
import cv2
import numpy as np
import tensorflow as tf
import urllib.request
import math

# ============================================================
# MEDIA PIPE — COMPATIBILIDADE LEGACY + API TASKS ATUAL
# ============================================================

MEDIAPIPE_DISPONIVEL = False
MEDIAPIPE_MODO = None

mp = None
mp_hands_module = None
mp_draw_module = None
mp_vision = None
mp_python = None
hand_landmarker = None

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEDIAPIPE_MODEL_DIR = os.path.join(BASE_DIR, "models", "mediapipe")
HAND_MODEL_PATH = os.path.join(MEDIAPIPE_MODEL_DIR, "hand_landmarker.task")

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

def baixar_modelo_hand_landmarker():
    os.makedirs(MEDIAPIPE_MODEL_DIR, exist_ok=True)
    if os.path.exists(HAND_MODEL_PATH):
        return True
    print("⬇️ Modelo MediaPipe Hand Landmarker não encontrado. Baixando...")
    try:
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
        print(f"✅ Modelo baixado: {HAND_MODEL_PATH}")
        return True
    except Exception as e:
        print(f"❌ Erro ao baixar: {e}")
        try:
            if os.path.exists(HAND_MODEL_PATH):
                os.remove(HAND_MODEL_PATH)
        except Exception:
            pass
        return False

# Tentativa 1: API Legacy
try:
    import mediapipe as mp
    if hasattr(mp, "solutions"):
        mp_hands_module = mp.solutions.hands
        mp_draw_module = mp.solutions.drawing_utils
        MEDIAPIPE_DISPONIVEL = True
        MEDIAPIPE_MODO = "legacy"
    elif not MEDIAPIPE_DISPONIVEL:
        try:
            from mediapipe.python.solutions import hands as _hands
            from mediapipe.python.solutions import drawing_utils as _drawing
            mp_hands_module = _hands
            mp_draw_module = _drawing
            MEDIAPIPE_DISPONIVEL = True
            MEDIAPIPE_MODO = "legacy"
        except Exception:
            pass
except Exception as e:
    print(f"⚠️ Falha ao importar MediaPipe Legacy: {e}")

# Tentativa 2: API Tasks atual
if not MEDIAPIPE_DISPONIVEL:
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        if baixar_modelo_hand_landmarker():
            MEDIAPIPE_DISPONIVEL = True
            MEDIAPIPE_MODO = "tasks"
    except Exception as e:
        print(f"⚠️ MediaPipe Tasks indisponível: {e}")

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QStackedWidget,
    QFileDialog, QProgressBar, QFrame, QGraphicsDropShadowEffect
)
from PySide6.QtCore import Qt, QPoint, QTimer
from PySide6.QtGui import (
    QPainter, QPen, QPixmap, QImage, QShortcut, QKeySequence, QColor
)

# ============================================================
# 1. Widget Customizado para Desenho (Mouse)
# ============================================================
class DrawingCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(380, 380)
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.black)
        self.drawing = False
        self.last_point = QPoint()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(self.rect(), self.image, self.image.rect())

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.position().toPoint()

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) and self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(Qt.white, 22, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            current_point = event.position().toPoint()
            painter.drawLine(self.last_point, current_point)
            self.last_point = current_point
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def clear_canvas(self):
        self.image.fill(Qt.black)
        self.update()

    def get_numpy_array(self):
        img_rgb = self.image.convertToFormat(QImage.Format_RGB888)
        width = img_rgb.width()
        height = img_rgb.height()
        ptr = img_rgb.bits()
        arr = np.array(ptr).reshape(height, width, 3)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        return gray

# ============================================================
# 2. Janela Principal (UX/UI Sênior)
# ============================================================
class MNISTApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("SCTEC - Classificador Neural MNIST (Mini Projeto Módulo 2)")
        self.setFixedSize(1250, 780)

        self.current_image_path = None
        self.active_mode = "mouse"

        # ----------------------------------------------------
        # CARREGAMENTO DO MODELO MNIST
        # ----------------------------------------------------
        caminhos_possiveis = [
            "models/mnist_model.keras",
            "../models/mnist_model.keras",
            os.path.join(os.path.dirname(__file__), "models", "mnist_model.keras"),
            "models/mnist_mlp.keras",
            os.path.join(os.path.dirname(__file__), "models", "mnist_mlp.keras"),
        ]

        self.model = None

        for caminho in caminhos_possiveis:
            if os.path.exists(caminho):
                try:
                    self.model = tf.keras.models.load_model(caminho)
                    print(f"✅ Modelo MNIST carregado com sucesso de: {caminho}")
                    break
                except Exception as e:
                    print(f"⚠️ Não foi possível carregar o modelo {caminho}: {e}")

        # ----------------------------------------------------
        # INICIALIZAÇÃO DO MEDIAPIPE
        # ----------------------------------------------------
        self.hands = None
        self.mp_draw = None
        self.mp_hands = None

        if MEDIAPIPE_DISPONIVEL and MEDIAPIPE_MODO == "legacy":
            try:
                self.mp_hands = mp_hands_module
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False, max_num_hands=1,
                    model_complexity=1, min_detection_confidence=0.65, min_tracking_confidence=0.65
                )
                self.mp_draw = mp_draw_module
            except Exception:
                self.hands = None

        elif MEDIAPIPE_DISPONIVEL and MEDIAPIPE_MODO == "tasks":
            try:
                base_options = mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
                options = mp_vision.HandLandmarkerOptions(
                    base_options=base_options, running_mode=mp_vision.RunningMode.IMAGE,
                    num_hands=1, min_hand_detection_confidence=0.65, min_hand_presence_confidence=0.65, min_tracking_confidence=0.65
                )
                self.hands = mp_vision.HandLandmarker.create_from_options(options)
            except Exception:
                self.hands = None

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_webcam)
        self.webcam_mask = np.zeros((480, 640), dtype=np.uint8)

        # --- Estado do desenho por webcam (melhorado) ---
        self.prev_x = 0
        self.prev_y = 0
        self.smooth_x = 0.0
        self.smooth_y = 0.0
        self.alpha = 0.35              # EMA: 0.2 = mais suave | 0.5 = mais responsivo
        self.lost_frames = 0
        self.max_lost = 6              # ~180 ms a 30 fps → quebra o traço
        self.last_move_time = 0.0
        self.move_threshold = 8        # pixels mínimos para considerar movimento
        self.pause_ms = 180            # ms parado = não desenha

        self.init_ui()

    # ========================================================
    # INTERFACE & LAYOUT UX
    # ========================================================
    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(25, 25, 25, 25)
        main_layout.setSpacing(25)

        # --- Atalhos Globais de Teclado ---
        QShortcut(QKeySequence(Qt.Key_Escape), self).activated.connect(self.clear_inputs)
        QShortcut(QKeySequence(Qt.Key_Space), self).activated.connect(self.run_prediction)

        # ----------------------------------------------------
        # PAINEL ESQUERDO (Controles & Telemetria)
        # ----------------------------------------------------
        left_panel = QFrame()
        left_panel.setObjectName("panel")
        left_panel.setFixedWidth(280)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(20, 20, 20, 20)

        titulo_entrada = QLabel("⚙️ Fonte de Entrada")
        titulo_entrada.setObjectName("header")
        left_layout.addWidget(titulo_entrada)

        self.btn_mouse = QPushButton("🖱️ Desenhar (Mouse)")
        self.btn_upload = QPushButton("📁 Carregar Imagem")
        self.btn_webcam = QPushButton("📷 Rastreamento (Webcam)")
        left_layout.addWidget(self.btn_mouse)
        left_layout.addWidget(self.btn_upload)
        left_layout.addWidget(self.btn_webcam)

        left_layout.addSpacing(15)

        # --- Painel de Telemetria da Rede Neural ---
        telemetry_box = QFrame()
        telemetry_box.setStyleSheet("background-color: #0B132B; border-radius: 8px; border: 1px solid #3A506B;")
        telemetry_layout = QVBoxLayout(telemetry_box)

        telemetry_title = QLabel("🟢 Engine Online")
        telemetry_title.setStyleSheet("color: #0ABAB5; font-weight: bold; font-size: 13px; border: none;")

        telemetry_text = QLabel(
            "<b>Topologia:</b> MLP (Dense Neural Net)<br>"
            "<b>Input Shape:</b> Tensor(1, 784)<br>"
            "<b>Filtro CV2:</b> Otsu Thresholding<br>"
            "<b>Ativação Saída:</b> Softmax (10 classes)<br>"
            "<b>Resolução Padrão:</b> 28x28 Grayscale"
        )
        telemetry_text.setStyleSheet("color: #94a3b8; font-size: 12px; border: none; line-height: 1.5; margin-top: 5px;")

        telemetry_layout.addWidget(telemetry_title)
        telemetry_layout.addWidget(telemetry_text)
        left_layout.addWidget(telemetry_box)

        left_layout.addStretch()

        # --- Logo Python em Neon ---
        self.lbl_python_img = QLabel()
        python_logo_path = os.path.join(BASE_DIR, "data", "python.png")

        if os.path.exists(python_logo_path):
            pixmap = QPixmap(python_logo_path).scaled(60, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.lbl_python_img.setPixmap(pixmap)
        else:
            self.lbl_python_img.setText("🐍")
            self.lbl_python_img.setStyleSheet("font-size: 50px;")

        self.lbl_python_img.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.lbl_python_img)

        self.lbl_python_text = QLabel("Python IA Engine\nBy Informático Floripa")
        self.lbl_python_text.setAlignment(Qt.AlignCenter)
        self.lbl_python_text.setStyleSheet("font-size: 14px; font-weight: bold; color: #94a3b8;")
        left_layout.addWidget(self.lbl_python_text)

        self.neon_effect = QGraphicsDropShadowEffect()
        self.neon_effect.setOffset(0, 0)
        self.lbl_python_img.setGraphicsEffect(self.neon_effect)

        self.color_step = 0
        self.neon_timer = QTimer(self)
        self.neon_timer.timeout.connect(self.animate_neon)
        self.neon_timer.start(50)

        left_layout.addSpacing(15)

        # Botões de Ação Final
        self.btn_clear = QPushButton("🗑️ Limpar Entrada (ESC)")
        self.btn_clear.setObjectName("btnRojo")
        self.aplicar_sombra(self.btn_clear, "#E63946", raio=10)
        left_layout.addWidget(self.btn_clear)

        self.btn_predict = QPushButton("🚀 EXECUTAR PREDIÇÃO\n(ESPAÇO)")
        self.btn_predict.setObjectName("btnTiffany")
        self.aplicar_sombra(self.btn_predict, "#0ABAB5", raio=20)
        left_layout.addWidget(self.btn_predict)

        # ----------------------------------------------------
        # PAINEL CENTRAL (Monitor)
        # ----------------------------------------------------
        center_panel = QFrame()
        center_panel.setObjectName("panel")
        self.aplicar_sombra(center_panel, "#000000", raio=25, offset=5)

        center_layout = QVBoxLayout(center_panel)
        center_layout.setAlignment(Qt.AlignCenter)

        self.stack = QStackedWidget()

        # 1. Canvas
        self.canvas_page = QWidget()
        canvas_layout = QVBoxLayout(self.canvas_page)
        canvas_layout.setAlignment(Qt.AlignCenter)
        self.canvas = DrawingCanvas()
        canvas_layout.addWidget(self.canvas, alignment=Qt.AlignCenter)
        self.stack.addWidget(self.canvas_page)

        # 2. Upload
        self.upload_page = QWidget()
        upload_layout = QVBoxLayout(self.upload_page)
        upload_layout.setAlignment(Qt.AlignCenter)
        self.lbl_image = QLabel("Clique no botão lateral para\ncarregar uma imagem.")
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setFixedSize(380, 380)
        self.lbl_image.setStyleSheet("border: 2px dashed #475569; background-color: #000; color: #94a3b8; border-radius: 8px;")
        upload_layout.addWidget(self.lbl_image, alignment=Qt.AlignCenter)
        self.stack.addWidget(self.upload_page)

        # 3. Webcam
        self.webcam_page = QWidget()
        webcam_layout = QVBoxLayout(self.webcam_page)
        webcam_layout.setAlignment(Qt.AlignCenter)

        if MEDIAPIPE_DISPONIVEL:
            modo = "MediaPipe Legacy" if MEDIAPIPE_MODO == "legacy" else "MediaPipe Tasks"
            texto_webcam = f"Câmera pronta\n\nRastreamento: {modo}\n\nAponte sua mão para a câmera."
        else:
            texto_webcam = "MediaPipe não disponível.\n\nModo alternativo OpenCV ativo."

        self.lbl_webcam = QLabel(texto_webcam)
        self.lbl_webcam.setAlignment(Qt.AlignCenter)
        self.lbl_webcam.setFixedSize(540, 400)
        self.lbl_webcam.setWordWrap(True)
        self.lbl_webcam.setStyleSheet("background-color: #000; border-radius: 8px; color: #94a3b8; font-size: 15px; padding: 20px;")
        webcam_layout.addWidget(self.lbl_webcam, alignment=Qt.AlignCenter)
        self.stack.addWidget(self.webcam_page)

        center_layout.addWidget(self.stack)

        # ----------------------------------------------------
        # PAINEL DIREITO (Resultados & Analytics)
        # ----------------------------------------------------
        right_panel = QFrame()
        right_panel.setObjectName("panel")
        right_panel.setFixedWidth(320)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(20, 20, 20, 20)

        titulo_resultado = QLabel("🎯 Resultado da Inferência")
        titulo_resultado.setObjectName("header")
        right_layout.addWidget(titulo_resultado, alignment=Qt.AlignTop)

        self.lbl_resultado = QLabel("-")
        self.lbl_resultado.setObjectName("resultadoBig")
        self.lbl_resultado.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.lbl_resultado)

        self.lbl_confianca = QLabel("Confiança: 0%")
        self.lbl_confianca.setStyleSheet("font-size: 14px; font-weight: bold; color: #0ABAB5;")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(14)
        right_layout.addWidget(self.lbl_confianca)
        right_layout.addWidget(self.progress_bar)

        self.lbl_pipeline = QLabel("👁️ Matriz Pós-Processada (28x28):")
        self.lbl_pipeline.setStyleSheet("color: #94a3b8; margin-top: 15px; font-weight: bold;")
        right_layout.addWidget(self.lbl_pipeline)

        self.lbl_mini = QLabel()
        self.lbl_mini.setFixedSize(140, 140)
        self.lbl_mini.setStyleSheet("border: 2px solid #334155; background-color: black; border-radius: 6px;")
        right_layout.addWidget(self.lbl_mini, alignment=Qt.AlignCenter)

        self.lbl_explicacao = QLabel(
            "<b>💡 Entendendo o Pipeline:</b><br><br>"
            "<b>1. Captura:</b> O traço é isolado utilizando caixas contornadoras (Bounding Box).<br><br>"
            "<b>2. Visão da IA:</b> A imagem é achatada para 784 pixels em escala de cinza e enviada à matriz.<br><br>"
            "<b>3. Output (Softmax):</b> A confiança representa o quão 'certeza' a rede tem estatisticamente em relação aos padrões treinados."
        )
        self.lbl_explicacao.setWordWrap(True)
        self.lbl_explicacao.setStyleSheet("background-color: #1D3557; color: #F1FAEE; padding: 15px; border-radius: 8px; font-size: 12px; margin-top: 15px; line-height: 1.4;")
        right_layout.addWidget(self.lbl_explicacao)

        right_layout.addStretch()

        main_layout.addWidget(left_panel)
        main_layout.addWidget(center_panel)
        main_layout.addWidget(right_panel)

        # ----------------------------------------------------
        # CONEXÕES
        # ----------------------------------------------------
        self.btn_mouse.clicked.connect(lambda: self.switch_mode("mouse"))
        self.btn_upload.clicked.connect(lambda: self.switch_mode("upload"))
        self.btn_webcam.clicked.connect(lambda: self.switch_mode("webcam"))
        self.btn_clear.clicked.connect(self.clear_inputs)
        self.btn_predict.clicked.connect(self.run_prediction)

        self.apply_styles()

    # ========================================================
    # UX HELPERS E ANIMAÇÃO NEON (INTERPOLAÇÃO)
    # ========================================================
    def aplicar_sombra(self, widget, hex_color, raio, offset=0):
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(raio)
        shadow.setColor(QColor(hex_color))
        shadow.setOffset(offset, offset)
        widget.setGraphicsEffect(shadow)

    def animate_neon(self):
        self.color_step += 0.1
        wave = (math.sin(self.color_step) + 1) / 2.0

        r = int(55 + (255 - 55) * wave)
        g = int(118 + (212 - 118) * wave)
        b = int(171 + (59 - 171) * wave)

        color = QColor(r, g, b)
        raio = int(15 + 8 * math.sin(self.color_step * 1.5))

        self.neon_effect.setColor(color)
        self.neon_effect.setBlurRadius(abs(raio))

    # ========================================================
    # TROCA DE MODO
    # ========================================================
    def switch_mode(self, mode):
        self.active_mode = mode
        if mode == "mouse":
            self.stack.setCurrentIndex(0)
            self.stop_webcam()
        elif mode == "upload":
            self.stack.setCurrentIndex(1)
            self.stop_webcam()
            file_name, _ = QFileDialog.getOpenFileName(self, "Abrir Imagem", "", "Imagens (*.png *.jpg *.jpeg)")
            if file_name:
                self.current_image_path = file_name
                self.lbl_image.setPixmap(QPixmap(file_name).scaled(380, 380, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        elif mode == "webcam":
            self.stack.setCurrentIndex(2)
            self.start_webcam()

    # ========================================================
    # RESET DO ESTADO DA WEBCAM
    # ========================================================
    def _reset_webcam_drawing_state(self):
        self.prev_x = 0
        self.prev_y = 0
        self.smooth_x = 0.0
        self.smooth_y = 0.0
        self.lost_frames = 0
        self.last_move_time = 0.0

    # ========================================================
    # LIMPAR
    # ========================================================
    def clear_inputs(self):
        if self.active_mode == "mouse":
            self.canvas.clear_canvas()
        elif self.active_mode == "webcam":
            self.webcam_mask.fill(0)
            self._reset_webcam_drawing_state()

        self.lbl_resultado.setText("-")
        self.progress_bar.setValue(0)
        self.lbl_confianca.setText("Confiança: 0%")
        self.lbl_mini.clear()

    # ========================================================
    # WEBCAM
    # ========================================================
    def start_webcam(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.cap.release()
                self.cap = None
                self.lbl_webcam.setText("❌ Não foi possível abrir a câmera.\n\nVerifique se outro aplicativo está usando a webcam.")
                return
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.webcam_mask = np.zeros((480, 640), dtype=np.uint8)
            self._reset_webcam_drawing_state()
        self.timer.start(30)

    def stop_webcam(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._reset_webcam_drawing_state()

    # ========================================================
    # DESENHO DOS LANDMARKS
    # ========================================================
    def draw_hand_landmarks(self, frame, landmarks):
        h, w, _ = frame.shape
        points = []
        for lm in landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            points.append((x, y))
        for a, b in HAND_CONNECTIONS:
            if a < len(points) and b < len(points):
                cv2.line(frame, points[a], points[b], (0, 255, 0), 2)
        for x, y in points:
            cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)
        return points

    # ========================================================
    # WEBCAM — MEDIA PIPE LEGACY
    # ========================================================
    def process_legacy_hand(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            h, w, _ = frame.shape
            cx = int(hand_landmarks.landmark[8].x * w)
            cy = int(hand_landmarks.landmark[8].y * h)
            return cx, cy
        return None

    # ========================================================
    # WEBCAM — MEDIA PIPE TASKS
    # ========================================================
    def process_tasks_hand(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.hands.detect(mp_image)
        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            points = self.draw_hand_landmarks(frame, landmarks)
            if len(points) > 8:
                return points[8]
        return None

    # ========================================================
    # WEBCAM — LOOP PRINCIPAL (COM SUAVIZAÇÃO E CONTROLE)
    # ========================================================
    def update_webcam(self):
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)
        point = None

        if MEDIAPIPE_DISPONIVEL and self.hands:
            try:
                if MEDIAPIPE_MODO == "legacy":
                    point = self.process_legacy_hand(frame)
                elif MEDIAPIPE_MODO == "tasks":
                    point = self.process_tasks_hand(frame)
            except Exception:
                cv2.putText(frame, "MediaPipe: erro de leitura", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                self._reset_webcam_drawing_state()
        else:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.GaussianBlur(gray_frame, (15, 15), 0)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray_frame)
            if max_val > 252:
                point = max_loc

        now = time.time() * 1000.0

        if point is not None:
            self.lost_frames = 0
            cx, cy = float(point[0]), float(point[1])

            # --- 1. Suavização EMA ---
            if self.smooth_x == 0.0 and self.smooth_y == 0.0:
                self.smooth_x, self.smooth_y = cx, cy
            else:
                self.smooth_x = self.alpha * cx + (1.0 - self.alpha) * self.smooth_x
                self.smooth_y = self.alpha * cy + (1.0 - self.alpha) * self.smooth_y

            sx, sy = int(self.smooth_x), int(self.smooth_y)

            # --- 2. Detecção de movimento / mão parada ---
            if self.prev_x == 0 and self.prev_y == 0:
                self.prev_x, self.prev_y = sx, sy
                self.last_move_time = now
            else:
                dist = math.hypot(sx - self.prev_x, sy - self.prev_y)

                if dist > self.move_threshold:
                    self.last_move_time = now

                    # --- 3. Espessura dinâmica pela velocidade ---
                    thickness = int(np.clip(22 - dist * 0.35, 10, 22))
                    cv2.line(
                        self.webcam_mask,
                        (self.prev_x, self.prev_y),
                        (sx, sy),
                        255,
                        thickness,
                        lineType=cv2.LINE_AA,
                    )
                    self.prev_x, self.prev_y = sx, sy
                else:
                    # Mão praticamente parada: não desenha e evita micro-ruído
                    if now - self.last_move_time > self.pause_ms:
                        # Mantém o ponto atual como referência sem riscar
                        self.prev_x, self.prev_y = sx, sy

            cv2.circle(frame, (sx, sy), 10, (0, 255, 0), cv2.FILLED)
        else:
            # --- 4. Quebra de traço quando a mão some ---
            self.lost_frames += 1
            if self.lost_frames > self.max_lost:
                self.prev_x = 0
                self.prev_y = 0
                self.smooth_x = 0.0
                self.smooth_y = 0.0

        # Overlay do mask no frame
        mask_bgr = cv2.cvtColor(self.webcam_mask, cv2.COLOR_GRAY2BGR)
        mask_pixels = (mask_bgr[:, :, 0] > 0)
        mask_bgr[mask_pixels] = [0, 255, 0]
        combined = cv2.addWeighted(frame, 1.0, mask_bgr, 0.7, 0)

        if MEDIAPIPE_DISPONIVEL:
            status = "MediaPipe Legacy" if MEDIAPIPE_MODO == "legacy" else "MediaPipe Tasks"
            cv2.putText(combined, status, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        else:
            cv2.putText(combined, "OpenCV fallback", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)

        rgb_combined = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_combined.shape
        qt_img = QImage(rgb_combined.data, w, h, ch * w, QImage.Format_RGB888)
        self.lbl_webcam.setPixmap(
            QPixmap.fromImage(qt_img).scaled(540, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    # ========================================================
    # PIPELINE DE IA — MNIST
    # ========================================================
    def run_prediction(self):
        if not self.model:
            self.lbl_resultado.setText("ERRO")
            self.lbl_confianca.setText("Modelo não carregado!")
            return

        img_gray = None

        if self.active_mode == "mouse":
            img_gray = self.canvas.get_numpy_array()
        elif self.active_mode == "upload" and self.current_image_path:
            raw = cv2.imread(self.current_image_path, cv2.IMREAD_GRAYSCALE)
            if raw is not None:
                _, img_gray = cv2.threshold(raw, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        elif self.active_mode == "webcam":
            img_gray = self.webcam_mask.copy()
            # Limpeza morfológica leve para deixar o traço mais “cheio” e próximo do MNIST
            kernel = np.ones((3, 3), np.uint8)
            img_gray = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)
            img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
            _, img_gray = cv2.threshold(img_gray, 40, 255, cv2.THRESH_BINARY)

        if img_gray is None:
            return

        coords = cv2.findNonZero(img_gray)
        if coords is None:
            self.lbl_resultado.setText("-")
            self.lbl_confianca.setText("Confiança: 0%")
            self.progress_bar.setValue(0)
            return

        x, y, w, h = cv2.boundingRect(coords)
        img_cropped = img_gray[y:y + h, x:x + w]
        escala = 20.0 / max(max(w, h), 1)
        novo_w = max(1, int(w * escala))
        novo_h = max(1, int(h * escala))

        img_resized = cv2.resize(img_cropped, (novo_w, novo_h), interpolation=cv2.INTER_AREA)
        img_final = np.zeros((28, 28), dtype=np.uint8)

        y_off = (28 - novo_h) // 2
        x_off = (28 - novo_w) // 2
        img_final[y_off:y_off + novo_h, x_off:x_off + novo_w] = img_resized

        qt_mini = QImage(img_final.data, 28, 28, 28, QImage.Format_Grayscale8)
        self.lbl_mini.setPixmap(QPixmap.fromImage(qt_mini).scaled(140, 140, Qt.KeepAspectRatio))

        img_vector = img_final.reshape(1, 784).astype("float32") / 255.0
        pred_probs = self.model.predict(img_vector, verbose=0)[0]

        classe = int(np.argmax(pred_probs))
        confianca = float(pred_probs[classe] * 100)

        self.lbl_resultado.setText(str(classe))
        self.lbl_confianca.setText(f"Confiança: {confianca:.1f}%")
        self.progress_bar.setValue(max(0, min(100, int(confianca))))

    # ========================================================
    # ESTILO E CORES
    # ========================================================
    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0B132B;
            }
            QLabel {
                color: #f8fafc;
                font-family: 'Segoe UI', Arial;
                font-size: 13px;
            }
            QLabel#header {
                font-size: 16px;
                font-weight: bold;
                color: #94a3b8;
                margin-bottom: 8px;
            }
            QLabel#resultadoBig {
                font-size: 110px;
                font-weight: bold;
                color: #0ABAB5;
            }
            QFrame#panel {
                background-color: #1C2541;
                border-radius: 12px;
            }
            QPushButton {
                background-color: #3A506B;
                color: white;
                border: none;
                padding: 12px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 13px;
                margin-bottom: 6px;
            }
            QPushButton:hover {
                background-color: #5BC0BE;
                color: #0B132B;
            }
            QPushButton#btnRojo {
                background-color: #1C2541;
                border: 2px solid #E63946;
                color: #E63946;
                margin-top: 10px;
            }
            QPushButton#btnRojo:hover {
                background-color: #E63946;
                color: white;
            }
            QPushButton#btnTiffany {
                background-color: #0ABAB5;
                color: #0B132B;
                margin-top: 10px;
                font-size: 14px;
                padding: 15px;
            }
            QPushButton#btnTiffany:hover {
                background-color: #5BC0BE;
            }
            QProgressBar {
                border: none;
                background-color: #3A506B;
                border-radius: 7px;
                height: 14px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #0ABAB5;
                border-radius: 7px;
            }
        """)

    # ========================================================
    # ENCERRAMENTO
    # ========================================================
    def closeEvent(self, event):
        self.stop_webcam()
        try:
            if self.hands is not None:
                if MEDIAPIPE_MODO == "legacy":
                    self.hands.close()
                elif MEDIAPIPE_MODO == "tasks":
                    self.hands.close()
        except Exception:
            pass
        event.accept()

# ============================================================
# EXECUÇÃO
# ============================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MNISTApp()
    window.show()
    sys.exit(app.exec())
