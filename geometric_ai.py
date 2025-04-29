import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import mediapipe as mp
import threading
from PIL import Image, ImageTk
import math
import random

# Configuración de MediaPipe para el seguimiento de manos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class GeometricAI:
    def __init__(self, root):
        self.root = root
        self.root.title("Asistente Geométrico con IA")
        
        # Variables de estado
        self.selected_shape = ""
        self.drawn_shapes = []
        self.selected_figure = None
        self.hand_closed = False
        self.last_hand_pos = None
        
        # Configurar interfaz gráfica
        self.setup_ui()
        
        # Inicializar cámara
        self.cap = cv2.VideoCapture(0)
        self.current_frame = None
        
        # Inicializar modelo de manos
        self.hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5)
        
        # Hilo para procesamiento de video
        self.thread = threading.Thread(target=self.update_video)
        self.thread.daemon = True
        self.thread.start()
        
        # Evento de cierre
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self.pointer_dot = None  # NUEVO: Para el punto en el canvas
        self.pointing_gesture = False  # NUEVO: Estado del gesto

        self.pointer_dot = None
        self.last_canvas_coords = (0, 0)
    def setup_ui(self):
        """Configura la interfaz gráfica de usuario"""
        # Panel de control izquierdo
        control_frame = tk.Frame(self.root, width=200)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Entrada de texto
        self.shape_entry = ttk.Entry(control_frame)
        self.shape_entry.pack(pady=10, fill=tk.X)
        
        # Botones
        ttk.Button(control_frame, text="Generar Figura", 
                 command=self.generate_shape).pack(pady=5, fill=tk.X)
        ttk.Button(control_frame, text="Mejorar Figura", 
                 command=self.improve_shape).pack(pady=5, fill=tk.X)
        
        # Canvas para dibujar
        self.canvas = tk.Canvas(self.root, width=800, height=600, bg="white")
        self.canvas.pack(side=tk.RIGHT, padx=10, pady=10)
        
        # Etiqueta para video
        self.video_label = tk.Label(self.root)
        self.video_label.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.canvas.bind("<Button-1>", self.select_figure_with_click)  

    

    def select_figure_with_click(self, event):
        """Selecciona figuras haciendo clic en ellas"""
        x, y = event.x, event.y
        items = self.canvas.find_overlapping(x-5, y-5, x+5, y+5)
    
        if items:
            self.selected_figure = items[-1]
            self.highlight_selected_figure()
        
    def highlight_selected_figure(self):
        """Resalta la figura seleccionada"""
        # Quitar resaltado anterior
        self.canvas.itemconfig("figure", width=2, outline="black")
    
        # Resaltar selección actual
        if self.selected_figure:
            self.canvas.itemconfig(self.selected_figure, width=4, outline="red")

    def generate_shape(self):
        """Genera una figura geométrica perfecta basada en texto"""
        shape_name = self.shape_entry.get().lower()
        shapes = {
            "circulo": self.draw_circle,
            "cuadrado": self.draw_square,
            "triangulo": self.draw_triangle,
            "rectangulo": self.draw_rectangle
        }
        
        if shape_name in shapes:
            shapes[shape_name]()
        else:
            messagebox.showerror("Error", "Figura no reconocida")

    def recognize_shape(self, contour):
        """Reconoce la forma geométrica a partir de un contorno"""
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        sides = len(approx)
        
        if sides == 3:
            return "triangulo"
        elif sides == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w)/h
            return "cuadrado" if 0.95 <= aspect_ratio <= 1.05 else "rectangulo"
        else:
            return "circulo"

    def improve_shape(self):
        """Mejora una figura dibujada a mano alzada"""
        if self.current_frame is not None:
            # Procesamiento de imagen
            gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                shape_type = self.recognize_shape(largest_contour)
                self.draw_perfect_shape(shape_type, largest_contour)

    def draw_perfect_shape(self, shape_type, contour):
        """Dibuja una versión perfecta de la figura detectada"""
        x, y, w, h = cv2.boundingRect(contour)
        center = (x + w//2, y + h//2)
        color = "#{:02x}{:02x}{:02x}".format(*np.random.randint(0, 255, 3))
        
        self.canvas.delete("current")  # Eliminar dibujo imperfecto
        
        if shape_type == "circulo":
            radius = max(w, h) // 2
            self.canvas.create_oval(
                center[0]-radius, center[1]-radius,
                center[0]+radius, center[1]+radius,
                fill=color, tags=("perfect", "circle"))
        elif shape_type == "cuadrado":
            size = max(w, h)
            self.canvas.create_rectangle(
                center[0]-size//2, center[1]-size//2,
                center[0]+size//2, center[1]+size//2,
                fill=color, tags=("perfect", "square"))
        elif shape_type == "rectangulo":
            self.canvas.create_rectangle(
                center[0]-w//2, center[1]-h//2,
                center[0]+w//2, center[1]+h//2,
                fill=color, tags=("perfect", "rectangle"))
        elif shape_type == "triangulo":
            points = [
                center[0], center[1]-h//2,
                center[0]-w//2, center[1]+h//2,
                center[0]+w//2, center[1]+h//2
            ]
            self.canvas.create_polygon(
                points, fill=color,
                tags=("perfect", "triangle"))
        
        self.selected_figure = self.canvas.find_withtag("current")

    # Métodos de dibujo de figuras
    def draw_circle(self):
        x = random.randint(100, 700)
        y = random.randint(100, 500)
        radius = random.randint(30, 100)
        color = "#{:02x}{:02x}{:02x}".format(*np.random.randint(0, 255, 3))
        
        figure = self.canvas.create_oval(
            x - radius, y - radius,
            x + radius, y + radius,
            fill=color, tags=("figure", "circle"))
        self.drawn_shapes.append(figure)
        self.selected_figure = figure

    def draw_square(self):
        size = random.randint(50, 150)
        x = random.randint(100, 700 - size)
        y = random.randint(100, 500 - size)
        color = "#{:02x}{:02x}{:02x}".format(*np.random.randint(0, 255, 3))
        
        figure = self.canvas.create_rectangle(
            x, y, x + size, y + size,
            fill=color, tags=("figure", "square"))
        self.drawn_shapes.append(figure)
        self.selected_figure = figure

    def draw_triangle(self):
        side = random.randint(80, 150)
        x = random.randint(150, 650)
        y = random.randint(150, 450)
        height = side * math.sqrt(3) / 2
        points = [
            x, y,
            x - side/2, y + height,
            x + side/2, y + height
        ]
        color = "#{:02x}{:02x}{:02x}".format(*np.random.randint(0, 255, 3))
        
        figure = self.canvas.create_polygon(
            points, fill=color,
            tags=("figure", "triangle"))
        self.drawn_shapes.append(figure)
        self.selected_figure = figure

    def draw_rectangle(self):
        width = random.randint(80, 200)
        height = random.randint(50, 150)
        x = random.randint(100, 700 - width)
        y = random.randint(100, 500 - height)
        color = "#{:02x}{:02x}{:02x}".format(*np.random.randint(0, 255, 3))
        
        figure = self.canvas.create_rectangle(
            x, y, x + width, y + height,
            fill=color, tags=("figure", "rectangle"))
        self.drawn_shapes.append(figure)
        self.selected_figure = figure

    def update_video(self):
        """Actualiza el feed de video y procesa los gestos"""
        while True:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                
                if results.multi_hand_landmarks:
                    self.process_gesture(results.multi_hand_landmarks[0], frame)
                
                self.current_frame = frame
                self.show_video(frame)

    def process_gesture(self, landmarks, frame):
        """Procesa los gestos de la mano en tiempo real"""
        # Obtener puntos clave de los dedos
        index = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
        thumb = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

        # Detectar articulaciones para la lógica del gesto
        index_pip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
        middle_mcp = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        ring_mcp = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
        pinky_mcp = landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

        # 1. Detectar gesto de señalar
        index_extended = index.y < index_pip.y  # Índice extendido
        other_fingers_closed = all([
        middle.y > middle_mcp.y,
        ring.y > ring_mcp.y,
        pinky.y > pinky_mcp.y
    ])
        self.pointing_gesture = index_extended and other_fingers_closed

        # 2. Convertir coordenadas a espacio del canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        canvas_x = int(index.x * canvas_width)
        canvas_y = int(index.y * canvas_height)

    # 3. Actualizar punto indicador
        if self.pointing_gesture:
            self.update_pointer_dot(canvas_x, canvas_y)
            self.select_figure_with_gesture(canvas_x, canvas_y)
        else:
            self.remove_pointer_dot()
            self.deselect_figure()

        # 4. Detectar gesto de mano cerrada para mover
        thumb_to_middle = math.hypot(
            thumb.x - middle.x,
            thumb.y - middle.y
        )
        self.hand_closed = thumb_to_middle < 0.05  # Umbral ajustable

        # 5. Mover figura si está seleccionada y mano cerrada
        current_hand_pos = (index.x * canvas_width, index.y * canvas_height)
    
        if self.hand_closed and self.selected_figure and self.last_hand_pos:
            dx = current_hand_pos[0] - self.last_hand_pos[0]
            dy = current_hand_pos[1] - self.last_hand_pos[1]
            self.canvas.move(self.selected_figure, dx, dy)

        self.last_hand_pos = current_hand_pos

        # 6. Dibujar landmarks en el feed de video (opcional)
        mp_drawing.draw_landmarks(
            frame,
            landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2))
    def update_pointer_dot(self, x, y):
        """Actualiza la posición del punto indicador en el canvas"""
        if self.pointer_dot:
            self.canvas.coords(self.pointer_dot, x-5, y-5, x+5, y+5)
        else:
            self.pointer_dot = self.canvas.create_oval(
                x-5, y-5, x+5, y+5,
                fill='red', outline='', tags="pointer"
            )
    def update_pointer_dot(self, x, y):
        """Actualiza el punto indicador en el canvas"""
        dot_size = 8
        dot_color = "#ff0000"
        
        if self.pointer_dot:
            self.canvas.coords(self.pointer_dot, 
                              x - dot_size, y - dot_size,
                              x + dot_size, y + dot_size)
        else:
            self.pointer_dot = self.canvas.create_oval(
                x - dot_size, y - dot_size,
                x + dot_size, y + dot_size,
                fill=dot_color, outline="", tags="pointer"
            )
            
    def remove_pointer_dot(self):
        """Elimina el punto del canvas"""
        if self.pointer_dot:
            self.canvas.delete(self.pointer_dot)
            self.pointer_dot = None

    def select_figure_with_gesture(self, x, y):
        """Selecciona la figura más cercana al punto con lógica mejorada"""
        detection_radius = 25  # Radio aumentado para mejor detección
        items = self.canvas.find_overlapping(
            x - detection_radius, 
            y - detection_radius, 
            x + detection_radius, 
            y + detection_radius
        )
    
        closest_figure = None
        min_distance = float('inf')
    
        for item in items:
            # Filtrar solo figuras geométricas
            tags = self.canvas.gettags(item)
            if "figure" in tags or "perfect" in tags:
                # Obtener coordenadas del centro de la figura
                coords = self.canvas.coords(item)
            
                # Calcular centro según el tipo de figura
                if "circle" in tags or "oval" in tags:
                    cx = (coords[0] + coords[2]) / 2
                    cy = (coords[1] + coords[3]) / 2
                elif "rectangle" in tags or "square" in tags:
                    cx = (coords[0] + coords[2]) / 2
                    cy = (coords[1] + coords[3]) / 2
                elif "triangle" in tags:
                    cx = sum(coords[0::2]) / 3
                    cy = sum(coords[1::2]) / 3
            
                # Calcular distancia al punto
                distance = math.hypot(x - cx, y - cy)
            
                # Mantener la figura más cercana
                if distance < min_distance:
                    min_distance = distance
                    closest_figure = item
    
        if closest_figure and closest_figure != self.selected_figure:
            # Deseleccionar anterior
            self.deselect_figure()
            # Seleccionar nuevo
            self.selected_figure = closest_figure
            self.highlight_selected_figure()
            print(f"Figura seleccionada: {self.canvas.gettags(closest_figure)}")  # Debug
    def remove_pointer_dot(self):
        """Elimina el punto indicador del canvas"""
        if self.pointer_dot:
            self.canvas.delete(self.pointer_dot)
            self.pointer_dot = None

    def deselect_figure(self):
        """Deselecciona la figura actual"""
        if self.selected_figure:
            self.canvas.itemconfig(self.selected_figure, width=2, outline="black")
            self.selected_figure = None

    def show_video(self, frame):
        """Muestra el video en la interfaz"""
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.config(image=imgtk)
        self.video_label.image = imgtk

    def on_close(self):
        """Maneja el cierre de la aplicación"""
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = GeometricAI(root)
    root.mainloop()