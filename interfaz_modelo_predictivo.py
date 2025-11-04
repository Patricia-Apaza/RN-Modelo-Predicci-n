import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
import threading
import os
from datetime import datetime

# Importaciones de ML
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class ModeloPredictivo:
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 Sistema de Predicción con Redes Neuronales - Análisis Multivariado")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Variables
        self.df = None
        self.ruta_archivo = None
        self.columna_objetivo = None
        self.modelo = None
        self.scaler = None
        self.resultados = {}
        self.recomendaciones = {}
        
        # Crear interfaz
        self.crear_interfaz()
        
    def crear_interfaz(self):
        # ===== TÍTULO =====
        frame_titulo = tk.Frame(self.root, bg='#2c3e50', height=80)
        frame_titulo.pack(fill='x')
        frame_titulo.pack_propagate(False)
        
        titulo = tk.Label(
            frame_titulo, 
            text="🧠 SISTEMA DE PREDICCIÓN CON REDES NEURONALES", 
            font=('Arial', 20, 'bold'),
            fg='white',
            bg='#2c3e50'
        )
        titulo.pack(pady=15)
        
        subtitulo = tk.Label(
            frame_titulo,
            text="Con Recomendaciones Inteligentes - Entrena automáticamente tu modelo predictivo",
            font=('Arial', 10),
            fg='#ecf0f1',
            bg='#2c3e50'
        )
        subtitulo.pack()
        
        # ===== CONTENEDOR PRINCIPAL CON PANED WINDOW (RESPONSIVO) =====
        contenedor = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg='#f0f0f0', 
                                   sashwidth=8, sashrelief=tk.RAISED)
        contenedor.pack(fill='both', expand=True, padx=20, pady=20)
        
        # ===== PANEL IZQUIERDO: CONTROLES =====
        panel_izq = tk.Frame(contenedor, bg='white', relief='raised', bd=2)
        contenedor.add(panel_izq, minsize=400, width=450)
        
        # PASO 1: Cargar Dataset
        self.crear_seccion_carga(panel_izq)
        
        # PASO 2: Configuración
        self.crear_seccion_configuracion(panel_izq)
        
        # PASO 3: Entrenar
        self.crear_seccion_entrenamiento(panel_izq)
        
        # ===== PANEL DERECHO: RESULTADOS =====
        panel_der = tk.Frame(contenedor, bg='white', relief='raised', bd=2)
        contenedor.add(panel_der, minsize=400)
        
        self.crear_seccion_resultados(panel_der)
        
    def crear_seccion_carga(self, parent):
        frame = tk.LabelFrame(parent, text="📂 PASO 1: Cargar Dataset", 
                             font=('Arial', 12, 'bold'), bg='white', fg='#2c3e50')
        frame.pack(fill='x', padx=10, pady=10)
        
        # Botón cargar
        self.btn_cargar = tk.Button(
            frame,
            text="📁 Seleccionar Archivo CSV/Excel",
            command=self.cargar_archivo,
            bg='#3498db',
            fg='white',
            font=('Arial', 11, 'bold'),
            cursor='hand2',
            relief='raised',
            bd=3
        )
        self.btn_cargar.pack(pady=10, padx=10, fill='x')
        
        # Info del archivo
        self.lbl_archivo = tk.Label(
            frame,
            text="📄 Ningún archivo seleccionado",
            font=('Arial', 9),
            bg='white',
            fg='#7f8c8d'
        )
        self.lbl_archivo.pack(pady=5)
        
        # Info del dataset
        self.lbl_info_dataset = tk.Label(
            frame,
            text="",
            font=('Arial', 9),
            bg='white',
            fg='#27ae60',
            justify='left'
        )
        self.lbl_info_dataset.pack(pady=5)
        
    def crear_seccion_configuracion(self, parent):
        frame = tk.LabelFrame(parent, text="⚙️ PASO 2: Configuración", 
                             font=('Arial', 12, 'bold'), bg='white', fg='#2c3e50')
        frame.pack(fill='x', padx=10, pady=10)
        
        # Columna objetivo
        tk.Label(frame, text="Variable a predecir:", bg='white', 
                font=('Arial', 10, 'bold')).pack(pady=(10,5), padx=10, anchor='w')
        
        self.combo_objetivo = ttk.Combobox(frame, state='disabled', font=('Arial', 10))
        self.combo_objetivo.pack(pady=5, padx=10, fill='x')
        self.combo_objetivo.bind('<<ComboboxSelected>>', self.mostrar_info_variable)
        
        # Frame para recomendación compacta
        self.frame_recomendacion = tk.Frame(frame, bg='#e8f5e9', relief='solid', bd=1)
        self.frame_recomendacion.pack(fill='x', padx=10, pady=5)
        
        self.lbl_recomendacion = tk.Label(
            self.frame_recomendacion,
            text="💡 Selecciona una variable para ver recomendaciones detalladas",
            font=('Arial', 8),
            bg='#e8f5e9',
            fg='#2e7d32',
            justify='left',
            wraplength=380
        )
        self.lbl_recomendacion.pack(pady=5, padx=5)
        
        # Porcentaje de prueba
        tk.Label(frame, text="Porcentaje de datos para prueba:", 
                bg='white', font=('Arial', 10, 'bold')).pack(pady=(10,5), padx=10, anchor='w')
        
        frame_slider = tk.Frame(frame, bg='white')
        frame_slider.pack(fill='x', padx=10)
        
        self.slider_prueba = tk.Scale(
            frame_slider,
            from_=10,
            to=40,
            orient='horizontal',
            bg='white',
            font=('Arial', 9),
            length=200
        )
        self.slider_prueba.set(20)
        self.slider_prueba.pack(side='left', fill='x', expand=True)
        
        self.lbl_porcentaje = tk.Label(frame_slider, text="20%", bg='white', 
                                      font=('Arial', 10, 'bold'), fg='#3498db')
        self.lbl_porcentaje.pack(side='right', padx=10)
        self.slider_prueba.config(command=self.actualizar_porcentaje)
        
        # Épocas
        tk.Label(frame, text="Número de épocas:", 
                bg='white', font=('Arial', 10, 'bold')).pack(pady=(10,5), padx=10, anchor='w')
        
        self.spin_epocas = tk.Spinbox(
            frame,
            from_=50,
            to=500,
            increment=10,
            font=('Arial', 10),
            width=15
        )
        self.spin_epocas.delete(0, 'end')
        self.spin_epocas.insert(0, '150')
        self.spin_epocas.pack(pady=5, padx=10, anchor='w')
        
    def crear_seccion_entrenamiento(self, parent):
        frame = tk.LabelFrame(parent, text="🚀 PASO 3: Entrenar Modelo", 
                             font=('Arial', 12, 'bold'), bg='white', fg='#2c3e50')
        frame.pack(fill='x', padx=10, pady=10)
        
        # Botón entrenar
        self.btn_entrenar = tk.Button(
            frame,
            text="🎯 ENTRENAR MODELO",
            command=self.entrenar_modelo,
            bg='#27ae60',
            fg='white',
            font=('Arial', 12, 'bold'),
            cursor='hand2',
            state='disabled',
            relief='raised',
            bd=3,
            height=2
        )
        self.btn_entrenar.pack(pady=10, padx=10, fill='x')
        
        # Barra de progreso
        self.progress = ttk.Progressbar(frame, mode='indeterminate')
        self.progress.pack(pady=5, padx=10, fill='x')
        
        # Estado
        self.lbl_estado = tk.Label(
            frame,
            text="⏳ Esperando datos...",
            font=('Arial', 9),
            bg='white',
            fg='#7f8c8d'
        )
        self.lbl_estado.pack(pady=5)
        
    def crear_seccion_resultados(self, parent):
        # Título
        tk.Label(
            parent,
            text="📊 RESULTADOS DEL ENTRENAMIENTO",
            font=('Arial', 14, 'bold'),
            bg='white',
            fg='#2c3e50'
        ).pack(pady=10)
        
        # Área de texto con scroll
        self.txt_resultados = scrolledtext.ScrolledText(
            parent,
            wrap=tk.WORD,
            font=('Consolas', 9),
            bg='#f8f9fa',
            fg='#2c3e50',
            relief='sunken',
            bd=2
        )
        self.txt_resultados.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Mensaje inicial
        self.txt_resultados.insert('1.0', 
            "🎓 Bienvenido al Sistema de Predicción Inteligente\n\n"
            "📌 INSTRUCCIONES:\n"
            "1. Carga tu archivo CSV o Excel\n"
            "2. El sistema analizará tus datos y te recomendará la mejor variable\n"
            "3. Selecciona la variable que quieres predecir\n"
            "4. Ajusta la configuración si lo deseas\n"
            "5. Presiona 'ENTRENAR MODELO'\n"
            "6. Espera los resultados (2-5 minutos)\n\n"
            "✨ Características especiales:\n"
            "• Análisis inteligente de variables\n"
            "• Recomendaciones automáticas\n"
            "• Explicaciones detalladas\n"
            "• Gráficas y reportes automáticos\n"
        )
        self.txt_resultados.config(state='disabled')
        
        # Frame de botones inferiores
        frame_botones = tk.Frame(parent, bg='white')
        frame_botones.pack(fill='x', padx=10, pady=5)
        
        self.btn_guardar = tk.Button(
            frame_botones,
            text="💾 Guardar Resultados",
            command=self.guardar_resultados,
            bg='#9b59b6',
            fg='white',
            font=('Arial', 10, 'bold'),
            state='disabled'
        )
        self.btn_guardar.pack(side='left', padx=5)
        
        self.btn_graficas = tk.Button(
            frame_botones,
            text="📈 Ver Gráficas",
            command=self.mostrar_graficas,
            bg='#e67e22',
            fg='white',
            font=('Arial', 10, 'bold'),
            state='disabled'
        )
        self.btn_graficas.pack(side='left', padx=5)
        
    def actualizar_porcentaje(self, valor):
        self.lbl_porcentaje.config(text=f"{int(float(valor))}%")
        
    def log(self, mensaje, color='black'):
        """Agregar mensaje al área de resultados"""
        self.txt_resultados.config(state='normal')
        self.txt_resultados.insert('end', mensaje + '\n')
        self.txt_resultados.see('end')
        self.txt_resultados.config(state='disabled')
        self.root.update()
    
    def generar_consejos_variable(self, col, datos_col, analisis_info):
        """
        Genera consejos inteligentes basados en el análisis de la variable
        """
        consejos = []
        col_lower = col.lower()
        
        media = analisis_info['media']
        std = analisis_info['std']
        cv = analisis_info['cv']
        minimo = analisis_info['min']
        maximo = analisis_info['max']
        
        # Calcular percentiles
        q25 = np.percentile(datos_col, 25)
        q75 = np.percentile(datos_col, 75)
        mediana = np.median(datos_col)
        
        # Detectar tendencia (valores recientes vs antiguos)
        primera_mitad = datos_col[:len(datos_col)//2].mean()
        segunda_mitad = datos_col[len(datos_col)//2:].mean()
        cambio_porcentual = ((segunda_mitad - primera_mitad) / primera_mitad * 100) if primera_mitad != 0 else 0
        
        # 1. CONSEJOS DE INVERSIÓN/COMPRA (para precios, costos, valores)
        if any(k in col_lower for k in ['price', 'precio', 'cost', 'costo', 'value', 'valor', 'rent', 'alquiler']):
            # Análisis de nivel de precios
            if media < mediana * 0.9:  # Media menor que mediana indica valores bajos predominantes
                consejos.append("💰 OPORTUNIDAD DE COMPRA: Los precios actuales están por debajo del promedio histórico. Es un buen momento para invertir.")
            elif media > mediana * 1.1:
                consejos.append("⚠️ PRECAUCIÓN: Los precios están por encima del promedio. Considera esperar una corrección del mercado.")
            else:
                consejos.append("📊 MERCADO ESTABLE: Los precios están en niveles normales. Momento neutro para comprar.")
            
            # Análisis de variación
            if cv > 50:
                consejos.append("🎲 ALTO RIESGO: Hay mucha volatilidad en los precios. Si inviertes, diversifica para reducir riesgo.")
            elif cv < 20:
                consejos.append("🛡️ BAJO RIESGO: Los precios son estables y predecibles. Inversión segura.")
            
            # Comparación con percentiles
            valores_bajos = (datos_col < q25).sum()
            porcentaje_bajos = (valores_bajos / len(datos_col)) * 100
            if porcentaje_bajos > 30:
                consejos.append(f"💎 OPORTUNIDAD: El {porcentaje_bajos:.0f}% de los valores están en el rango bajo (${minimo:,.0f} - ${q25:,.0f}). Busca ofertas en ese rango.")
        
        # 2. CONSEJOS DE TENDENCIAS (si están subiendo/bajando)
        if cambio_porcentual > 10:
            consejos.append(f"📈 TENDENCIA ALCISTA: Los valores han aumentado un {cambio_porcentual:.1f}% en el período analizado. El mercado está en crecimiento.")
            if any(k in col_lower for k in ['price', 'precio', 'value', 'valor']):
                consejos.append("⏰ ACTÚA PRONTO: Si planeas comprar, hazlo antes de que los precios sigan subiendo.")
        elif cambio_porcentual < -10:
            consejos.append(f"📉 TENDENCIA BAJISTA: Los valores han disminuido un {abs(cambio_porcentual):.1f}%. El mercado está en corrección.")
            if any(k in col_lower for k in ['price', 'precio', 'value', 'valor']):
                consejos.append("🎯 MOMENTO IDEAL: Los precios están cayendo. Excelente oportunidad para comprar con descuento.")
        elif abs(cambio_porcentual) < 5:
            consejos.append(f"➡️ TENDENCIA LATERAL: Los valores se mantienen estables (variación: {cambio_porcentual:.1f}%). Mercado sin dirección clara.")
        
        # 3. CONSEJOS DE OPORTUNIDADES (momento ideal para actuar)
        # Detectar valores outliers bajos (oportunidades)
        outliers_bajos = datos_col[datos_col < (media - 2*std)]
        if len(outliers_bajos) > 0:
            consejos.append(f"🌟 GANGAS DETECTADAS: Hay {len(outliers_bajos)} valores excepcionalmente bajos (por debajo de ${(media - 2*std):,.0f}). ¡Oportunidades únicas!")
        
        # Análisis de distribución
        if minimo < q25 * 0.7:
            consejos.append(f"💥 SUPER OFERTA: El valor mínimo (${minimo:,.0f}) está muy por debajo del rango normal. Si encuentras algo a ese precio, ¡cómpralo!")
        
        # Para áreas, habitaciones, tamaño
        if any(k in col_lower for k in ['area', 'size', 'sqft', 'metros', 'habitacion', 'bedroom', 'room']):
            if media > mediana:
                consejos.append(f"🏠 ESPACIOS AMPLIOS: La mayoría de propiedades tienen {mediana:.0f} unidades, pero el promedio es {media:.0f}. Hay opciones grandes disponibles.")
            consejos.append(f"📏 RANGO ÓPTIMO: Busca valores entre {q25:.0f} y {q75:.0f} para obtener la mejor relación calidad-precio.")
        
        # 4. ALERTAS DE RIESGO (datos anómalos, mucha variación)
        if cv > 100:
            consejos.append(f"⚠️ ALERTA MÁXIMA: Variabilidad extrema (CV: {cv:.0f}%). El mercado es impredecible. No inviertas todo tu capital.")
        
        # Rango muy amplio
        ratio_rango = (maximo / minimo) if minimo > 0 else float('inf')
        if ratio_rango > 10:
            consejos.append(f"📊 MERCADO DIVERSO: Los valores varían desde ${minimo:,.0f} hasta ${maximo:,.0f} (ratio {ratio_rango:.1f}x). Hay opciones para todos los presupuestos.")
        
        # Valores extremos
        if maximo > media * 3:
            consejos.append(f"👑 SEGMENTO PREMIUM: Existen opciones de lujo hasta ${maximo:,.0f}. El mercado tiene productos para todos los segmentos.")
        
        # 5. PREDICCIONES FUTURAS (qué podría pasar)
        if cambio_porcentual > 15:
            consejos.append(f"🔮 PREDICCIÓN: Si la tendencia continúa (+{cambio_porcentual:.1f}%), los valores podrían alcanzar ${media * (1 + cambio_porcentual/100):,.0f} en el próximo período.")
            consejos.append("💡 RECOMENDACIÓN: Si eres vendedor, es buen momento. Si eres comprador, actúa rápido antes de que suban más.")
        elif cambio_porcentual < -15:
            consejos.append(f"🔮 PREDICCIÓN: La caída continúa ({cambio_porcentual:.1f}%). Los valores podrían bajar a ${media * (1 + cambio_porcentual/100):,.0f} si la tendencia persiste.")
            consejos.append("💡 RECOMENDACIÓN: Espera un poco más si puedes, los precios podrían seguir bajando.")
        else:
            consejos.append(f"🔮 PREDICCIÓN: El mercado está estable. Se espera que los valores se mantengan alrededor de ${media:,.0f} en el corto plazo.")
        
        # Consejo estacional/temporal (basado en datos)
        if any(k in col_lower for k in ['price', 'precio']):
            if media < q25 * 1.2:
                consejos.append("🎁 MOMENTO ESTRATÉGICO: Los precios actuales favorecen a los compradores. Negocia con confianza.")
            elif media > q75 * 0.8:
                consejos.append("💼 MOMENTO VENDEDOR: Los precios favorecen a los vendedores. Si tienes algo que vender, este es tu momento.")
        
        # Consejo de diversificación
        if cv > 40:
            consejos.append("🎯 ESTRATEGIA: Debido a la alta variación, considera diversificar tu inversión en diferentes rangos de precio.")
        
        return consejos
    
    def analizar_variables(self, df):
        """
        Analiza todas las variables numéricas y determina cuál es la mejor para predecir
        """
        columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(columnas_numericas) == 0:
            return None, {}
        
        analisis = {}
        
        for col in columnas_numericas:
            datos_col = df[col]
            
            # Calcular características
            media = datos_col.mean()
            std = datos_col.std()
            cv = (std / media * 100) if media != 0 else 0  # Coeficiente de variación
            rango = datos_col.max() - datos_col.min()
            valores_unicos = datos_col.nunique()
            porcentaje_unicos = (valores_unicos / len(datos_col)) * 100
            
            # Puntaje de aptitud para ser variable objetivo
            puntaje = 0
            razones = []
            
            # 1. Palabras clave que sugieren variable objetivo
            col_lower = col.lower()
            keywords_precio = ['price', 'precio', 'cost', 'costo', 'value', 'valor', 
                             'rent', 'alquiler', 'sale', 'venta', 'amount', 'monto']
            keywords_cantidad = ['total', 'sum', 'cantidad', 'count', 'numero']
            
            if any(keyword in col_lower for keyword in keywords_precio):
                puntaje += 50
                razones.append("✅ Nombre sugiere variable de precio/valor")
            elif any(keyword in col_lower for keyword in keywords_cantidad):
                puntaje += 30
                razones.append("✅ Nombre sugiere variable cuantitativa")
            
            # 2. Variabilidad (variables con buena variación son mejores para predecir)
            if 10 < cv < 100:
                puntaje += 30
                razones.append(f"✅ Buena variabilidad (CV: {cv:.1f}%)")
            elif cv >= 100:
                puntaje += 20
                razones.append(f"⚠️ Alta variabilidad (CV: {cv:.1f}%)")
            else:
                puntaje += 5
                razones.append(f"⚠️ Baja variabilidad (CV: {cv:.1f}%)")
            
            # 3. Número de valores únicos (continuas son mejores)
            if porcentaje_unicos > 50:
                puntaje += 30
                razones.append(f"✅ Variable continua ({valores_unicos} valores únicos)")
            elif porcentaje_unicos > 20:
                puntaje += 20
                razones.append(f"⚠️ Variable semi-continua ({valores_unicos} valores únicos)")
            else:
                puntaje += 5
                razones.append(f"⚠️ Pocos valores únicos ({valores_unicos})")
            
            # 4. Magnitud de valores (precios suelen ser > 100)
            if media > 100:
                puntaje += 20
                razones.append(f"✅ Valores en escala apropiada (media: {media:,.2f})")
            elif media > 10:
                puntaje += 10
                razones.append(f"⚠️ Valores moderados (media: {media:,.2f})")
            
            # 5. Sin valores negativos (precios no pueden ser negativos)
            if datos_col.min() >= 0:
                puntaje += 10
                razones.append("✅ Sin valores negativos")
            else:
                razones.append("⚠️ Contiene valores negativos")
            
            # Generar consejos inteligentes para esta variable
            consejos = self.generar_consejos_variable(col, datos_col, {
                'media': media,
                'std': std,
                'cv': cv,
                'min': datos_col.min(),
                'max': datos_col.max()
            })
            
            analisis[col] = {
                'puntaje': puntaje,
                'razones': razones,
                'media': media,
                'std': std,
                'cv': cv,
                'min': datos_col.min(),
                'max': datos_col.max(),
                'valores_unicos': valores_unicos,
                'porcentaje_unicos': porcentaje_unicos,
                'consejos': consejos  # NUEVO: Consejos inteligentes
            }
        
        # Encontrar la mejor variable
        mejor_variable = max(analisis.items(), key=lambda x: x[1]['puntaje'])
        
        return mejor_variable[0], analisis
    
    def mostrar_info_variable(self, event=None):
        """Muestra información completa sobre la variable seleccionada"""
        variable = self.combo_objetivo.get()
        
        if not variable or variable not in self.recomendaciones:
            return
        
        info = self.recomendaciones[variable]
        
        # Limpiar y mostrar información completa en el panel de resultados
        self.txt_resultados.config(state='normal')
        self.txt_resultados.delete('1.0', 'end')
        
        # Título
        self.txt_resultados.insert('end', "="*80 + "\n")
        self.txt_resultados.insert('end', f"🔍 ANÁLISIS DETALLADO: {variable}\n")
        self.txt_resultados.insert('end', "="*80 + "\n\n")
        
        # Calificación
        if info['puntaje'] >= 80:
            emoji = "🌟"
            calificacion = "EXCELENTE"
            color_bg = '#e8f5e9'
            color_fg = '#2e7d32'
        elif info['puntaje'] >= 60:
            emoji = "✅"
            calificacion = "BUENA"
            color_bg = '#e3f2fd'
            color_fg = '#1565c0'
        elif info['puntaje'] >= 40:
            emoji = "⚠️"
            calificacion = "ACEPTABLE"
            color_bg = '#fff3e0'
            color_fg = '#e65100'
        else:
            emoji = "❌"
            calificacion = "NO RECOMENDADA"
            color_bg = '#ffebee'
            color_fg = '#c62828'
        
        self.txt_resultados.insert('end', f"{emoji} CALIFICACIÓN: {calificacion}\n")
        self.txt_resultados.insert('end', f"Puntaje de Aptitud: {info['puntaje']}/150\n\n")
        
        # Estadísticas
        self.txt_resultados.insert('end', "📊 ESTADÍSTICAS:\n")
        self.txt_resultados.insert('end', f"   • Promedio: {info['media']:,.2f}\n")
        self.txt_resultados.insert('end', f"   • Desviación Estándar: {info['std']:,.2f}\n")
        self.txt_resultados.insert('end', f"   • Coeficiente de Variación: {info['cv']:.2f}%\n")
        self.txt_resultados.insert('end', f"   • Mínimo: {info['min']:,.2f}\n")
        self.txt_resultados.insert('end', f"   • Máximo: {info['max']:,.2f}\n")
        self.txt_resultados.insert('end', f"   • Valores Únicos: {info['valores_unicos']}\n")
        self.txt_resultados.insert('end', f"   • Rango: {info['max'] - info['min']:,.2f}\n\n")
        
        # Razones de aptitud
        self.txt_resultados.insert('end', "💡 RAZONES DE LA CALIFICACIÓN:\n")
        for i, razon in enumerate(info['razones'], 1):
            self.txt_resultados.insert('end', f"   {i}. {razon}\n")
        
        # CONSEJOS COMPLETOS
        if info['consejos']:
            self.txt_resultados.insert('end', "\n" + "="*80 + "\n")
            self.txt_resultados.insert('end', f"💡 CONSEJOS Y RECOMENDACIONES PARA '{variable}'\n")
            self.txt_resultados.insert('end', "="*80 + "\n\n")
            
            for i, consejo in enumerate(info['consejos'], 1):
                self.txt_resultados.insert('end', f"{i}. {consejo}\n\n")
            
            self.txt_resultados.insert('end', "="*80 + "\n")
        
        # Instrucciones
        self.txt_resultados.insert('end', "\n📌 SIGUIENTE PASO:\n")
        self.txt_resultados.insert('end', "Si esta variable te parece adecuada, presiona 'ENTRENAR MODELO'.\n")
        self.txt_resultados.insert('end', "Si quieres ver otra variable, selecciónala del menú desplegable.\n")
        
        self.txt_resultados.config(state='disabled')
        
        # Actualizar el mini-resumen en el panel de configuración
        mensaje_corto = f"{emoji} {calificacion}\n"
        mensaje_corto += f"Puntaje: {info['puntaje']}/150\n\n"
        mensaje_corto += "Consejo principal:\n"
        if info['consejos']:
            mensaje_corto += f"• {info['consejos'][0][:120]}"
        
        self.lbl_recomendacion.config(text=mensaje_corto, bg=color_bg, fg=color_fg)
        self.frame_recomendacion.config(bg=color_bg)
        
    def cargar_archivo(self):
        """Cargar archivo CSV o Excel"""
        ruta = filedialog.askopenfilename(
            title="Seleccionar archivo",
            filetypes=[
                ("Archivos CSV", "*.csv"),
                ("Archivos Excel", "*.xlsx *.xls"),
                ("Todos los archivos", "*.*")
            ]
        )
        
        if not ruta:
            return
            
        try:
            # Limpiar resultados anteriores
            self.txt_resultados.config(state='normal')
            self.txt_resultados.delete('1.0', 'end')
            self.txt_resultados.config(state='disabled')
            
            self.log("="*80)
            self.log("📂 CARGANDO Y ANALIZANDO ARCHIVO...")
            self.log("="*80)
            
            self.ruta_archivo = ruta
            nombre_archivo = os.path.basename(ruta)
            
            # Cargar según extensión
            if ruta.endswith('.csv'):
                self.df = pd.read_csv(ruta)
                self.log(f"✅ Archivo CSV cargado: {nombre_archivo}")
            elif ruta.endswith(('.xlsx', '.xls')):
                self.df = pd.read_excel(ruta)
                self.log(f"✅ Archivo Excel cargado: {nombre_archivo}")
            
            # Mostrar info
            self.lbl_archivo.config(text=f"📄 {nombre_archivo}")
            info = f"📊 {len(self.df)} filas × {len(self.df.columns)} columnas"
            self.lbl_info_dataset.config(text=info)
            
            self.log(f"\n📐 Dimensiones: {len(self.df)} filas × {len(self.df.columns)} columnas")
            self.log(f"\n🔤 Columnas encontradas:")
            for i, col in enumerate(self.df.columns, 1):
                tipo = str(self.df[col].dtype)
                self.log(f"   {i}. {col} ({tipo})")
            
            # ANÁLISIS INTELIGENTE
            self.log("\n" + "="*80)
            self.log("🤖 ANÁLISIS INTELIGENTE DE VARIABLES")
            self.log("="*80)
            
            mejor_variable, analisis = self.analizar_variables(self.df)
            self.recomendaciones = analisis
            
            if mejor_variable:
                self.log(f"\n🌟 VARIABLE RECOMENDADA: {mejor_variable}")
                self.log(f"   Puntaje: {analisis[mejor_variable]['puntaje']}/150")
                self.log(f"\n   Razones:")
                for razon in analisis[mejor_variable]['razones']:
                    self.log(f"   {razon}")
                
                self.log(f"\n   Estadísticas:")
                self.log(f"   • Media: {analisis[mejor_variable]['media']:,.2f}")
                self.log(f"   • Rango: {analisis[mejor_variable]['min']:,.2f} - {analisis[mejor_variable]['max']:,.2f}")
                self.log(f"   • Valores únicos: {analisis[mejor_variable]['valores_unicos']}")
                
                # NUEVO: Mostrar consejos inteligentes
                self.log(f"\n💡 CONSEJOS Y RECOMENDACIONES PARA '{mejor_variable}':")
                self.log("="*80)
                for i, consejo in enumerate(analisis[mejor_variable]['consejos'], 1):
                    self.log(f"{i}. {consejo}")
                self.log("="*80)
                
                # Mostrar ranking de top 3
                top_3 = sorted(analisis.items(), key=lambda x: x[1]['puntaje'], reverse=True)[:3]
                self.log(f"\n📊 TOP 3 VARIABLES PARA PREDECIR:")
                for i, (var, info) in enumerate(top_3, 1):
                    emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
                    self.log(f"   {emoji} {i}. {var} - Puntaje: {info['puntaje']}/150")
                    # Mostrar primer consejo de cada variable top
                    if info['consejos']:
                        self.log(f"      💭 {info['consejos'][0][:80]}...")
            
            # Actualizar combo de columnas
            columnas_numericas = self.df.select_dtypes(include=[np.number]).columns.tolist()
            self.combo_objetivo['values'] = columnas_numericas
            self.combo_objetivo['state'] = 'readonly'
            
            if columnas_numericas and mejor_variable:
                # Seleccionar automáticamente la mejor variable
                idx = columnas_numericas.index(mejor_variable)
                self.combo_objetivo.current(idx)
                self.mostrar_info_variable()
                self.btn_entrenar['state'] = 'normal'
                self.lbl_estado.config(text="✅ Listo para entrenar", fg='#27ae60')
                
            self.log(f"\n✅ Dataset analizado correctamente")
            self.log(f"📌 Revisa la recomendación y presiona 'ENTRENAR MODELO'")
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el archivo:\n{str(e)}")
            self.log(f"❌ ERROR: {str(e)}")
            
    def entrenar_modelo(self):
        """Entrenar el modelo en un hilo separado"""
        if self.df is None:
            messagebox.showwarning("Advertencia", "Primero carga un archivo")
            return
            
        self.columna_objetivo = self.combo_objetivo.get()
        if not self.columna_objetivo:
            messagebox.showwarning("Advertencia", "Selecciona una variable objetivo")
            return
        
        # Deshabilitar controles
        self.btn_entrenar['state'] = 'disabled'
        self.btn_cargar['state'] = 'disabled'
        self.combo_objetivo['state'] = 'disabled'
        self.progress.start()
        self.lbl_estado.config(text="🔄 Entrenando modelo...", fg='#f39c12')
        
        # Ejecutar en hilo separado
        hilo = threading.Thread(target=self.proceso_entrenamiento)
        hilo.daemon = True
        hilo.start()
        
    def proceso_entrenamiento(self):
        """Proceso completo de entrenamiento"""
        try:
            self.txt_resultados.config(state='normal')
            self.txt_resultados.delete('1.0', 'end')
            self.txt_resultados.config(state='disabled')
            
            self.log("="*80)
            self.log("🚀 INICIANDO ENTRENAMIENTO DEL MODELO")
            self.log("="*80)
            
            # Configuración
            tamano_prueba = self.slider_prueba.get() / 100
            epocas = int(self.spin_epocas.get())
            
            self.log(f"\n⚙️ CONFIGURACIÓN:")
            self.log(f"   Variable objetivo: {self.columna_objetivo}")
            if self.columna_objetivo in self.recomendaciones:
                puntaje = self.recomendaciones[self.columna_objetivo]['puntaje']
                self.log(f"   Aptitud de la variable: {puntaje}/150")
            self.log(f"   Datos entrenamiento: {int((1-tamano_prueba)*100)}%")
            self.log(f"   Datos prueba: {int(tamano_prueba*100)}%")
            self.log(f"   Épocas: {epocas}")
            
            # Preparar datos
            self.log("\n" + "="*80)
            self.log("🔧 PREPARANDO DATOS")
            self.log("="*80)
            
            df = self.df.copy()
            
            # Eliminar valores nulos
            if df.isnull().sum().sum() > 0:
                antes = len(df)
                df = df.dropna()
                self.log(f"⚠️  Eliminadas {antes - len(df)} filas con valores nulos")
            
            # Seleccionar características numéricas
            columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
            columnas_numericas = [col for col in columnas_numericas if col != self.columna_objetivo]
            
            self.log(f"\n✅ Características seleccionadas ({len(columnas_numericas)}):")
            for i, col in enumerate(columnas_numericas, 1):
                self.log(f"   {i}. {col}")
            
            X = df[columnas_numericas]
            y = df[self.columna_objetivo]
            
            # Dividir datos
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=tamano_prueba, random_state=42
            )
            
            self.log(f"\n📊 División completada:")
            self.log(f"   Entrenamiento: {len(X_train)} muestras")
            self.log(f"   Prueba: {len(X_test)} muestras")
            
            # Normalizar
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.log(f"\n✅ Datos normalizados (StandardScaler)")
            
            # Construir modelo
            self.log("\n" + "="*80)
            self.log("🧠 CONSTRUYENDO RED NEURONAL")
            self.log("="*80)
            
            n_features = X_train_scaled.shape[1]
            n_layer1 = max(64, n_features * 8)
            n_layer2 = max(32, n_features * 4)
            n_layer3 = max(16, n_features * 2)
            
            self.log(f"\n🏗️  Arquitectura:")
            self.log(f"   Entrada: {n_features} neuronas")
            self.log(f"   Capa 1: {n_layer1} neuronas + ReLU + Dropout(20%)")
            self.log(f"   Capa 2: {n_layer2} neuronas + ReLU + Dropout(20%)")
            self.log(f"   Capa 3: {n_layer3} neuronas + ReLU + Dropout(10%)")
            self.log(f"   Salida: 1 neurona (Linear)")
            
            self.modelo = Sequential([
                Dense(n_layer1, input_dim=n_features, activation='relu'),
                Dropout(0.2),
                Dense(n_layer2, activation='relu'),
                Dropout(0.2),
                Dense(n_layer3, activation='relu'),
                Dropout(0.1),
                Dense(1, activation='linear')
            ])
            
            self.modelo.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
            
            self.log(f"\n📊 Total de parámetros: {self.modelo.count_params():,}")
            
            # Entrenar
            self.log("\n" + "="*80)
            self.log(f"🎯 ENTRENANDO ({epocas} épocas)...")
            self.log("="*80)
            self.log("\n⏳ Por favor espera, esto puede tomar unos minutos...\n")
            
            early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0)
            
            history = self.modelo.fit(
                X_train_scaled, y_train,
                epochs=epocas,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stop],
                verbose=0
            )
            
            self.log(f"✅ Entrenamiento completado en {len(history.history['loss'])} épocas")
            
            # Evaluar
            self.log("\n" + "="*80)
            self.log("📈 EVALUANDO MODELO")
            self.log("="*80)
            
            y_pred_train = self.modelo.predict(X_train_scaled, verbose=0)
            y_pred_test = self.modelo.predict(X_test_scaled, verbose=0)
            
            # Métricas
            r2_train = r2_score(y_train, y_pred_train)
            rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
            mae_train = mean_absolute_error(y_train, y_pred_train)
            
            r2_test = r2_score(y_test, y_pred_test)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            mae_test = mean_absolute_error(y_test, y_pred_test)
            
            # Errores relativos
            promedio = y.mean()
            error_rel_rmse = (rmse_test / promedio) * 100
            error_rel_mae = (mae_test / promedio) * 100
            
            self.log("\n" + "="*80)
            self.log("📊 RESULTADOS FINALES")
            self.log("="*80)
            
            self.log(f"\n🎓 CONJUNTO DE ENTRENAMIENTO:")
            self.log(f"   R² Score:  {r2_train:.4f} ({r2_train*100:.2f}%)")
            self.log(f"   RMSE:      {rmse_train:,.4f}")
            self.log(f"   MAE:       {mae_train:,.4f}")
            
            self.log(f"\n🧪 CONJUNTO DE PRUEBA:")
            self.log(f"   R² Score:  {r2_test:.4f} ({r2_test*100:.2f}%)")
            self.log(f"   RMSE:      {rmse_test:,.4f}")
            self.log(f"   MAE:       {mae_test:,.4f}")
            
            self.log(f"\n📊 ERRORES RELATIVOS:")
            self.log(f"   Promedio {self.columna_objetivo}: {promedio:,.2f}")
            self.log(f"   Error relativo RMSE: {error_rel_rmse:.2f}%")
            self.log(f"   Error relativo MAE: {error_rel_mae:.2f}%")
            
            self.log("\n💡 INTERPRETACIÓN:")
            if r2_test > 0.85:
                self.log("   ✅ Rendimiento EXCELENTE")
            elif r2_test > 0.70:
                self.log("   ✅ Rendimiento BUENO")
            elif r2_test > 0.50:
                self.log("   ⚠️  Rendimiento ACEPTABLE")
            else:
                self.log("   ❌ Rendimiento MEJORABLE - Considera más datos o ajustar el modelo")
            
            diferencia_r2 = abs(r2_train - r2_test)
            if diferencia_r2 < 0.05:
                self.log("   ✅ Buena generalización (no hay overfitting)")
            elif diferencia_r2 < 0.10:
                self.log("   ⚠️  Ligero overfitting detectado")
            else:
                self.log("   ❌ Overfitting significativo - El modelo memoriza en lugar de aprender")
            
            # Análisis de la elección de variable
            self.log("\n" + "="*80)
            self.log("🎯 ANÁLISIS DE LA VARIABLE ELEGIDA")
            self.log("="*80)
            
            if self.columna_objetivo in self.recomendaciones:
                info_var = self.recomendaciones[self.columna_objetivo]
                puntaje_var = info_var['puntaje']
                
                self.log(f"\nVariable elegida: {self.columna_objetivo}")
                self.log(f"Puntaje de aptitud: {puntaje_var}/150")
                
                # Correlación entre aptitud y rendimiento
                if r2_test > 0.70 and puntaje_var >= 80:
                    self.log("\n✅ EXCELENTE ELECCIÓN:")
                    self.log("   La variable tenía alta aptitud y el modelo tiene buen rendimiento.")
                    self.log("   La recomendación fue acertada.")
                elif r2_test > 0.70 and puntaje_var < 80:
                    self.log("\n🎉 SORPRESA POSITIVA:")
                    self.log("   Aunque la variable no tenía la mayor aptitud, el modelo funciona bien.")
                    self.log("   Las relaciones en los datos son muy buenas.")
                elif r2_test < 0.70 and puntaje_var >= 80:
                    self.log("\n⚠️ RESULTADO INESPERADO:")
                    self.log("   La variable tenía buena aptitud pero el rendimiento es moderado.")
                    self.log("   Posibles causas: datos insuficientes, ruido, relaciones no lineales complejas.")
                else:
                    self.log("\n⚠️ VARIABLE SUBÓPTIMA:")
                    self.log("   La variable elegida no era la mejor opción.")
                    self.log("   Considera probar con la variable recomendada.")
                
                # NUEVO: Mostrar TODOS los consejos después del entrenamiento
                self.log("\n" + "="*80)
                self.log(f"💡 CONSEJOS INTELIGENTES BASADOS EN LOS RESULTADOS")
                self.log("="*80)
                
                self.log(f"\n🔍 Análisis de '{self.columna_objetivo}':")
                self.log(f"   Promedio: {promedio:,.2f}")
                self.log(f"   Error del modelo: {rmse_test:,.2f} ({error_rel_rmse:.2f}%)")
                self.log(f"   Precisión (R²): {r2_test:.4f}")
                
                self.log(f"\n📋 RECOMENDACIONES DE ACCIÓN:")
                if info_var['consejos']:
                    for i, consejo in enumerate(info_var['consejos'], 1):
                        self.log(f"\n{i}. {consejo}")
                
                # Consejos adicionales basados en el rendimiento del modelo
                self.log(f"\n🎯 CONSEJOS BASADOS EN EL MODELO ENTRENADO:")
                
                if r2_test > 0.85:
                    self.log("\n✅ MODELO ALTAMENTE CONFIABLE:")
                    self.log(f"   • Puedes confiar en las predicciones con {r2_test*100:.1f}% de certeza")
                    self.log(f"   • Usa este modelo para tomar decisiones importantes")
                    self.log(f"   • El margen de error promedio es de solo ${rmse_test:,.0f}")
                elif r2_test > 0.70:
                    self.log("\n✅ MODELO CONFIABLE:")
                    self.log(f"   • Las predicciones son buenas ({r2_test*100:.1f}% de precisión)")
                    self.log(f"   • Considera el margen de error de ${rmse_test:,.0f} en tus decisiones")
                    self.log(f"   • Complementa con análisis adicional para decisiones críticas")
                else:
                    self.log("\n⚠️ MODELO CON LIMITACIONES:")
                    self.log(f"   • La precisión es moderada ({r2_test*100:.1f}%)")
                    self.log(f"   • Usa las predicciones como referencia, no como valor exacto")
                    self.log(f"   • Consulta con expertos antes de tomar decisiones importantes")
                
                # Análisis del error relativo
                if error_rel_rmse < 10:
                    self.log(f"\n🎯 PREDICCIONES MUY PRECISAS:")
                    self.log(f"   • Error relativo de solo {error_rel_rmse:.1f}%")
                    self.log(f"   • En promedio, el modelo se equivoca menos del 10%")
                    self.log(f"   • Excelente para planificación y toma de decisiones")
                elif error_rel_rmse < 20:
                    self.log(f"\n✅ PREDICCIONES CONFIABLES:")
                    self.log(f"   • Error relativo del {error_rel_rmse:.1f}%")
                    self.log(f"   • Buen nivel de precisión para la mayoría de aplicaciones")
                    self.log(f"   • Ajusta tus expectativas considerando este margen")
                else:
                    self.log(f"\n⚠️ MARGEN DE ERROR CONSIDERABLE:")
                    self.log(f"   • Error relativo del {error_rel_rmse:.1f}%")
                    self.log(f"   • Las predicciones tienen alta variabilidad")
                    self.log(f"   • Usa rangos en lugar de valores exactos")
                
                # Consejos de negocio
                self.log(f"\n💼 APLICACIÓN PRÁCTICA:")
                if any(k in self.columna_objetivo.lower() for k in ['price', 'precio', 'value', 'valor']):
                    valores_test = y_test.values
                    pred_test = y_pred_test.flatten()
                    
                    # Detectar si el modelo predice alto o bajo
                    sesgo = np.mean(pred_test - valores_test)
                    
                    if abs(sesgo) < rmse_test * 0.2:
                        self.log(f"   • El modelo es neutral (sesgo: ${sesgo:,.0f})")
                        self.log(f"   • Las predicciones no favorecen ni compradores ni vendedores")
                    elif sesgo > 0:
                        self.log(f"   • El modelo tiende a sobre-estimar (sesgo: +${sesgo:,.0f})")
                        self.log(f"   • COMPRADORES: Negocien por debajo de la predicción")
                        self.log(f"   • VENDEDORES: Pueden pedir cerca de la predicción")
                    else:
                        self.log(f"   • El modelo tiende a sub-estimar (sesgo: ${sesgo:,.0f})")
                        self.log(f"   • COMPRADORES: Buen momento, el modelo predice por debajo del real")
                        self.log(f"   • VENDEDORES: Pidan más que la predicción del modelo")
                
                self.log("\n" + "="*80)
            
            # Comparación con otras variables
            top_3 = sorted(self.recomendaciones.items(), key=lambda x: x[1]['puntaje'], reverse=True)[:3]
            self.log(f"\n💡 SUGERENCIA:")
            if self.columna_objetivo != top_3[0][0]:
                self.log(f"   Considera entrenar también con: {top_3[0][0]}")
                self.log(f"   (Puntaje: {top_3[0][1]['puntaje']}/150)")
            else:
                self.log(f"   ✅ Ya elegiste la mejor variable según el análisis")
            
            # Guardar resultados
            self.resultados = {
                'r2_train': r2_train,
                'r2_test': r2_test,
                'rmse_test': rmse_test,
                'mae_test': mae_test,
                'error_rel_rmse': error_rel_rmse,
                'error_rel_mae': error_rel_mae,
                'promedio': promedio,
                'history': history,
                'y_test': y_test,
                'y_pred_test': y_pred_test,
                'columnas': columnas_numericas
            }
            
            # Generar gráficas
            self.generar_graficas(history, y_test, y_pred_test)
            
            # Guardar modelo
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.modelo.save(f'modelo_{self.columna_objetivo}_{timestamp}.h5')
            
            import pickle
            with open(f'scaler_{self.columna_objetivo}_{timestamp}.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
            
            info_modelo = {
                'columnas': columnas_numericas,
                'columna_objetivo': self.columna_objetivo,
                'r2_test': r2_test,
                'rmse_test': rmse_test,
                'aptitud_variable': self.recomendaciones.get(self.columna_objetivo, {}).get('puntaje', 0)
            }
            with open(f'info_modelo_{self.columna_objetivo}_{timestamp}.pkl', 'wb') as f:
                pickle.dump(info_modelo, f)
            
            self.log(f"\n💾 ARCHIVOS GUARDADOS:")
            self.log(f"   ✅ modelo_{self.columna_objetivo}_{timestamp}.h5")
            self.log(f"   ✅ scaler_{self.columna_objetivo}_{timestamp}.pkl")
            self.log(f"   ✅ info_modelo_{self.columna_objetivo}_{timestamp}.pkl")
            self.log(f"   ✅ resultados_{self.columna_objetivo}_{timestamp}.png")
            
            # Ejemplos de predicción
            self.log("\n" + "="*80)
            self.log("🔮 EJEMPLOS DE PREDICCIÓN")
            self.log("="*80)
            
            self.log("\nPrimeras 10 predicciones del conjunto de prueba:")
            self.log("-"*80)
            
            comparacion = pd.DataFrame({
                'Real': y_test.values[:10],
                'Predicho': y_pred_test.flatten()[:10],
                'Error': y_test.values[:10] - y_pred_test.flatten()[:10],
                'Error %': ((y_test.values[:10] - y_pred_test.flatten()[:10]) / y_test.values[:10] * 100)
            })
            
            for idx in range(len(comparacion)):
                row = comparacion.iloc[idx]
                self.log(f"  #{idx+1:2d} | Real: {row['Real']:10,.2f} | "
                        f"Predicho: {row['Predicho']:10,.2f} | "
                        f"Error: {row['Error %']:6.2f}%")
            
            self.log("\n" + "="*80)
            self.log("🎉 PROCESO COMPLETADO EXITOSAMENTE")
            self.log("="*80)
            
            self.log("\n📋 RESUMEN:")
            self.log(f"   • Variable objetivo: {self.columna_objetivo}")
            self.log(f"   • Aptitud de variable: {info_modelo['aptitud_variable']}/150")
            self.log(f"   • Precisión del modelo (R²): {r2_test:.4f}")
            self.log(f"   • Error promedio (RMSE): {rmse_test:,.2f}")
            self.log(f"   • Error relativo: {error_rel_rmse:.2f}%")
            
            # Habilitar botones
            self.root.after(0, self.finalizar_entrenamiento, True)
            
        except Exception as e:
            self.log(f"\n❌ ERROR: {str(e)}")
            import traceback
            self.log(f"\n{traceback.format_exc()}")
            self.root.after(0, self.finalizar_entrenamiento, False)
            
    def generar_graficas(self, history, y_test, y_pred):
        """Generar y guardar gráficas"""
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 5))
        
        # Gráfica 1
        plt.subplot(1, 4, 1)
        plt.plot(history.history['loss'], label='Entrenamiento', linewidth=2)
        plt.plot(history.history['val_loss'], label='Validación', linewidth=2)
        plt.title('Pérdida Durante el Entrenamiento', fontweight='bold')
        plt.xlabel('Época')
        plt.ylabel('MSE')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Gráfica 2
        plt.subplot(1, 4, 2)
        plt.plot(history.history['mae'], label='Entrenamiento', linewidth=2)
        plt.plot(history.history['val_mae'], label='Validación', linewidth=2)
        plt.title('MAE Durante el Entrenamiento', fontweight='bold')
        plt.xlabel('Época')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Gráfica 3
        plt.subplot(1, 4, 3)
        plt.scatter(y_test, y_pred, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3)
        plt.xlabel('Valores Reales')
        plt.ylabel('Valores Predichos')
        plt.title('Predicciones vs Valores Reales', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Gráfica 4
        plt.subplot(1, 4, 4)
        errores = y_test.values - y_pred.flatten()
        plt.hist(errores, bins=30, edgecolor='black', alpha=0.7)
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
        plt.xlabel('Error de Predicción')
        plt.ylabel('Frecuencia')
        plt.title('Distribución de Errores', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'resultados_{self.columna_objetivo}_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def finalizar_entrenamiento(self, exito):
        """Finalizar proceso de entrenamiento"""
        self.progress.stop()
        
        if exito:
            self.lbl_estado.config(text="✅ Modelo entrenado exitosamente", fg='#27ae60')
            self.btn_guardar['state'] = 'normal'
            self.btn_graficas['state'] = 'normal'
        else:
            self.lbl_estado.config(text="❌ Error en el entrenamiento", fg='#e74c3c')
        
        self.btn_entrenar['state'] = 'normal'
        self.btn_cargar['state'] = 'normal'
        self.combo_objetivo['state'] = 'readonly'
        
    def guardar_resultados(self):
        """Guardar reporte de resultados"""
        if not self.resultados:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f'reporte_{self.columna_objetivo}_{timestamp}.txt'
        
        with open(nombre_archivo, 'w', encoding='utf-8') as f:
            contenido = self.txt_resultados.get('1.0', 'end')
            f.write(contenido)
        
        messagebox.showinfo("Guardado", f"Reporte guardado como:\n{nombre_archivo}")
        
    def mostrar_graficas(self):
        """Mostrar las gráficas generadas"""
        import glob
        archivos_graficas = glob.glob(f'resultados_{self.columna_objetivo}_*.png')
        
        if not archivos_graficas:
            archivos_graficas = glob.glob('resultados_*.png')
        
        if not archivos_graficas:
            messagebox.showwarning("Advertencia", "No se encontraron gráficas")
            return
        
        # Abrir la última gráfica generada
        ultima_grafica = max(archivos_graficas, key=os.path.getctime)
        
        try:
            if os.name == 'nt':  # Windows
                os.startfile(ultima_grafica)
            elif os.name == 'posix':  # macOS y Linux
                os.system(f'open {ultima_grafica}')
        except:
            messagebox.showinfo("Gráficas", f"Gráfica guardada en:\n{ultima_grafica}")

# ============================================================================
# EJECUTAR APLICACIÓN
# ============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = ModeloPredictivo(root)
    root.mainloop()