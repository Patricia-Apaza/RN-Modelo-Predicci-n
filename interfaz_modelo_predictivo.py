"""
============================================================================
INTERFAZ GRÁFICA - MODELO PREDICTIVO CON RED NEURONAL
Curso: Análisis Multivariado
Descripción: Sube tu dataset y entrena automáticamente
============================================================================
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
import threading
import os
from datetime import datetime

# Importaciones de ML (se cargarán después)
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
            text="Sube tu dataset y entrena automáticamente tu modelo predictivo",
            font=('Arial', 10),
            fg='#ecf0f1',
            bg='#2c3e50'
        )
        subtitulo.pack()
        
        # ===== CONTENEDOR PRINCIPAL =====
        contenedor = tk.Frame(self.root, bg='#f0f0f0')
        contenedor.pack(fill='both', expand=True, padx=20, pady=20)
        
        # ===== PANEL IZQUIERDO: CONTROLES =====
        panel_izq = tk.Frame(contenedor, bg='white', relief='raised', bd=2)
        panel_izq.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        # PASO 1: Cargar Dataset
        self.crear_seccion_carga(panel_izq)
        
        # PASO 2: Configuración
        self.crear_seccion_configuracion(panel_izq)
        
        # PASO 3: Entrenar
        self.crear_seccion_entrenamiento(panel_izq)
        
        # ===== PANEL DERECHO: RESULTADOS =====
        panel_der = tk.Frame(contenedor, bg='white', relief='raised', bd=2)
        panel_der.pack(side='right', fill='both', expand=True)
        
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
            "🎓 Bienvenido al Sistema de Predicción\n\n"
            "📌 INSTRUCCIONES:\n"
            "1. Carga tu archivo CSV o Excel\n"
            "2. Selecciona la variable que quieres predecir\n"
            "3. Ajusta la configuración si lo deseas\n"
            "4. Presiona 'ENTRENAR MODELO'\n"
            "5. Espera los resultados\n\n"
            "✨ Los resultados aparecerán aquí automáticamente\n"
            "📁 Se guardarán archivos con el modelo y gráficas\n"
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
            self.log("📂 CARGANDO ARCHIVO...")
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
            
            # Actualizar combo de columnas
            columnas_numericas = self.df.select_dtypes(include=[np.number]).columns.tolist()
            self.combo_objetivo['values'] = columnas_numericas
            self.combo_objetivo['state'] = 'readonly'
            
            if columnas_numericas:
                self.combo_objetivo.current(0)
                self.btn_entrenar['state'] = 'normal'
                self.lbl_estado.config(text="✅ Listo para entrenar", fg='#27ae60')
                
            self.log(f"\n✅ Dataset cargado correctamente")
            self.log(f"📌 Ahora selecciona la variable a predecir y presiona 'ENTRENAR MODELO'")
            
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
                self.log("   ❌ Rendimiento MEJORABLE")
            
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
            self.modelo.save(f'modelo_{timestamp}.h5')
            
            import pickle
            with open(f'scaler_{timestamp}.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
            
            self.log(f"\n💾 ARCHIVOS GUARDADOS:")
            self.log(f"   ✅ modelo_{timestamp}.h5")
            self.log(f"   ✅ scaler_{timestamp}.pkl")
            self.log(f"   ✅ resultados_{timestamp}.png")
            
            self.log("\n" + "="*80)
            self.log("🎉 PROCESO COMPLETADO EXITOSAMENTE")
            self.log("="*80)
            
            # Habilitar botones
            self.root.after(0, self.finalizar_entrenamiento, True)
            
        except Exception as e:
            self.log(f"\n❌ ERROR: {str(e)}")
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
        plt.savefig(f'resultados_{timestamp}.png', dpi=300, bbox_inches='tight')
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
        self.combo_objetivo['state'] = 'readonly'
        
    def guardar_resultados(self):
        """Guardar reporte de resultados"""
        if not self.resultados:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f'reporte_{timestamp}.txt'
        
        with open(nombre_archivo, 'w', encoding='utf-8') as f:
            contenido = self.txt_resultados.get('1.0', 'end')
            f.write(contenido)
        
        messagebox.showinfo("Guardado", f"Reporte guardado como:\n{nombre_archivo}")
        
    def mostrar_graficas(self):
        """Mostrar las gráficas generadas"""
        import glob
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