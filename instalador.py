"""
============================================================================
INSTALADOR AUTOMÁTICO - INTERFAZ GRÁFICA MODELO PREDICTIVO
Ejecuta este archivo primero para verificar e instalar todo lo necesario
============================================================================
"""

import subprocess
import sys
import os

def verificar_e_instalar():
    """Verificar y instalar librerías necesarias"""
    
    print("="*80)
    print("🔧 VERIFICANDO E INSTALANDO DEPENDENCIAS")
    print("="*80)
    
    librerias = [
        'pandas',
        'numpy',
        'tensorflow',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'openpyxl'
    ]
    
    for libreria in librerias:
        print(f"\n📦 Verificando {libreria}...", end=" ")
        try:
            __import__(libreria.replace('-', '_'))
            print("✅ Instalada")
        except ImportError:
            print("❌ No instalada")
            print(f"   Instalando {libreria}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", libreria])
            print(f"   ✅ {libreria} instalada correctamente")
    
    print("\n" + "="*80)
    print("✅ TODAS LAS DEPENDENCIAS ESTÁN INSTALADAS")
    print("="*80)
    print("\n🚀 Ahora puedes ejecutar: python interfaz_modelo_predictivo.py")
    print("\nPresiona Enter para continuar...")
    input()

if __name__ == "__main__":
    verificar_e_instalar()