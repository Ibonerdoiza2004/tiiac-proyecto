"""Aplicativo de la E2: la API FastAPI del dashboard, que también sirve el frontend.

Se arranca con ``python -m src.app``. La API no importa ``src/training``: lo que
entrena, juega o evalúa corre en procesos trabajadores aparte, que importan el
entrenamiento dentro del proceso hijo.
"""
