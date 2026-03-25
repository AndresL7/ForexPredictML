# Forex Predict ML

Proyecto de prediccion direccional para mercados financieros (EURUSD, COPUSD, NASDAQ) usando XGBoost y analisis en notebook.

## Estructura

- `notebooks/ForexPredictML.ipynb`: notebook principal con experimentos.
- `src/forex_predict_clean.py`: version en script del pipeline (descarga, features, entrenamiento y evaluacion).
- `docs/project_notes.md`: notas y recomendaciones para evolucion del proyecto.
- `requirements.txt`: dependencias Python.

## Requisitos

- Python 3.10+
- pip

## Instalacion

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso rapido

### 1) Ejecutar notebook

```bash
jupyter notebook notebooks/ForexPredictML.ipynb
```

### 2) Ejecutar pipeline limpio

```bash
python src/forex_predict_clean.py
```

## Reproducibilidad

- El notebook ya incluye una version de tuning temporal (train/valid/test) para evitar fuga de informacion en test.
- El pipeline de `src/forex_predict_clean.py` fija `random_state` para resultados mas consistentes.

## Publicar en GitHub

```bash
cd /home/andres/Descargas/forex-predict-ml
git init
git add .
git commit -m "Initial commit: forex prediction project structure"
```

Luego crea un repositorio vacio en GitHub y conecta el remoto:

```bash
git remote add origin <TU_URL_GITHUB>
git branch -M main
git push -u origin main
```
