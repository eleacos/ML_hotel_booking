# Hotel Booking — Predicción de Cancelaciones de Reservas

> **Clasificación binaria para anticipar cancelaciones hoteleras y optimizar la gestión de ocupación.**

---

## 🇪🇸 Español

### Descripción del problema

Las cancelaciones de reservas representan uno de los principales desafíos operativos y económicos de la industria hotelera. Generan pérdidas de ingresos, dificultan la planificación de ocupación y provocan sobrecostes operativos.

El objetivo de este proyecto es desarrollar un **modelo de clasificación binaria** capaz de predecir con antelación si una reserva será cancelada, permitiendo al hotel:

- Ajustar estrategias de **overbooking** de forma controlada.
- Aplicar **políticas de cancelación** diferenciadas según el perfil de riesgo.
- Optimizar la **gestión de precios** y la planificación de la ocupación.

### Dataset utilizado

| Atributo | Detalle |
|----------|---------|
| **Nombre** | Hotel Booking |
| **Origen** | Público — disponible en [Kaggle](https://www.kaggle.com/datasets/mojtaba142/hotel-booking/data) |
| **Periodo** | Reservas entre 2015 y 2017 |
| **Tamaño** | 119.390 registros y 36 variables |
| **Target** | `is_canceled` — 0: no cancelada / 1: cancelada |
| **Desbalance** | ~62% no canceladas / ~37% canceladas |

El dataset incluye variables sobre el tipo de hotel, antelación de la reserva, canal de distribución, historial del cliente, tipo de depósito, precio medio diario (ADR), entre otras.

> Se eliminaron columnas con **data leakage** (`reservation_status`, `reservation_status_date`) y datos personales no predictivos (`name`, `email`, `phone-number`, `credit_card`).

### Solución adoptada

Se ha seguido un enfoque de **clasificación supervisada** estructurado en las siguientes etapas:

1. **Análisis exploratorio (EDA):** estudio de distribuciones, correlaciones, estacionalidad y segmentación de clientes.
2. **Feature Engineering:** transformaciones logarítmicas sobre variables asimétricas (`lead_time`, `adr`, `previous_cancellations`), agrupación geográfica de países y creación de 7 nuevas variables derivadas (`cancel_ratio`, `loyal_customer`, `lead_time_x_adr`, etc.).
3. **Preprocesado mediante Pipeline:** imputación, escalado y codificación One-Hot integrados en un `ColumnTransformer`, ajustado únicamente sobre datos de entrenamiento.
4. **Comparativa de modelos:** Regresión Logística (baseline), Random Forest, Gradient Boosting y KNN.
5. **Optimización:** `GridSearchCV` con validación cruzada estratificada de 5 folds sobre Random Forest, optimizando `balanced_accuracy`.
6. **Modelo final:** Random Forest optimizado, persistido en `src/models/` mediante `joblib`.

### Estructura del repositorio

```
├── src/
│   ├── data_sample/    # Muestra del dataset (máx. 5 MB) para reproducir el código
│   ├── img/            # Imágenes y gráficos generados durante el proyecto
│   ├── models/         # Modelos entrenados guardados en formato joblib
│   │   ├── random_forest_optimized.joblib
│   │   └── logistic_regression_baseline.joblib
│   ├── notebooks/      # Notebooks de exploración, EDA y pruebas de modelado
│   └── utils/          # Módulos y funciones auxiliares reutilizables
├── main.ipynb          # Notebook final: proceso completo, limpio y ejecutable
├── Presentacion.pdf    # Documento soporte de la exposición en vídeo
└── README.md           # Este fichero
```

### Tecnologías utilizadas

| Categoría | Herramientas |
|-----------|-------------|
| **Lenguaje** | Python 3.10+ |
| **Manipulación de datos** | `pandas`, `numpy` |
| **Visualización** | `matplotlib`, `seaborn` |
| **Machine Learning** | `scikit-learn` (Pipeline, ColumnTransformer, GridSearchCV, RandomForestClassifier, LogisticRegression, GradientBoostingClassifier, KNeighborsClassifier) |
| **Persistencia del modelo** | `joblib` |
| **Entorno** | Jupyter Notebook |

### Instrucciones de reproducción

**1. Clonar el repositorio:**
```bash
git clone https://github.com/eleacos/ML_hotel_booking.git
cd hotel-booking-cancelaciones
```

**2. Instalar dependencias:**
```bash
pip install -r requirements.txt
```
O manualmente:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
```

**3. Colocar el dataset:**

Descarga el fichero `hotel_booking.csv` desde [Kaggle](https://www.kaggle.com/datasets/mojtaba142/hotel-booking/data) y colócalo en:
```
src/data_sample/hotel_booking.csv
```

**4. Ejecutar el notebook principal:**
```bash
jupyter notebook main.ipynb
```
Ejecuta todas las celdas en orden (`Kernel > Restart & Run All`). Al finalizar, el modelo quedará guardado en `src/models/`.

### Principales resultados

| Modelo | Balanced Accuracy | ROC-AUC | F1 — Canceladas |
|--------|:-----------------:|:-------:|:---------------:|
| Logistic Regression (baseline) | 0.775 | 0.866 | 0.72 |
| **Random Forest optimizado** | **0.820** | **0.910** | **0.77** |

**Variables más influyentes:** `lead_time`, `lead_time_x_adr`, `adr_log`, `deposit_type`, `cancel_ratio`.

**Conclusiones clave:**
- La **antelación de la reserva** (`lead_time`) es el predictor más relevante: a mayor antelación, mayor probabilidad de cancelación.
- El **tipo de depósito** y el **canal de distribución** son factores clave: las reservas sin reembolso y las realizadas por agencias online presentan mayor tasa de cancelación.
- El modelo optimizado permite anticipar correctamente el **~80% de las cancelaciones reales**, con un equilibrio adecuado entre ambas clases.

---
---

## 🇬🇧 English

### Problem Description

Hotel booking cancellations represent one of the main operational and financial challenges in the hospitality industry. They lead to revenue losses, complicate occupancy planning, and generate operational overhead.

The goal of this project is to build a **binary classification model** capable of predicting in advance whether a booking will be cancelled, enabling the hotel to:

- Manage **overbooking** strategies in a controlled way.
- Apply **differentiated cancellation policies** based on risk profiles.
- Optimise **pricing** and occupancy planning.

### Dataset

| Attribute | Detail |
|-----------|--------|
| **Name** | Hotel Booking Demand |
| **Source** | Public — available on [Kaggle](https://www.kaggle.com/datasets/mojtaba142/hotel-booking/data) |
| **Period** | Bookings from 2015 to 2017 |
| **Size** | ~119,000 records and 36 features |
| **Target** | `is_canceled` — 0: not cancelled / 1: cancelled |
| **Class balance** | ~62% not cancelled / ~37% cancelled |

The dataset includes variables on hotel type, booking lead time, distribution channel, customer history, deposit type, average daily rate (ADR), and more.

> olumns with **data leakage** (`reservation_status`, `reservation_status_date`) and non-predictive personal data (`name`, `email`, `phone-number`, `credit_card`) were removed.

### Approach

A **supervised classification** pipeline was followed, structured in the following stages:

1. **Exploratory Data Analysis (EDA):** distribution analysis, correlations, seasonality and customer segmentation.
2. **Feature Engineering:** log transformations on skewed variables (`lead_time`, `adr`, `previous_cancellations`), geographic grouping of countries, and creation of 7 new derived features (`cancel_ratio`, `loyal_customer`, `lead_time_x_adr`, etc.).
3. **Preprocessing Pipeline:** imputation, scaling and One-Hot encoding integrated in a `ColumnTransformer`, fitted exclusively on training data.
4. **Model comparison:** Logistic Regression (baseline), Random Forest, Gradient Boosting, and KNN.
5. **Hyperparameter tuning:** `GridSearchCV` with 5-fold stratified cross-validation on Random Forest, optimising `balanced_accuracy`.
6. **Final model:** optimised Random Forest, persisted in `src/models/` via `joblib`.

### Repository Structure

```
├── src/
│   ├── data_sample/    # Dataset sample (max. 5 MB) to reproduce the code
│   ├── img/            # Images and charts generated during the project
│   ├── models/         # Trained models saved in joblib format
│   │   ├── random_forest_optimized.joblib
│   │   └── logistic_regression_baseline.joblib
│   ├── notebooks/      # Exploration, EDA and modelling notebooks
│   └── utils/          # Reusable helper modules and functions
├── main.ipynb          # Final notebook: complete, clean and fully executable
├── Presentacion.pdf    # Slide deck supporting the video presentation
└── README.md           # This file
```

### Technologies

| Category | Tools |
|----------|-------|
| **Language** | Python 3.10+ |
| **Data manipulation** | `pandas`, `numpy` |
| **Visualisation** | `matplotlib`, `seaborn` |
| **Machine Learning** | `scikit-learn` (Pipeline, ColumnTransformer, GridSearchCV, RandomForestClassifier, LogisticRegression, GradientBoostingClassifier, KNeighborsClassifier) |
| **Model persistence** | `joblib` |
| **Environment** | Jupyter Notebook |

### How to Reproduce

**1. Clone the repository:**
```bash
git clone https://github.com/eleacos/ML_hotel_booking.git
cd hotel-booking-cancellations
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```
Or manually:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
```

**3. Place the dataset:**

Download `hotel_booking.csv` from [Kaggle](https://www.kaggle.com/datasets/mojtaba142/hotel-booking/data) and place it at:
```
src/data_sample/hotel_booking.csv
```

**4. Run the main notebook:**
```bash
jupyter notebook main.ipynb
```
Run all cells in order (`Kernel > Restart & Run All`). The trained model will be saved to `src/models/` upon completion.

### Results

| Model | Balanced Accuracy | ROC-AUC | F1 — Cancellations |
|-------|:-----------------:|:-------:|:------------------:|
| Logistic Regression (baseline) | 0.775 | 0.866 | 0.72 |
| **Random Forest (optimised)** | **0.820** | **0.910** | **0.77** |

**Most influential features:** `lead_time`, `lead_time_x_adr`, `adr_log`, `deposit_type`, `cancel_ratio`.

**Key conclusions:**
- **Booking lead time** (`lead_time`) is the strongest predictor: longer lead times correlate strongly with higher cancellation probability.
- **Deposit type** and **distribution channel** are key factors: non-refundable bookings and online travel agency bookings show higher cancellation rates.
- The optimised model correctly anticipates approximately **80% of actual cancellations**, with a solid balance between both classes.

---

### Authors

**Brenda Oyola** — https://github.com/Brendaluoyola
**Diana Hoyos** — https://github.com/dianahoyos
**Elena Acosta** — https://github.com/eleacos
