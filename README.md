# 📊 FIC Advisor 2026 — Dashboard Interactivo de Asesoría de Inversión

Dashboard Fintech estilo ejecutivo para analizar y recomendar Fondos de Inversión Colectiva (FICs) de Colombia, basado en datos oficiales de la Superintendencia Financiera.

---

## ¿Qué hace?

| Pestaña | Contenido |
|---|---|
| **📌 Visión General** | KPIs del mercado, Scatter Riesgo/Retorno interactivo, Matriz de Correlación |
| **🗂️ Explorador de Clusters** | Perfiles K-Means, filtros por entidad/perfil/tamaño, tabla dinámica ordenable |
| **🤖 Recomendador** | Wizard personalizado por edad, perfil y horizonte → Asset Allocation + proyección de capital + exportar resumen |

---

## Instalación y ejecución

### Requisitos previos
- Python 3.9 o superior
- pip actualizado

### Paso 1 — Instalar dependencias

Abre una terminal en esta carpeta y ejecuta:

```bash
pip install -r requirements.txt
```

### Paso 2 — Ejecutar la aplicación

```bash
streamlit run app.py
```

La app abre automáticamente en el navegador en `http://localhost:8501`

### Paso 3 — Cargar el archivo CSV

En la pantalla de bienvenida, sube el archivo oficial:

```
Rentabilidades_de_los_Fondos_de_Inversión_Colectiva_(FIC)_20260911.csv
```

El pipeline se ejecuta automáticamente en memoria (~10–30 segundos para 259k filas).

---

## Estructura del proyecto

```
Analitica de negocios MINE 11/
├── app.py                          ← App Streamlit principal  ← EJECUTAR ESTE
├── requirements.txt                ← Dependencias Python
├── README.md                       ← Este archivo
├── analisis_fic.py                 ← Script Python de análisis batch (opcional)
├── index.html                      ← Dashboard HTML estático (versión alternativa)
├── ARQUITECTURA_DASHBOARD_FIC.md   ← Documento de arquitectura y diseño
│
├── Rentabilidades_de_los_Fondos_de_Inversión_Colectiva_(FIC)_20260911.csv  ← DATOS
│
└── outputs/                        ← Generado por analisis_fic.py
    ├── analisis_dashboard.csv      ← Subir al tab Análisis Avanzado del HTML
    ├── analisis_fic.json
    ├── metricas_fondo_participacion.csv
    ├── alertas_anomalias.csv
    └── *.png                       ← Gráficas estáticas
```

---

## Pipeline interno (automático al cargar el CSV)

```
CSV  →  Limpieza UTF-8  →  Filtro fondos ABIERTOS activos
     →  Remoción outliers rentabilidad diaria (-100% a +100%)
     →  Agregación por ID_FONDO_SERIE (CODIGO_NEGOCIO + TIPO_PARTICIPACION)
     →  Cálculo: Patrimonio Medio, Rentabilidad Media, Volatilidad, Ratio Retorno/Riesgo,
                 Log Patrimonio, Flujo Neto Relativo
     →  K-Means k=3 (StandardScaler, nstart=25, seed=123)
     →  Asignación de perfiles de negocio
     →  Dashboard interactivo
```

### Variables del clustering (sin data leakage)
- `RENTABILIDAD_MEDIA` — Retorno esperado
- `VOLATILIDAD_DIARIA` — Riesgo (σ de rentabilidad diaria)
- `RATIO_RETORNO_RIESGO` — Eficiencia (proxy Sharpe, Rf = 0)
- `LOG_PATRIMONIO` — Tamaño (log₁₀ del patrimonio medio)

> El flujo neto se **excluye** del clustering para evitar data leakage (es la variable objetivo del modelo predictivo).

---

## Perfiles de Cluster

| Perfil | Color | Descripción |
|---|---|---|
| 🟢 **Perfil 2 · Fondos Sólidos** | Verde | Rentabilidad positiva y baja volatilidad. **Recomendación principal.** |
| 🔵 **Perfil 3 · Mega Fondos** | Azul | Patrimonio masivo, alta liquidez, escala institucional. |
| 🔴 **Perfil 1 · Riesgo Elevado** | Rojo | Rentabilidad negativa o alta volatilidad. **Señal de alerta.** |

---

## Lógica del Recomendador

| Perfil Cliente | Horizonte | P2 Sólidos | P3 Mega | P1 Riesgo |
|---|---|---|---|---|
| Conservador | Corto <1 año | 80% | 20% | 0% |
| Conservador | Mediano 1–3 a | 75% | 20% | 5% |
| Moderado | Mediano 1–3 a | 60% | 30% | 10% |
| Arriesgado | Largo >3 años | 40% | 25% | 35% |

> La exposición a P1 se reduce automáticamente según la edad del inversionista (factor de ajuste por edad).

---

## Exportaciones disponibles

Desde la pestaña **Recomendador** puedes descargar directamente:
- `Recomendacion_FIC_YYYYMMDD_HHMM.txt` — Resumen ejecutivo en texto
- `Fondos_Recomendados_FIC_YYYYMMDD_HHMM.csv` — Lista de fondos sugeridos con montos

Opcionalmente, configura una **ruta de exportación** en el sidebar para guardar automáticamente.

---

## Dependencias

| Librería | Versión | Uso |
|---|---|---|
| `streamlit` | 1.36.0 | Framework web interactivo |
| `pandas` | 2.2.2 | Manipulación de datos |
| `numpy` | 1.26.4 | Álgebra numérica |
| `plotly` | 5.22.0 | Gráficas interactivas |
| `scikit-learn` | 1.5.1 | K-Means clustering |

---

## Solución de problemas

**El CSV no carga correctamente**
→ Verifica que sea el archivo oficial de la Superfinanciera con las columnas estándar.

**El clustering asigna perfiles inesperados**
→ Los perfiles se asignan dinámicamente por los centroides del K-Means. Los labels de negocio (Sólido/Mega/Riesgo) se determinan comparando los centroides en cada ejecución.

**La app es lenta al cargar**
→ Normal con 259k filas. El procesamiento tarda ~15–30 seg la primera vez; después queda en caché hasta que recargas el archivo.

**Puerto 8501 ocupado**
→ Usa `streamlit run app.py --server.port 8502`

---

*Fuente de datos: Superintendencia Financiera de Colombia — Rentabilidades FIC 2026*
