# ExoData Exoplanet Clustering

Este proyecto organiza el analisis inicial del archivo `PSCompPars` del NASA
Exoplanet Archive para el ExoData Challenge.

## Decision de variables

La guia del concurso no pide usar las 320 columnas como variables de modelado.
El foco esta en variables fisicas, orbitales, estelares y de habitabilidad:

- Planeta: `pl_rade`, `pl_bmasse`, `pl_dens`
- Orbita: `pl_orbper`, `pl_orbsmax`, `pl_orbeccen`
- Estrella/sistema: `st_teff`, `st_met`, `sy_pnum`
- Habitabilidad opcional: `pl_insol`, `pl_eqt`
- Sesgo observacional: `discoverymethod`, `disc_year`

Las 320 columnas si se perfilan para revisar nulos, rangos y metadatos, pero
para clustering conviene comenzar con subconjuntos numericos interpretables.

## Estructura

- `src/eda_exodata.py`: genera el EDA reproducible en Plotly.
- `src/feature_config.py`: define variables clave y grupos de features para clustering.
- `src/impute_exodata.py`: genera matrices imputadas y auditorias para Mapper/TDA y ML.
- `src/mapper_exodata.py`: construye grafos Mapper/TDA y compara su estructura topologica.
- `src/imputation/steps/`: pasos auditables del pipeline de imputacion.
- `src/mapper_tda/`: pipeline modular de preprocesamiento, lenses, clustering, metricas y reportes Mapper.
- `notebooks/`: notebooks renderizables para revisar EDA y preparar clustering.
- `reports/`: salidas generadas: HTML interactivo, tablas de nulos, rangos y correlaciones.
- `data/`: espacio recomendado para organizar datos si despues se mueve el CSV.
- `tests/`: pruebas unitarias del pipeline de imputacion.

## Datos

Los CSV estan en `data/`.

- `PSCompPars_2026.04.25_14.43.08.csv`: tabla completa, 320 columnas.
- `PSCompPars_2026.04.25_17.36.36.csv`: tabla compacta, 84 columnas, mismas filas.

Para clustering inicial, la tabla compacta es mas manejable porque conserva las
variables centrales y elimina muchos enlaces/metadatos. La excepcion importante
es `pl_dens`: si no viene en el CSV, el script la deriva como
`5.514 * pl_bmasse / pl_rade^3`.

## Como ejecutar

Crear el entorno Conda del proyecto:

```powershell
conda env create -f .\environment.yml
conda activate planetas
```

Si tu instalacion de Conda pide aceptar terminos de los canales `defaults`,
crea el mismo entorno solo con `conda-forge`:

```powershell
conda create -y -n planetas --override-channels -c conda-forge python=3.12 pandas numpy plotly scikit-learn scipy networkx nbformat nbconvert ipykernel jupyterlab pytest pip
conda activate planetas
python -m pip install "kmapper>=2.1"
```

```powershell
python .\src\eda_exodata.py
```

El script detecta automaticamente `PSCompPars_*.csv` en `data/` y toma el mas
reciente por nombre.
Tambien puedes pasar un archivo explicitamente:

```powershell
python .\src\eda_exodata.py --csv .\data\PSCompPars_2026.04.25_17.36.36.csv --reports-dir .\reports\PSCompPars_2026.04.25_17.36.36
```

## Imputación de valores faltantes

El pipeline de imputacion esta pensado para no inventar datos sin control. La
familia de metodos evaluados sigue este flujo:

```text
derivacion fisica -> log-transform -> RobustScaler -> KNNImputer -> inversion de escala -> auditoria
```

`KNNImputer` sigue siendo la referencia local natural para Mapper/TDA porque
imputa usando planetas cercanos en el espacio de variables observadas. `median`
queda como baseline robusto. En el reporte actual, `iterative` se visualiza como
metodo principal porque obtuvo el menor rank promedio de error en la validacion
enmascarada de este dataset. Esto no decide por si solo la topologia: Mapper
debe compararse contra casos completos y metodos alternativos.

Antes de imputar se preservan observaciones y se derivan solo propiedades con
formula fisica clara:

```text
pl_dens = 5.514 * pl_bmasse / pl_rade**3
pl_orbsmax = (st_mass * (pl_orbper / 365.25)**2)**(1/3)
```

La transformacion log10 se aplica antes del escalado a variables positivas y
sesgadas (`pl_rade`, `pl_bmasse`, `pl_dens`, `pl_orbper`, `pl_orbsmax`,
`pl_insol`, `sy_dist`, `st_mass`, `st_rad`, `st_lum`). No se aplica log a
`pl_eqt`, `pl_orbeccen`, `st_teff`, `st_met`, flags 0/1 ni categoricas.
`RobustScaler` reduce el impacto de colas largas antes de calcular distancias.

Comandos principales:

```powershell
python .\src\impute_exodata.py --method knn
python .\src\impute_exodata.py --method median
python .\src\impute_exodata.py --method iterative
python .\src\impute_exodata.py --method compare
```

El default equivale a:

```powershell
python .\src\impute_exodata.py --method iterative --visualized-method iterative --n-neighbors 15 --weights distance
```

Ejemplo completo:

```powershell
python .\src\impute_exodata.py --csv .\data\PSCompPars_2026.04.25_17.36.36.csv --reports-dir .\reports\imputation --method compare --visualized-method iterative --n-neighbors 15 --weights distance --max-missing-pct 60 --validation-mask-frac 0.15 --random-state 42
```

Flags opcionales:

- `--include-orbital-eccentricity`: agrega `pl_orbeccen`.
- `--include-stellar-context`: agrega `st_teff`, `st_met`, `st_mass`, `st_rad`, `st_lum`, `sy_pnum`, `sy_snum`.
- `--n-multiple-imputations`: genera varias salidas si `--method iterative`.
- `--visualized-method`: selecciona el metodo que aparece como principal en el reporte; default `iterative`.
- `--outputs-dir`: carpeta para PDFs y tablas exportadas; default `reports/imputation/outputs`.

Salidas principales en `reports/imputation/`:

- `PSCompPars_imputed_knn.csv`, `PSCompPars_imputed_median.csv`, `PSCompPars_imputed_iterative.csv`.
- `mapper_features_complete_case.csv`.
- `mapper_features_imputed_knn.csv`, `mapper_features_imputed_median.csv`, `mapper_features_imputed_iterative.csv`.
- `validation_metrics_<method>.csv` y `validation_predictions_long_<method>.csv`.
- `missingness_profile_before.csv`, `missingness_profile_after_<method>.csv`.
- `feature_coverage_before_after_<method>.csv`.
- `imputed_values_long_<method>.csv`.
- `method_comparison.csv`.
- `imputation_report.html`.
- `outputs/figures_pdf/*.pdf`: cada grafico individual listo para reporte.
- `outputs/tables/*.csv` y `outputs/tables/*.json`: tablas principales de comparacion, validacion, fuentes, missingness y cobertura.

Cada salida agrega indicadores `<col>_was_missing`, `<col>_was_observed`,
`<col>_was_physically_derived`, `<col>_was_imputed` y fuentes `<col>_source`
con valores como `observed`, `derived_density`, `derived_kepler`,
`imputed_knn`, `imputed_median`, `imputed_iterative` o `excluded_too_missing`.
Identificadores, categorias, links, referencias, errores y limites no entran en
la matriz numerica de imputacion. En particular, `pl_dens` se etiqueta como
derivada fisicamente cuando se reconstruye desde masa y radio, no como
observacion astronomica independiente.

Para conclusiones topologicas, comparar siempre:

1. casos completos,
2. KNN imputado,
3. median baseline,
4. iterative sensitivity,
5. bootstrap o variacion de parametros Mapper.

## Mapper/TDA

El pipeline Mapper construye grafos separados para tres espacios de variables:

- `G_phys = Mapper(X_phys)` con `pl_rade`, `pl_bmasse`, `pl_dens`.
- `G_orb = Mapper(X_orb)` con `pl_orbper`, `pl_orbsmax`, `pl_insol`, `pl_eqt`.
- `G_joint = Mapper(X_joint)` con la union de ambos espacios.

Un lens es una funcion `f: X -> R^k`. En este proyecto:

- `pca2` es el lens principal: un solo lens vectorial 2D `f(x) = (PC1, PC2)`.
- `density` es sensibilidad: `f(x) = (PC1, log(d_k + eps))` para rarezas, bordes y periferias.
- `domain` es un lens interpretativo adicional, no el analisis principal.

Los grafos principales son:

- `G_phys_pca2`
- `G_orb_pca2`
- `G_joint_pca2`
- `G_phys_density`
- `G_orb_density`
- `G_joint_density`

PCA2 es el lens principal porque es comparable, reproducible y resume la
variacion global. PCA1 + densidad local queda como analisis de sensibilidad.
No mezclamos ambos lenses en un solo lens 3D/4D por default porque el cover
crece demasiado rapido en dimension alta:

- `10^2 = 100` celdas
- `10^3 = 1000` celdas
- `10^4 = 10000` celdas

Eso puede fragmentar el grafo y volverlo inestable.

Configuracion default:

- `lens=pca2`
- `n_cubes=10`
- `overlap=0.35`
- `clusterer=DBSCAN`
- `min_samples=4`
- `eps_percentile=90`

El script acepta un CSV explicito con `--csv`. Si no se pasa, busca en este
orden:

1. `reports/imputation/mapper_features_imputed_knn.csv`
2. `reports/imputation/mapper_features_complete_case.csv`
3. el `data/PSCompPars_*.csv` mas reciente por nombre

Comandos principales:

```powershell
python .\src\mapper_exodata.py --space phys --lens pca2
python .\src\mapper_exodata.py --space orb --lens pca2
python .\src\mapper_exodata.py --space joint --lens pca2
python .\src\mapper_exodata.py --space all --lens all
python .\src\mapper_exodata.py --space all --lens all --grid
```

Configuracion principal:

```powershell
python .\src\mapper_exodata.py --space all --lens pca2 --n-cubes 10 --overlap 0.35 --clusterer dbscan --min-samples 4 --eps-percentile 90
```

Sensibilidad con densidad:

```powershell
python .\src\mapper_exodata.py --space all --lens density --n-cubes 10 --overlap 0.35 --clusterer dbscan --min-samples 4 --eps-percentile 90
```

Modo completo con grilla:

```powershell
python .\src\mapper_exodata.py --space all --lens all --grid
```

El pipeline genera en `reports/mapper/`:

- grafos JSON
- HTML interactivos por grafo
- tablas de nodos
- tablas de aristas
- metricas de grafos
- distancias entre grafos
- `mapper_report.html`

Regla de interpretacion:

- Los grafos `pca2` son el analisis principal.
- Los grafos `density` son sensibilidad.
- Las conclusiones fuertes requieren estabilidad entre lenses, cubiertas e imputaciones.
- No hacer inferencias cientificas fuertes si una estructura aparece solo bajo una configuracion.
- Las variables observacionales como `discoverymethod` sirven para auditoria y color, no como features principales del Mapper.

## Salida principal

Abre el archivo:

```text
reports/PSCompPars_2026.04.25_17.36.36/exodata_eda_plotly.html
```

Ese reporte contiene visualizaciones interactivas en Plotly para nulos,
distribuciones, correlaciones, sesgos por metodo de descubrimiento y cobertura
de conjuntos de variables para clustering.

## Notebooks

Los notebooks estan pensados para verse en GitHub y ejecutarse localmente:

- `notebooks/01_eda_overview.ipynb`: resumen del EDA, nulos, rangos, cobertura y correlaciones.
- `notebooks/02_clustering_prep.ipynb`: seleccion de variables, transformaciones logaritmicas, escalado y PCA exploratorio.

Para ejecutarlos:

```powershell
jupyter lab
```

## Tests

```powershell
python -m pytest
```

## Repositorio sugerido

Nombre recomendado para GitHub: `exodata-exoplanet-clustering`.

Para publicarlo con GitHub CLI:

```powershell
gh auth login
gh repo create exodata-exoplanet-clustering --public --source . --remote origin --push
```

Si creas primero el repositorio desde la web de GitHub:

```powershell
git remote add origin https://github.com/<tu-usuario>/exodata-exoplanet-clustering.git
git push -u origin main
```
