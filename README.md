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
- `notebooks/`: notebooks renderizables para revisar EDA y preparar clustering.
- `reports/`: salidas generadas: HTML interactivo, tablas de nulos, rangos y correlaciones.
- `data/`: espacio recomendado para organizar datos si despues se mueve el CSV.

## Datos

Los CSV estan en `data/`.

- `PSCompPars_2026.04.25_14.43.08.csv`: tabla completa, 320 columnas.
- `PSCompPars_2026.04.25_17.36.36.csv`: tabla compacta, 84 columnas, mismas filas.

Para clustering inicial, la tabla compacta es mas manejable porque conserva las
variables centrales y elimina muchos enlaces/metadatos. La excepcion importante
es `pl_dens`: si no viene en el CSV, el script la deriva como
`5.514 * pl_bmasse / pl_rade^3`.

## Como ejecutar

```powershell
python .\src\eda_exodata.py
```

El script detecta automaticamente `PSCompPars_*.csv` en `data/` y toma el mas
reciente por nombre.
Tambien puedes pasar un archivo explicitamente:

```powershell
python .\src\eda_exodata.py --csv .\data\PSCompPars_2026.04.25_17.36.36.csv --reports-dir .\reports\PSCompPars_2026.04.25_17.36.36
```

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
