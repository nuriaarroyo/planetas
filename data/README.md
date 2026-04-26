# Datos

Los CSV del NASA Exoplanet Archive viven aqui para que el repositorio tenga una
estructura clara.

Archivos actuales:

- `PSCompPars_2026.04.25_14.43.08.csv`: descarga completa con 320 columnas.
- `PSCompPars_2026.04.25_17.36.36.csv`: descarga mas compacta con 84 columnas
  seleccionadas y las mismas 6273 filas. Es mas comoda para empezar clustering,
  pero no trae `pl_dens`; el script deriva esa variable desde masa y radio si
  hace falta.

Estructura recomendada si crece el proyecto:

```text
data/
  raw/          # archivos descargados sin modificar
  interim/      # datos filtrados o parcialmente limpios
  processed/    # matrices listas para modelado
```

Los scripts detectan automaticamente archivos `PSCompPars_*.csv` en `data/` o
pueden recibir uno explicito con `--csv`.
