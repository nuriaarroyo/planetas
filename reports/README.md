# Reportes generados

Esta carpeta contiene salidas reproducibles del EDA:

- `exodata_eda_plotly.html`: reporte interactivo principal.
- `missingness_all_columns.csv`: porcentaje de nulos por columna.
- `key_variable_stats.csv`: rangos y cuantiles de variables clave del concurso.
- `numeric_profile_all_columns.csv`: perfil numerico de todas las columnas numericas.
- `correlation_spearman_core.csv`: correlacion Spearman de variables clave.
- `strong_correlations.csv`: pares de variables con correlacion alta.
- `clustering_feature_coverage.csv`: cobertura de distintos grupos para clustering.
- `guia_contexto_texto.txt`: texto extraido del PDF para busqueda rapida.
- `notebooks_html/`: versiones HTML exportadas de los notebooks para lectura rapida.

Reportes por dataset:

- `PSCompPars_2026.04.25_14.43.08/`: EDA del CSV completo con 320 columnas.
  Sirve como auditoria amplia de todas las columnas del archivo original.
- `PSCompPars_2026.04.25_17.36.36/`: EDA del CSV compacto ubicado en `data/`.
  Este es el reporte recomendado para comenzar clustering porque usa la descarga
  mas reciente y deriva `pl_dens` cuando el archivo no la trae.
