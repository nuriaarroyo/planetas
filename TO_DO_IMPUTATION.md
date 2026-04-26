# Prompt para Codex: cerrar pipeline/reporte de imputacion

Trabaja sobre el repo existente `planetas` sin reescribirlo desde cero. Mantén la estructura actual y respeta los cambios ya hechos en `src/imputation/pipeline.py`, `src/impute_exodata.py`, `tests/test_impute_exodata.py`, `README.md`, `requirements.txt` y `environment.yml`.

## Contexto actual

Se corrigio gran parte del pipeline de imputacion:

- `ImputationConfig` ahora tiene `method="iterative"` y `visualized_method="iterative"` por default.
- El CLI `src/impute_exodata.py` acepta `--visualized-method`, `--outputs-dir` y `--skip-figure-export`.
- El reporte ya deberia visualizar `iterative` cuando se corre:

```powershell
conda run -n planetas python .\src\impute_exodata.py --method compare --visualized-method iterative
```

- Se generaron outputs en:

```text
reports/imputation/outputs/
  figures_pdf/
  tables/
```

- Existen 17 PDFs individuales en `reports/imputation/outputs/figures_pdf/`.
- Se agregaron tablas CSV/JSON en `reports/imputation/outputs/tables/`.
- Se agrego `matplotlib` como dependencia para exportar PDFs estaticos.
- `pl_dens` se etiqueta como derivada fisicamente cuando viene de masa y radio, no como observacion independiente.
- La validacion enmascarada ahora incluye columnas para distinguir `observed`, `physically_derived` y bases mixtas.

## Tareas pendientes

1. Verificar el HTML final

Confirma que `reports/imputation/imputation_report.html` contiene:

- `METODO VISUALIZADO`
- `<strong>iterative</strong>`
- notas claras sobre imputacion, derivacion fisica, `pl_dens` y estabilidad topologica.
- texto de validacion que no diga incorrectamente que todos los targets eran observaciones originales.

Usa algo como:

```powershell
Select-String -Path reports\imputation\imputation_report.html -Pattern "METODO VISUALIZADO|<strong>iterative</strong>|pl_dens|Derived/available|fisicamente|topologica"
```

2. Verificar que los scatter checks usan `iterative`

Confirma en el HTML y/o PDFs que los scatter finales fueron generados desde `method_result` de `iterative`, no de `knn`.

Revisar especialmente:

- `14_scatter_mass_radius.pdf`
- `15_scatter_density_radius.pdf`
- `16_scatter_orbper_orbsmax.pdf`
- `17_scatter_insol_eqt.pdf`

3. Verificar PDFs no vacios

Confirma que existan y pesen más de 0 bytes:

```text
01_missingness_before_after.pdf
02_value_source_composition.pdf
03_mapper_coverage.pdf
04_masked_validation_mae_by_feature.pdf
05_masked_validation_spearman_heatmap.pdf
06_method_comparison.pdf
07_distribution_pl_rade.pdf
08_distribution_pl_bmasse.pdf
09_distribution_pl_dens.pdf
10_distribution_pl_orbper.pdf
11_distribution_pl_orbsmax.pdf
12_distribution_pl_insol.pdf
13_distribution_pl_eqt.pdf
14_scatter_mass_radius.pdf
15_scatter_density_radius.pdf
16_scatter_orbper_orbsmax.pdf
17_scatter_insol_eqt.pdf
```

Comando sugerido:

```powershell
Get-ChildItem reports\imputation\outputs\figures_pdf\*.pdf |
  Select-Object Name,Length |
  Sort-Object Name
```

4. Verificar tablas exportadas

Confirma que existan en `reports/imputation/outputs/tables/` tanto CSV como JSON:

- `imputation_method_comparison`
- `imputation_validation_metrics`
- `imputation_value_source_composition`
- `imputation_missingness_summary`
- `mapper_coverage_summary`

Revisa especialmente:

```powershell
Get-Content reports\imputation\outputs\tables\imputation_value_source_composition.csv
Get-Content reports\imputation\outputs\tables\imputation_validation_metrics.csv
```

Criterio esperado:

- `pl_dens` debe tener `observed_count = 0` o muy bajo si el CSV no trae observaciones originales.
- `pl_dens` debe tener `physically_derived_count` alto.
- `pl_dens` debe tener `validation_basis = physically_derived` en las metricas de validacion si la validacion se hizo contra valores derivados.

5. Correr pruebas completas

Ejecuta:

```powershell
conda run -n planetas python -m pytest -q
```

Debe pasar. Si aparece solo el warning de OpenMP/threadpool, documentalo como warning del stack numerico local, no como fallo del pipeline.

6. Ejecutar `git diff --check`

```powershell
git diff --check
```

Si hay warnings de CRLF/LF, no hace falta cambiarlos salvo que aparezcan errores reales de whitespace.

7. Revisar layout del HTML de forma visual

Abre:

```text
reports/imputation/imputation_report.html
```

Revisa manualmente:

- tablas con overflow horizontal;
- leyendas visibles;
- titulos no cortados;
- histogramas no vacios;
- distribuciones con `log10(...)` cuando corresponda;
- mensajes explicitos cuando una grafica no tiene suficientes valores;
- que el reporte sea legible al imprimir/exportar a PDF desde navegador.

8. Revisar nombres y narrativa de fuentes

Actualmente las fuentes internas pueden incluir:

- `observed`
- `derived_density`
- `derived_kepler`
- `imputed_knn`
- `imputed_median`
- `imputed_iterative`
- `excluded_too_missing`

Y las categorias resumidas son:

- `observed`
- `physically_derived`
- `imputed`
- `excluded_too_missing`
- `missing`

Verifica que esta distincion aparezca consistentemente en:

- HTML;
- `imputation_value_source_composition.csv`;
- plots de composicion;
- scatter status;
- tablas largas de imputacion.

9. Confirmar que no quedan NaN ni infinitos en features Mapper

Para el metodo visualizado `iterative`, verifica:

```python
features = ["pl_rade", "pl_bmasse", "pl_dens", "pl_orbper", "pl_orbsmax", "pl_insol", "pl_eqt"]
```

Criterios:

- sin `NaN`;
- sin `inf` ni `-inf`;
- features positivas usadas en log deben ser `> 0`:
  - `pl_rade`
  - `pl_bmasse`
  - `pl_dens`
  - `pl_orbper`
  - `pl_orbsmax`
  - `pl_insol`

10. Revisar que no haya archivos accidentales

Hubo archivos accidentales tipo `3.8`, `2.1`, `8.0` causados por PowerShell interpretando `>=` como redireccion. Verifica que ya no existan:

```powershell
Test-Path .\3.8
Test-Path .\2.1
Test-Path .\8.0
```

Si existen y son logs accidentales, eliminalos con `Remove-Item -LiteralPath`.

11. Actualizar README si hace falta

Ya se actualizo el README para indicar:

```powershell
python .\src\impute_exodata.py --method compare --visualized-method iterative
```

Pero revisa si conviene agregar explicitamente:

```powershell
python .\src\impute_exodata.py --method compare --visualized-method iterative --outputs-dir .\reports\imputation\outputs
```

Tambien confirma que README no siga diciendo que KNN es el default visualizado.

12. Revisar dependencias

Confirmar que existan:

- `matplotlib>=3.8`
- `networkx>=3.2`
- `kmapper>=2.1`

En:

- `requirements.txt`
- `environment.yml`

No uses comandos PowerShell sin comillas con `>=`, porque crean archivos accidentales. Usa comillas:

```powershell
conda install -y -n planetas --override-channels -c conda-forge "matplotlib>=3.8"
python -m pip install "kmapper>=2.1"
```

13. Cierre final esperado

Al final, reporta:

- ruta del HTML final;
- ruta de `reports/imputation/outputs/figures_pdf/`;
- numero de PDFs generados;
- ruta de tablas exportadas;
- resultado de tests;
- cualquier warning no bloqueante;
- confirmacion de que `METODO VISUALIZADO = iterative`.

## Criterios de aceptacion

El trabajo queda completo si:

1. El reporte dice `METODO VISUALIZADO = iterative`.
2. Los plots dependientes del metodo usan `iterative`.
3. Los histogramas de Distribution Checks no estan vacios por problemas de escala.
4. Los scatter checks finales usan `iterative`.
5. Se exportan los 17 PDFs individuales con tamaño mayor a 0 bytes.
6. Se exportan las tablas principales como CSV y JSON.
7. El reporte distingue observados, derivados fisicamente e imputados.
8. `pl_dens` se describe como derivada desde masa y radio cuando aplique.
9. No quedan NaN/inf en las 7 features Mapper principales.
10. La estructura existente del repo no se rompe.

