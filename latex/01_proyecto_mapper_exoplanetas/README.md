# Atlas topologico de exoplanetas con Mapper

Este proyecto LaTeX contiene un reporte metodologico para estudiar exoplanetas con Topological Data Analysis usando Mapper.

## Archivos

- `main.tex`: reporte principal.
- `references.bib`: bibliografia BibTeX.
- `atlas_topologico_exoplanetas.pdf`: PDF compilado.

## Compilacion

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Tema

El reporte propone construir tres Mappers:

- Mapper fisico: masa, radio y densidad.
- Mapper orbital: periodo, semieje mayor, insolacion y temperatura de equilibrio.
- Mapper conjunto: union de variables fisicas y orbitales.

Tambien incluye diagnostico de sesgo por metodo de descubrimiento, validacion por bootstrap, pruebas de permutacion y ruta hacia machine learning.
