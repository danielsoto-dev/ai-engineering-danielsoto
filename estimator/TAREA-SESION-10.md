# Sesión 10 — Qué pedía la tarea y cómo se resolvió

Documento de acompañamiento al entregable `recuperacion-avanzada.md`. Explica el
enunciado, lo que había antes de empezar, y los problemas que hubo que resolver
por el camino.

## Lo que pedía el enunciado

El pipeline RAG recupera presupuestos por similitud vectorial, pero "similar" no
siempre significa "relevante": el sistema puede devolver un presupuesto de una app
de pagos cuando la consulta describe una plataforma de e-commerce. Cercano en el
espacio vectorial, inútil para estimar.

El ejercicio pedía atacar eso con dos técnicas y, sobre todo, **medir si compensan**:

1. Búsqueda full-text en PostgreSQL — columna `tsvector` con índice GIN, en español
2. Rama léxica y fusión RRF — combinar keyword y vectorial en un ranking único
3. Integración del reranker — cross-encoder con patrón recall-then-rerank (top-50 → top-5)
4. Golden set y medición — 5 consultas anotadas a mano, cuatro configuraciones
5. Conclusiones — qué configuración usar y por qué

Alcance explícitamente excluido: expansión de consultas, routing multi-índice y
filtrado por metadatos. Eso se construye en el directo.

| Configuración | Búsqueda | Reranking |
| --- | --- | --- |
| A | Vectorial | No |
| B | Híbrida | No |
| C | Vectorial | Sí |
| D | Híbrida | Sí |

## Punto de partida

El repo venía de Sesión 09, donde el trabajo fue un diagnóstico arquitectónico, no
implementación. Lo que existía:

- `POST /search` con búsqueda vectorial pura (distancia coseno sobre `chunks.embedding`)
- PostgreSQL con pgvector, migración `0001`
- Pipeline de chunking y embeddings
- Corpus de **3 presupuestos → 6 chunks**

El enunciado también daba por supuesto un wrapper de cross-encoder ya construido en
el repo del profesor. Como este trabajo parte del repo propio, el wrapper se
implementó desde cero con `sentence-transformers`.

## Problema 1 — El corpus no permitía medir

**Síntoma.** Con 6 chunks, cualquier top-5 devuelve el 83% del corpus. Las cuatro
configuraciones darían prácticamente el mismo número, y el patrón recall-then-rerank
no tiene 50 candidatos que reordenar.

**Contexto.** Ya estaba anotado como Fallo 4 en el diagnóstico de Sesión 09. Aquí
pasó de ser una observación a un bloqueante.

**Qué se buscó primero.** Se revisaron todas las ramas del repo por si había un seed
mayor. Los únicos que existían estaban en las ramas del profesor y además usaban otro
esquema (`phases`/`amount`, sin `components` ni `tech_stack`), incompatible con el
chunker.

**Solución.** `data/budgets_extended.json`: 25 presupuestos, 64 componentes, 12
sectores, mismo esquema que el original. Escritos en español —el enunciado asume un
corpus en español y el original estaba en inglés, sin lo cual la configuración
`spanish` del tsvector no haría nada útil— e incluyendo distractores deliberados:
componentes que comparten vocabulario entre dominios distintos.

Corpus final: **28 documentos, 70 chunks**.

## Problema 2 — La rama léxica devolvía cero resultados

**Síntoma.** La configuración B (híbrida) daba exactamente los mismos P@5 que A
(vectorial) en las cinco consultas, y `candidates_considered` se quedaba en 5.

**Diagnóstico.** La primera implementación usaba `websearch_to_tsquery`, que une
todos los términos con AND:

```
websearch_to_tsquery('spanish','tienda online con catálogo de productos y carrito de la compra')
→ 'tiend' & 'onlin' & 'catalog' & 'product' & 'carrit' & 'compr'
```

Ningún chunk contiene esas seis palabras a la vez. La rama léxica no casaba con nada,
así que RRF fusionaba la rama vectorial con una lista vacía.

**Solución.** Lexemas unidos por OR, dejando que `ts_rank_cd` ordene por cuántos
términos distintos cubre cada chunk. Tras el cambio, la misma consulta recupera 10
chunks encabezados por catálogo y carrito.

Este es el fallo más relevante del ejercicio: sin detectarlo, la tabla habría
mostrado que "la búsqueda híbrida no aporta nada", que es una conclusión falsa
derivada de un bug, no de la técnica.

## Problema 3 — Torch arrastraba 2.5 GB de CUDA al contenedor

**Síntoma.** El build de Docker corrió 90 minutos y murió con un timeout descargando
`nvidia-nvshmem`. Descargaba `nvidia-cublas` (517 MB), `nvidia-cudnn` (424 MB),
`torch` (407 MB) y 40 paquetes más.

**Diagnóstico.** Son drivers de GPU NVIDIA. La máquina es ARM sin GPU NVIDIA y nunca
se cargarían. Torch los arrastra por defecto en Linux, que es el sistema del
contenedor; en macOS el entorno local resuelve una build sin CUDA, por eso allí no
pasaba.

La solución documentada por Astral (`[tool.uv.sources]` apuntando al índice CPU) se
ignoraba en silencio. La causa real: **`torch` no era dependencia directa del
proyecto** —entraba por `sentence-transformers`— y `tool.uv.sources` solo gobierna
lo que el proyecto declara explícitamente.

**Solución.** Declarar `torch` como dependencia directa, con marker `sys_platform ==
'linux'` para que solo afecte al contenedor.

| | Antes | Después |
| --- | --- | --- |
| Paquetes NVIDIA en el lock | 15 | 0 |
| Paquetes CUDA (`triton`, `cuda-*`) | 4 | 0 |
| torch en Linux | `2.13.0` (PyPI) | `2.13.0+cpu` |

El lock se regeneró preservando las versiones fijadas: el diff final elimina 19
paquetes CUDA y cambia una sola versión (`torch`), sin subidas colaterales.

## Problema 4 — El reranker no podía escribir su caché

**Síntoma.** `POST /search` con `rerank: true` devolvía HTTP 500 con
`PermissionError: '/home/appuser/.cache/huggingface/hub'`.

**Diagnóstico.** El volumen para cachear el modelo lo crea Docker como `root`, y el
contenedor corre como `appuser` (uid 999).

**Solución.** Crear el directorio en la imagen antes de cambiar de usuario, de modo
que el volumen montado encima herede la propiedad correcta.

## Problema 5 — La suite de tests vacía la base de datos de desarrollo

**Síntoma.** Tras una ejecución de tests, `/search` devolvió `candidates_considered:
1` con un chunk `TEST-001` que no pertenece al corpus.

**Diagnóstico.** La fixture `clean_db` de `tests/test_embeddings_pgvector.py` ejecuta
`TRUNCATE documents, chunks RESTART IDENTITY CASCADE` sobre la misma base de datos que
usa el entorno de desarrollo.

**Estado.** Se re-ingirió el corpus y se verificó (28 docs / 70 chunks). **No se
corrigió**: viene de Sesión 08 y queda fuera del alcance de este ejercicio. Conviene
tenerlo presente — cualquier `uv run pytest` vacía el corpus y obliga a re-ingerir con
`scripts/ingest_budgets.py`.

## Resultados

| Config | Búsqueda | Reranking | P@5 medio | Latencia media |
| --- | --- | --- | --- | --- |
| A | Vectorial | No | 0.76 | 665 ms |
| **B** | **Híbrida** | **No** | **0.84** | **563 ms** |
| C | Vectorial | Sí | 0.68 | 802 ms |
| D | Híbrida | Sí | 0.72 | 637 ms |

Desglose de latencia: embedding 580 ms · retrieval + RRF 29 ms · reranking 184 ms.

**Conclusión: configuración B.** La búsqueda híbrida sube la precisión y además baja
la latencia; sus 29 ms cubren las dos ramas y la fusión, así que no hay trade-off que
discutir.

**El reranking no compensa en este corpus**, y la latencia no es el motivo. Baja la
precisión de 0.84 a 0.72. Al inspeccionar los scores se ve por qué: el cross-encoder
puntúa su mejor resultado con claridad y luego se aplana (en Q2: −0.39 seguido de
−5.9, −7.5, −7.7, −7.9). En esa zona plana el orden es casi arbitrario, y con 70
chunks un recall de 50 le entrega el 71% del corpus como ruido. El razonamiento
completo está en `recuperacion-avanzada.md`.

## Qué se entregó

| Archivo | Contenido |
| --- | --- |
| `recuperacion-avanzada.md` | Entregable: los cinco pasos, tabla y conclusiones |
| `alembic/versions/0002_chunk_fulltext_search.py` | Columna `tsvector` española + índice GIN |
| `app/retrieval/hybrid.py` | Búsqueda vectorial, léxica y fusión RRF |
| `app/retrieval/reranker.py` | Wrapper del cross-encoder |
| `evals/retrieval_golden_set.json` | 5 consultas con relevancia anotada |
| `evals/retrieval_results.md` | Resultados por consulta y configuración |
| `scripts/evaluate_retrieval.py` | Medición de las cuatro configuraciones |
| `scripts/ingest_budgets.py` | Ingesta del corpus |
| `data/budgets_extended.json` | 25 presupuestos en español |
| `tests/test_rrf.py` | Tests de la fusión RRF |

## Reproducir

```bash
docker compose up -d
docker compose exec estimator alembic upgrade head
uv run python scripts/ingest_budgets.py data/budgets_sample.json
uv run python scripts/ingest_budgets.py data/budgets_extended.json
uv run python scripts/evaluate_retrieval.py
```

Las cuatro configuraciones también son invocables por HTTP:

```bash
curl -s localhost:8000/search -H 'Content-Type: application/json' \
  -d '{"query":"plataforma de comercio electrónico con catálogo y carrito","k":5,"mode":"hybrid","rerank":true}'
```
