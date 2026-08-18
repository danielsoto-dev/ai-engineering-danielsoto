# Búsqueda híbrida y reranking

Sesión 10, pre-work. Mido si la búsqueda híbrida y el reranking mejoran la
recuperación sobre el corpus de presupuestos, y cuánto cuestan.

## 1. Búsqueda full-text en PostgreSQL

Migración `0002_chunk_fulltext_search.py`:

```sql
ALTER TABLE chunks
ADD COLUMN content_tsv tsvector
GENERATED ALWAYS AS (to_tsvector('spanish', content)) STORED;

CREATE INDEX ix_chunks_content_tsv ON chunks USING GIN (content_tsv);
```

Columna generada en vez de trigger: Postgres la mantiene sincronizada sola. La
configuración `spanish` importa porque aplica stemming del idioma:
`to_tsvector('spanish', 'pasarela de pagos')` da `'pag'` y `'pasarel'`. Con
`english` la rama léxica fallaría en cuanto la consulta usara el singular y el
chunk el plural.

## 2. Rama léxica y fusión RRF

En `app/retrieval/hybrid.py`. RRF combina los rankings por posición,
`Σ 1/(k + rank)` con k=60 configurable. Fusionar por posición y no por score es
lo que permite mezclar una distancia coseno con un `ts_rank_cd`, que viven en
escalas incomparables.

Aquí me encontré el fallo más interesante del ejercicio. La primera versión
usaba `websearch_to_tsquery`, que une todos los términos con AND:

```
websearch_to_tsquery('spanish','tienda online con catálogo de productos y carrito')
→ 'tiend' & 'onlin' & 'catalog' & 'product' & 'carrit'
```

Ningún chunk contiene esas cinco palabras a la vez, así que la rama léxica
devolvía cero en las cinco consultas y B daba exactamente los mismos números
que A: estaba fusionando la rama vectorial con una lista vacía. Cambié a
lexemas unidos por OR y dejé que `ts_rank_cd` ordenara por cobertura de
términos. La misma consulta pasó a recuperar 10 chunks.

Sin haberlo detectado, la tabla habría dicho que la búsqueda híbrida no aporta
nada, y esa conclusión habría venido de un bug.

## 3. Integración del reranker

`app/retrieval/reranker.py`, con `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`,
multilingüe porque corpus y consultas están en español. Recuperación a 50,
reordenación a 5.

`mode` y `rerank` son parámetros del endpoint, así que las cuatro
configuraciones se invocan sin tocar código:

```bash
curl -s localhost:8000/search -H 'Content-Type: application/json' \
  -d '{"query":"plataforma de comercio electrónico con catálogo y carrito","k":5,"mode":"hybrid","rerank":true}'
```

## 4. Golden set y medición

`evals/retrieval_golden_set.json`: 5 consultas con los chunks relevantes
anotados a mano. El criterio es mismo dominio funcional, es decir que un chunk
cuenta como relevante si resuelve el mismo problema técnico que describe la
consulta, aunque el cliente sea de otro sector. El sistema estima horas y las
horas dependen del trabajo de ingeniería: un servidor OAuth cuesta parecido en
un banco que en un gimnasio.

Empecé con 3 presupuestos, 6 chunks, y ahí la medición no dice nada: cualquier
top-5 devuelve el 83% del corpus y no hay 50 candidatos que reordenar. Lo
amplié a 25 presupuestos en español con distractores puestos a propósito. El
corpus final son 28 documentos y 70 chunks.

| Config | Búsqueda | Reranking | P@5 medio | Latencia media (ms) |
| ------ | --------- | --------- | --------- | ------------------- |
| A | Vectorial | No | 0.76 | 665 |
| B | Híbrida | No | 0.84 | 563 |
| C | Vectorial | Sí | 0.68 | 802 |
| D | Híbrida | Sí | 0.72 | 637 |

Por consulta:

| Query | A | B | C | D |
| --- | --- | --- | --- | --- |
| Q1 tienda online con catálogo y carrito | 0.80 | 1.00 | 0.80 | 0.80 |
| Q2 buscador con filtros por precio | 0.60 | 0.80 | 0.40 | 0.60 |
| Q3 OAuth, doble factor y cumplimiento | 0.80 | 0.80 | 0.80 | 0.80 |
| Q4 app de reserva de cita con avisos | 0.80 | 0.80 | 0.60 | 0.60 |
| Q5 panel de sensores en tiempo real | 0.80 | 0.80 | 0.80 | 0.80 |

La latencia total engaña, porque la domina una llamada de red ajena a estas
técnicas: embedding 580 ms, retrieval más RRF 29 ms, reranking de 50
candidatos 184 ms.

## 5. Conclusiones

Usaría la configuración B, híbrida sin reranking. Es la más precisa con 0.84 y
además la más rápida, cosa que no esperaba antes de medir.

La búsqueda híbrida aporta lo que promete. Donde más gana es en Q1, que sube de
0.80 a 1.00: la rama léxica ancla términos que el embedding diluye, porque una
transcripción larga produce un vector promediado donde "carrito de la compra"
pesa poco, mientras que en el índice full-text ese término aparece literal.
Cuesta 29 ms, así que no hay trade-off que discutir.

El reranking no compensa aquí, y la latencia no es el motivo. Baja la precisión
de 0.84 a 0.72. Mirando los scores se entiende por qué: en Q2 el cross-encoder
puntúa el mejor chunk con −0.39 y todo lo demás se desploma a −5.9, −7.5, −7.7,
−7.9. Identifica el primer resultado y a partir de ahí no discrimina, y en esa
zona plana expulsa un relevante para colar uno que no lo es. Con 70 chunks,
pedir un top-50 mete el 71% del corpus en la reordenación, así que el reranker
recibe sobre todo ruido.

Eso apunta a que el resultado depende más del tamaño del corpus que de la
técnica. Con miles de chunks, un top-50 sería una preselección de verdad. Antes
de descartarlo repetiría la medición bajando la profundidad de recall a 15 o 20
y ampliando el golden set, porque cinco consultas dan pasos de 0.20 en P@5 y
eso es demasiado grueso para diferencias finas.

## Reproducir

```bash
docker compose up -d
docker compose exec estimator alembic upgrade head
uv run python scripts/ingest_budgets.py data/budgets_sample.json
uv run python scripts/ingest_budgets.py data/budgets_extended.json
uv run python scripts/evaluate_retrieval.py
```

Resultados por consulta en `evals/retrieval_results.md`.
