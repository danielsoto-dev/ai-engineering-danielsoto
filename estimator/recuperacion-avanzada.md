# Búsqueda híbrida y reranking — Sesión 10 (pre-work)

Medición de si la búsqueda híbrida y el reranking con cross-encoder mejoran la
recuperación sobre el corpus de presupuestos históricos, y a qué coste.

## 0. Punto de partida y ampliación del corpus

El corpus al cierre de Sesión 09 eran 3 presupuestos → 6 chunks. Con 6 chunks
la medición no discrimina: cualquier top-5 devuelve el 83% del corpus, y el
patrón recall-then-rerank (top-50 → top-5) no tiene 50 candidatos que
reordenar. Es el Fallo 4 que ya anoté en el diagnóstico de la sesión anterior,
convertido ahora en bloqueante.

Añadí `data/budgets_extended.json`: 25 presupuestos, 64 componentes, 12
sectores, con el mismo esquema que `budgets_sample.json`. Están redactados en
español, porque el enunciado asume un corpus en español y el original estaba en
inglés — sin eso, la configuración `spanish` del tsvector no haría nada útil.
El dataset incluye distractores deliberados: la app de pagos entre particulares
(`BUD-2024-104`) comparte vocabulario con el e-commerce (`BUD-2024-101`) pero
resuelve otro problema.

Corpus final: **28 documentos, 70 chunks**, todos con embedding y tsvector.

```bash
docker compose exec estimator alembic upgrade head
uv run python scripts/ingest_budgets.py data/budgets_extended.json
```

## 1. Búsqueda full-text en PostgreSQL

Migración `0002_chunk_fulltext_search.py`: columna `content_tsv` generada
siempre a partir de `content` con la configuración `spanish`, más índice GIN.

```sql
ALTER TABLE chunks
ADD COLUMN content_tsv tsvector
GENERATED ALWAYS AS (to_tsvector('spanish', content)) STORED;

CREATE INDEX ix_chunks_content_tsv ON chunks USING GIN (content_tsv);
```

Columna generada en lugar de trigger: Postgres la mantiene sincronizada en cada
insert y update, sin código que mantener. La configuración `spanish` aplica
stemming y stopwords del idioma, lo que se comprueba fácil:

```
select to_tsvector('spanish', 'Integración de pasarela de pagos');
→ 'integr':1 'pag':5 'pasarel':3
```

`pagos` → `pag` y `pasarela` → `pasarel`: con la configuración `english` esas
palabras no se habrían reducido y la rama léxica fallaría en singular/plural.

## 2. Rama léxica y fusión RRF

En `app/retrieval/hybrid.py`. La fusión combina rankings por posición,
`Σ 1/(k + rank)` con k=60 configurable, que es lo que permite mezclar una
distancia coseno con un `ts_rank_cd` sin que sus escalas sean comparables.

**Un problema que encontré midiendo.** La primera implementación usaba
`websearch_to_tsquery`, y la rama léxica devolvía **cero resultados en las cinco
consultas**. La causa es que esa función une todos los términos con AND:

```
select websearch_to_tsquery('spanish','tienda online con catálogo de productos y carrito de la compra');
→ 'tiend' & 'onlin' & 'catalog' & 'product' & 'carrit' & 'compr'
```

Ningún chunk contiene esas seis palabras a la vez, así que la consulta no
casaba con nada. El síntoma en la tabla era que la configuración B daba
exactamente los mismos números que A: estaba fusionando la rama vectorial con
una lista vacía. Las consultas de este dominio son descripciones de proyecto en
lenguaje natural, no cadenas de palabras clave, así que cambié a lexemas unidos
por OR y dejé que `ts_rank_cd` ordene por cuántos términos distintos cubre cada
chunk. Tras el cambio la misma consulta recupera 10 chunks, encabezados por
catálogo (0.5) y carrito (0.4).

## 3. Integración del reranker

En `app/retrieval/reranker.py`, con `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
— multilingüe, porque corpus y consultas están en español. Patrón
recall-then-rerank: recuperación amplia a 50 y reordenación fina a 5.

El modo de búsqueda y el reranking son parámetros del endpoint, así que las
cuatro configuraciones se invocan sin tocar código:

```bash
curl -s localhost:8000/search -H 'Content-Type: application/json' \
  -d '{"query":"plataforma de comercio electrónico con catálogo y carrito","k":5,"mode":"hybrid","rerank":true}'
```

Que el cross-encoder distingue intención y no solo vocabulario se ve aislándolo:

| chunk | score |
| --- | --- |
| Carrito de la compra y proceso de checkout | **+2.46** |
| Servicio de catálogo de productos | **+2.08** |
| Motor de procesamiento de pagos (monederos P2P) | −6.59 |
| Modelo de predicción de averías (industrial) | −9.20 |

Separa el e-commerce de la app de pagos pese a compartir la palabra "pago", que
es justo el caso que el ejercicio plantea.

## 4. Golden set y medición

Golden set en `evals/retrieval_golden_set.json`: 5 consultas con los chunks
relevantes anotados a mano. El criterio es **mismo dominio funcional**: un chunk
es relevante si su componente resuelve el mismo problema técnico que describe la
consulta, aunque el sector del cliente sea otro. El sistema estima horas, y las
horas dependen del trabajo de ingeniería, no de la industria: un servidor OAuth
cuesta lo mismo en un banco que en un gimnasio. Anotar por sector habría contado
como fallo una recuperación que sí sirve para estimar.

```bash
uv run python scripts/evaluate_retrieval.py
```

### Tabla comparativa

| Config | Búsqueda | Reranking | P@5 medio | Latencia media (ms) | Latencia mediana (ms) |
| ------ | --------- | --------- | --------- | ------------------- | --------------------- |
| A | Vectorial | No | 0.76 | 665 | 514 |
| **B** | **Híbrida** | **No** | **0.84** | **563** | **510** |
| C | Vectorial | Sí | 0.68 | 802 | 627 |
| D | Híbrida | Sí | 0.72 | 637 | 643 |

### Detalle por consulta

| Query | A | B | C | D |
| --- | --- | --- | --- | --- |
| Q1 tienda online con catálogo y carrito | 0.80 | **1.00** | 0.80 | 0.80 |
| Q2 buscador con filtros por precio | 0.60 | **0.80** | 0.40 | 0.60 |
| Q3 OAuth, doble factor y cumplimiento | 0.80 | 0.80 | 0.80 | 0.80 |
| Q4 app de reserva de cita con avisos | 0.80 | **0.80** | 0.60 | 0.60 |
| Q5 panel de sensores en tiempo real | 0.80 | 0.80 | 0.80 | 0.80 |

### Desglose de latencia

La latencia total engaña, porque está dominada por una llamada de red que no
tiene nada que ver con estas técnicas:

| Etapa | Media |
| --- | --- |
| Embedding de la consulta (API OpenAI) | 580 ms |
| Retrieval vectorial + léxico + RRF | **29 ms** |
| Reranking de 50 candidatos | **184 ms** |

La búsqueda híbrida es esencialmente gratis: los 29 ms cubren las dos ramas y la
fusión. El reranking cuesta 184 ms reales sobre 50 candidatos.

## 5. Conclusiones

**Usaría la configuración B: híbrida sin reranking.** Es la más precisa (0.84) y
además la más rápida de las cuatro, lo que no esperaba antes de medir.

La búsqueda híbrida aporta lo que el enunciado anticipa: donde más gana es en
Q1, que pasa de 0.80 a 1.00. La rama léxica ancla términos que el embedding
diluye. Una transcripción larga produce un vector promediado donde "carrito de
la compra" pesa poco, mientras que en el índice full-text ese término aparece
literal y `ts_rank_cd` lo premia. Cuestan 29 ms las dos ramas juntas, así que no
hay trade-off que discutir: compensa.

**El reranking no compensa aquí, y la latencia no es el motivo.** Baja la
precisión: de 0.84 a 0.72 sobre híbrida, y de 0.76 a 0.68 sobre vectorial.
Los 184 ms serían asumibles en un flujo donde después va una generación con LLM
de varios segundos; el problema es que estoy pagándolos por empeorar el
resultado.

Al mirar los scores se ve por qué. En Q2 el cross-encoder puntúa el mejor chunk
con −0.39 y todo lo demás se desploma a −5.9, −7.5, −7.7, −7.9. Identifica con
claridad el primer resultado y luego no discrimina: en esa zona plana el orden
es casi arbitrario, y ahí es donde expulsa un relevante para colar uno que no lo
es. Con un corpus de 70 chunks, la recuperación amplia a 50 mete el 71% del
corpus en la reordenación, así que el reranker recibe sobre todo ruido y sus
decisiones en la cola pesan más que su acierto en la cabeza.

Eso sugiere que el resultado depende del tamaño del corpus más que de la técnica.
Con miles de chunks, un top-50 sería una preselección genuina y el cross-encoder
trabajaría sobre candidatos plausibles, que es el escenario para el que está
pensado. Antes de descartarlo repetiría la medición con dos cambios: bajar la
profundidad de recall a 15 o 20, y ampliar el golden set — cinco consultas dan
pasos de 0.20 en P@5, demasiado gruesos para diferencias finas.

Conclusión operativa: **B ahora, y reevaluar el reranking cuando el corpus
crezca**, con la medición como criterio y no la expectativa.

## Reproducir

```bash
docker compose up -d
docker compose exec estimator alembic upgrade head
uv run python scripts/ingest_budgets.py data/budgets_extended.json
uv run python scripts/evaluate_retrieval.py
```

Resultados completos en `evals/retrieval_results.md`.
