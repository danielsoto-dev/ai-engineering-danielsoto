# Estimator CAG - Servicio de Estimacion de Software con IA

Servicio de estimacion de proyectos de software impulsado por IA, utilizando una arquitectura **Cache Augmented Generation (CAG)**.

## Que es CAG y por que lo usamos

CAG (Cache Augmented Generation) es un patron de arquitectura donde el contexto relevante se inyecta directamente en el prompt del LLM como texto estatico. En esta fase del proyecto, las estimaciones de referencia se incluyen como ejemplos dentro del prompt del sistema, sin necesidad de una base de datos vectorial ni busqueda semantica.

Este enfoque es ideal para empezar porque:
- Es simple de implementar y depurar
- No requiere infraestructura adicional (ni embeddings, ni vector stores)
- Funciona bien cuando el volumen de contexto es manejable (pocos ejemplos)

En modulos posteriores del master, este servicio evolucionara a una arquitectura **RAG** (Retrieval Augmented Generation) con base de datos vectorial para manejar un volumen mayor de ejemplos.

## Requisitos previos

- **Docker** y **Docker Compose** instalados
- Una **API key** de OpenAI o Anthropic
- Python **NO** es necesario localmente — todo se ejecuta dentro del contenedor

## Inicio rapido con Docker (recomendado)

1. Clonar el repositorio y entrar al directorio:
   ```bash
   cd estimator
   ```

2. Copiar el archivo de variables de entorno y configurar las API keys:
   ```bash
   cp .env.example .env
   # Editar .env y poner tu API key real
   ```

3. Construir y levantar el servicio:
   ```bash
   docker compose up --build
   ```

4. El servicio estara disponible en `http://localhost:8000`

## Alternativa: ejecucion local sin Docker

```bash
uv sync
# Configurar .env con tus API keys
uv run uvicorn app.main:app --reload
```

## Probar el servicio

```bash
curl -X POST http://localhost:8000/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d '{
    "transcription": "The client wants to build a mobile app for managing restaurant reservations. They need user registration, a restaurant search with filters by cuisine and location, a real-time reservation system with availability checking, push notifications for reservation confirmations and reminders, and an admin panel for restaurant owners to manage their listings and view analytics."
  }'
```

## Estructura del proyecto

```
estimator/
├── app/
│   ├── main.py            # Aplicacion FastAPI, health check, CORS
│   ├── config.py           # Configuracion con Pydantic Settings
│   ├── routers/
│   │   └── estimations.py  # Endpoint POST /api/v1/estimate
│   ├── services/
│   │   └── llm_service.py  # Logica de negocio, llamadas al LLM
│   ├── schemas/
│   │   └── estimation.py   # Modelos Pydantic (request/response)
│   └── context/
│       └── examples.py     # Ejemplos de estimacion (contexto CAG)
├── tests/
│   └── test_health.py      # Tests basicos
├── Dockerfile              # Build multi-stage con uv
├── docker-compose.yml      # Configuracion para desarrollo local
└── pyproject.toml          # Dependencias y configuracion
```

## Documentacion interactiva

Con el servicio corriendo, accede a la documentacion Swagger UI en:

- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

## Sesion 3 — LiteLLM, Redis cache, SSE y Streamlit

A partir de la Sesion 3 el servicio incorpora una capa de wrapper sobre el LLM que anade:

- **Fallback de proveedor** (LiteLLM Router) — si el modelo primario falla, se intenta el secundario
- **Cache exact-match** en Redis — la misma transcripcion no vuelve a pagar tokens
- **Streaming SSE** — endpoint `POST /api/v1/estimate/stream` que emite los tokens segun llegan
- **UI Streamlit** — cliente real que consume el endpoint SSE

### Arrancar la stack completa

```bash
cd estimator
docker compose up --build
# La API queda en http://localhost:8000 y Redis en redis://localhost:6379
```

### Probar el endpoint SSE

Demo HTML: abrir [http://localhost:8000/static/sse_demo.html](http://localhost:8000/static/sse_demo.html).

Desde CLI:
```bash
curl -N -X POST http://localhost:8000/api/v1/estimate/stream \
  -H 'Content-Type: application/json' \
  -d '{"transcription": "We need a small CRM with auth, contacts and roles. MVP six weeks."}'
```

### Verificar la cache

```bash
# La misma peticion dos veces — la segunda devuelve cache_hit: true
curl -s localhost:8000/api/v1/estimate -H 'Content-Type: application/json' \
  -d '{"transcription": "We need a small CRM with auth, contacts and roles. MVP six weeks."}' \
  | jq '{cache_hit, cost_usd}'

# Inspeccionar las claves en Redis
docker compose exec redis redis-cli KEYS 'estimation:*'
```

### Streamlit

Streamlit corre **fuera** de Docker y consume el endpoint SSE por HTTP:

```bash
cd estimator
uv sync
uv run streamlit run streamlit_app.py
# Abrir http://localhost:8501
```

La URL del backend se lee de `ESTIMATOR_API_BASE_URL` (default `http://localhost:8000`).

---

## Sesion 7 (pre-exercise) — Embedding pipeline

Primer paso hacia RAG: convierte presupuestos historicos (JSON) en chunks
vectorizados con `text-embedding-3-small`.

- `app/embedding_pipeline/chunker.py` — un componente de presupuesto = un chunk.
- `app/embedding_pipeline/embedder.py` — llama a la API de OpenAI en batches (100 chunks/llamada).
- `app/embedding_pipeline/SANITY_CHECK.md` — similitud coseno sobre 3 parejas de prueba.

---

## Sesion 8 (pre-exercise) — Persistencia pgvector + busqueda semantica

El pipeline de la Sesion 7 pasa de ser en-memoria a persistir en PostgreSQL +
pgvector. `POST /embeddings/ingest` ahora persiste cada presupuesto como un
`document` con sus `chunks` (uno por componente) en una sola transaccion, y
un nuevo endpoint `POST /search` resuelve queries semanticas por distancia
coseno.

- `app/db/models.py` — modelos SQLAlchemy `Document` / `Chunk`.
- `alembic/` — migraciones (`0001_initial_schema.py` crea la extension `vector` + ambas tablas).
- `app/embedding_pipeline/router.py` — `POST /embeddings/ingest` (persistente) y `POST /search`.
- `scripts/query_examples.py` — ingesta el corpus de ejemplo y ejecuta 5 queries representativas (reemplaza `scripts/compare.py`).
- `output_examples.txt` — output real de `query_examples.py` contra `data/budgets_sample.json`.

### Levantar Postgres y migrar

```bash
docker compose up -d postgres
docker compose exec postgres psql -U estimator -d estimator -c "SELECT version();"

# Aplicar el esquema (extension vector + tablas documents/chunks)
uv run alembic upgrade head
```

> Nota: el `docker-compose.yml` de este repo mapea Postgres al puerto **5433**
> del host (`5433:5432`) para no chocar con un Postgres local ya corriendo en
> 5432. Otros servicios lo alcanzan en la red interna de Docker como
> `postgres:5432` sin cambios.

### Probar los endpoints

Con el servicio corriendo (`docker compose up --build` o `uv run uvicorn app.main:app --reload`):

```bash
# Ingesta (un presupuesto por llamada)
curl -s -X POST http://localhost:8000/embeddings/ingest \
  -H 'Content-Type: application/json' \
  -d "{\"source_path\": \"data/budgets_sample.json#BUD-2024-014\", \"document_type\": \"historical_budget\", \"content\": $(jq '.[0]' data/budgets_sample.json)}"

# Busqueda semantica
curl -s -X POST http://localhost:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "REST API with OAuth authentication for fintech sector", "k": 5}'
```

O directamente desde Swagger UI en `http://localhost:8000/docs`.

### `scripts/query_examples.py`

Ingesta los 3 presupuestos de `data/budgets_sample.json` (idempotente: un
409 en un re-run se trata como "ya ingerido" y se ignora) y luego ejecuta 5
queries que cubren angulos distintos del corpus (match directo,
reformulacion semantica, dominio no relacionado, query ambigua, vocabulario
tecnico especifico):

```bash
uv run python scripts/query_examples.py
# o dentro de Docker:
docker compose run --rm estimator python scripts/query_examples.py
```

### Decisiones de schema

**(a) Dos tablas (`documents` + `chunks`) en vez de una.** Un presupuesto
produce N chunks (uno por componente). Una tabla unica duplicaria la
metadata del documento en cada fila y perderia integridad referencial. Con
`chunks.document_id` + `ON DELETE CASCADE`, borrar un `Document` borra
automaticamente todos sus `Chunk` sin lógica adicional en la aplicacion.

**(b) `metadata` como JSONB en vez de columnas propias.** La metadata
estable (tipo de documento, tipo de chunk, fechas) vive en columnas
tipadas. La metadata variable — la que el chunker puede enriquecer con el
tiempo (sector, tecnologias mencionadas, tags) — vive en JSONB para no
requerir una migracion cada vez que cambia. El indice GIN sobre `metadata`
permite consultar por claves arbitrarias sin ese costo.

**(c) `cosine_distance` en vez de L2 o inner product.** Los embeddings de
`text-embedding-3-small` estan normalizados, asi que `cosine_distance` e
`inner_product` dan resultados equivalentes en el ranking. Se elige coseno
por ser la convencion mas comun en literatura RAG, y para que cuando se
añada el indice HNSW con `vector_cosine_ops` (sesion en vivo) el operador
de la query y la operator class del indice queden alineados — un mismatch
ahi hace que Postgres ignore el indice silenciosamente y caiga a sequential
scan.

**(d) Sin indice vectorial todavia.** Deliberado: con el volumen de este
ejercicio (unas pocas decenas de chunks) el sequential scan responde en
milisegundos, y este es exactamente el baseline contra el que la sesion en
vivo medira el impacto de HNSW/IVFFlat. Añadirlo ahora eliminaria ese punto
de comparacion.

## Sesión 12: agente de estimación

El agente manual está en `app/agent/`. Usa la Responses API para decidir qué componente buscar,
ejecuta `search_budgets` sobre el retrieval híbrido con reranking y calcula el resultado con una
función determinista. El bucle conserva cada `call_id`, devuelve los `function_call_output` y se
detiene cuando recibe la estimación estructurada o alcanza el máximo de iteraciones.

Prueba rápida con `gpt-5-mini` y el stub local:

```bash
uv run python scripts/run_agent_s12.py \
  exercises/session-12/sample_transcript_simple.txt \
  --model gpt-5-mini --effort minimal --stub
```

Ejecución completa con `gpt-5`, reasoning `medium` y el retrieval real:

```bash
uv run python scripts/run_agent_s12.py \
  exercises/session-12/sample_transcript_complex.txt \
  --model gpt-5 --effort medium \
  --output exercises/session-12/trace_complex.txt
```

La traza entregada está en `exercises/session-12/trace_complex.txt`.

> Este proyecto forma parte del **Master en AI Engineering** y servira como base para evolucionar hacia una arquitectura RAG con base de datos vectorial en modulos posteriores.
