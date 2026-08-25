# Generation evaluation (RAGAS) — Session 11

Judge: `gpt-4o-mini`. Embeddings: `text-embedding-3-small`. Retrieval: hybrid (vector + lexical, RRF), top-k 5, no reranking — config B, the best P@5 on the Session 10 golden set.

## Metrics

| Query | Faithfulness | Answer relevancy | Context precision | Context recall |
| --- | --- | --- | --- | --- |
| Q1 | 0.82 | 0.58 | 1.00 | 0.44 |
| Q2 | 0.75 | 0.00 | 1.00 | 0.83 |
| Q3 | 0.69 | 0.00 | 1.00 | 0.33 |
| Q4 | 0.31 | 0.45 | 1.00 | 0.67 |
| Q5 | 0.69 | 0.20 | 0.92 | 0.57 |
| **Media** | 0.65 | 0.25 | 0.98 | 0.57 |

## Citation verification

| Query | Líneas | Fundamentadas | Colgantes | Sin datos | Horas | Horas ref. |
| --- | --- | --- | --- | --- | --- | --- |
| Q1 | 2 | 2 | 0 | 0 | 400 | 1020 |
| Q2 | 1 | 1 | 0 | 0 | 180 | 400 |
| Q3 | 3 | 1 | 0 | 2 | 120 | 720 |
| Q4 | 2 | 2 | 0 | 0 | 400 | 440 |
| Q5 | 2 | 2 | 0 | 0 | 400 | 610 |

## Nota sobre los números

Lo que más chirría es el **answer relevancy medio de 0,25, con dos ceros absolutos (Q2 y Q3)**: las
respuestas son claramente pertinentes, pero RAGAS anula la métrica cuando su juez marca la respuesta
como *noncommittal*, y nuestra política de citación obliga precisamente a decir "no hay datos
suficientes". La honestidad que pide el enunciado se penaliza como evasión, así que este 0,25 mide el
choque entre dos criterios, no la calidad real de la generación.

El segundo problema es el **context recall de 0,57**, muy bajo en Q3 (0,33) y Q1 (0,44): con top-k 5
sobre presupuestos que aportan 2 chunks cada uno, no caben todos los componentes que la referencia
espera. Se ve en las horas: Q1 estima 400 h frente a 1.020 h de referencia y Q3 120 h frente a 720 h,
no por inventar de menos sino porque los chunks de pago, escaparate o cumplimiento nunca llegaron al
generador. La precisión es casi perfecta (0,98): lo que recuperamos es relevante, pero recuperamos
poco.

**Faithfulness 0,65** es el número más honesto y el más mejorable — sobre todo Q4 (0,31), donde el
modelo justifica una cita correcta con prosa que va más allá de lo que el chunk dice literalmente.
Ninguna de las 10 líneas generadas produjo una citación colgante, así que el fallo no es inventarse
fuentes, sino razonar de más sobre fuentes reales.
