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

| Query | Líneas | Con fuente | Chunk inexistente | Sin datos | Horas | Horas ref. |
| --- | --- | --- | --- | --- | --- | --- |
| Q1 | 2 | 2 | 0 | 0 | 400 | 1020 |
| Q2 | 1 | 1 | 0 | 0 | 180 | 400 |
| Q3 | 3 | 1 | 0 | 2 | 120 | 720 |
| Q4 | 2 | 2 | 0 | 0 | 400 | 440 |
| Q5 | 2 | 2 | 0 | 0 | 400 | 610 |

## Nota sobre los números

Corriendo los experimentos me llamó la atención el answer relevancy: 0,25 de media y dos consultas
en 0,00. Fui a mirar esas respuestas y estaban bien, hablaban justo de lo que se les preguntaba.
Revisando el código de RAGAS creo que la razón es que el juez las marca como noncommittal, porque
empiezan diciendo de qué no tienen datos, y ahí la métrica se va directa a cero. Si es eso, me
imagino que le pasará a cualquiera que haya implementado la política de insufficient context.

El otro que me llamó la atención fue el context recall, 0,57. Ese sí me parece un problema mío de
configuración: con top-k 5 y presupuestos que aportan 2 chunks cada uno no entran todos los
componentes, y por eso Q1 me da 400 h cuando la referencia son 1.020 h, los chunks de pago y
escaparate directamente nunca le llegaron al generador. La precisión en cambio es 0,98, así que lo
poco que recupero es relevante.

El faithfulness de 0,65 es lo que más margen tiene, sobre todo Q4 con 0,31, donde el modelo cita
bien pero luego se va por las ramas justificándolo. Ninguna línea citó un chunk que no existiera,
así que el problema no es inventarse fuentes, es razonar de más sobre fuentes reales.
