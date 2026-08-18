# Retrieval evaluation — Session 10

Golden set: 5 queries, criterion `functional_domain`. top-k = 5, recall depth for reranking = 50.

## Comparative table

| Config | Búsqueda | Reranking | P@5 medio | Latencia media (ms) | Latencia mediana (ms) |
| ------ | -------- | --------- | --------- | ------------------- | --------------------- |
| A | Vectorial | No | 0.76 | 665 | 514 |
| B | Híbrida | No | 0.84 | 563 | 510 |
| C | Vectorial | Sí | 0.68 | 802 | 627 |
| D | Híbrida | Sí | 0.72 | 637 | 643 |

## Per-query detail

| Config | Query | P@5 | Latencia (ms) | Candidatos |
| --- | --- | --- | --- | --- |
| A | Q1 | 0.80 | 1335 | 5 |
| A | Q2 | 0.60 | 480 | 5 |
| A | Q3 | 0.80 | 519 | 5 |
| A | Q4 | 0.80 | 477 | 5 |
| A | Q5 | 0.80 | 514 | 5 |
| B | Q1 | 1.00 | 513 | 8 |
| B | Q2 | 0.80 | 510 | 6 |
| B | Q3 | 0.80 | 493 | 7 |
| B | Q4 | 0.80 | 807 | 6 |
| B | Q5 | 0.80 | 493 | 9 |
| C | Q1 | 0.80 | 1507 | 50 |
| C | Q2 | 0.40 | 663 | 50 |
| C | Q3 | 0.80 | 626 | 50 |
| C | Q4 | 0.60 | 627 | 50 |
| C | Q5 | 0.80 | 588 | 50 |
| D | Q1 | 0.80 | 632 | 50 |
| D | Q2 | 0.60 | 643 | 51 |
| D | Q3 | 0.80 | 646 | 52 |
| D | Q4 | 0.60 | 612 | 54 |
| D | Q5 | 0.80 | 653 | 52 |
