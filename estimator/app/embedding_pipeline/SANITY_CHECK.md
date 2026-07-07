# Sanity check — text-embedding-3-small

Generated with `scripts/compare.py` against the three required pairs.

| Pair | Text A | Text B | Cosine similarity |
|---|---|---|---|
| A (close) | OAuth 2.0 authentication backend with JWT tokens for fintech mobile app | Authorization service using JSON Web Tokens for a banking application | **0.5957** |
| B (unrelated) | OAuth 2.0 authentication backend with JWT tokens for fintech mobile app | Database migration from MySQL to PostgreSQL with zero downtime | **0.1920** |
| C (generic) | Backend services | API development | **0.5407** |

## Comentario

El pipeline discrimina en la dirección correcta: la pareja B (temas no
relacionados) cae muy por debajo de la A (mismo dominio, vocabulario
distinto), con un margen amplio (0.60 vs 0.19). Lo que llama la atención es
que la pareja A queda justo por debajo del umbral orientativo de 0.6 pese a
ser semánticamente la misma idea (auth OAuth/JWT) — probablemente porque los
dos textos comparten pocas palabras literales ("OAuth", "JWT" vs "Authorization
Service", "JSON Web Tokens") y el modelo pondera bastante la superficie léxica,
no solo el significado. La pareja C (términos genéricos y cortos) da una
similitud media-alta (0.54), casi tan alta como la pareja A "cercana" — con
frases tan cortas y ambiguas el embedding tiene poca señal semántica real de
la que agarrarse, así que el resultado dice más sobre la brevedad del texto
que sobre una relación de significado fuerte. Esto es exactamente el tipo de
caso límite a discutir en el directo: los umbrales orientativos son eso,
orientativos.
