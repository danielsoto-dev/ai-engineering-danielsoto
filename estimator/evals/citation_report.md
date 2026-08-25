# Citation verification report

Consulta `Q1`: Necesitamos una tienda online con catálogo de productos, carrito de la compra y proceso de pago para vender ropa por internet

## Estimación generada

```
El proyecto de tienda online para vender ropa por internet incluye los componentes de catálogo de productos y carrito de la compra con proceso de pago. Se puede proporcionar estimaciones basadas en proyectos similares en el contexto proporcionado.

- Catálogo de productos: 220 h. El servicio de catálogo de productos requiere una implementación de alta complejidad similar a la descrita en el proyecto de una plataforma de comercio electrónico headless. [fuentes: 8]
- Carrito de la compra y proceso de pago: 180 h. El carrito de la compra y proceso de checkout sigue la especificación de alta complejidad documentada en un proyecto de comercio electrónico comparable. [fuentes: 9]

Total: 400 horas.
```

## Verificación real

Chunks entregados al LLM: `['4', '8', '9', '11', '16']`

| Línea | Estado | Cita | Colgantes |
| --- | --- | --- | --- |
| Catálogo de productos | `grounded` | 8 | — |
| Carrito de la compra y proceso de pago | `grounded` | 9 | — |

Resumen: `{'lines': 2, 'grounded': 2, 'dangling': 0, 'insufficient_context': 0, 'grounding_rate': 1.0, 'is_valid': True}`

## Verificación con una citación colgante inyectada (`9999`)

Chunks entregados al LLM: `['4', '8', '9', '11', '16']`

| Línea | Estado | Cita | Colgantes |
| --- | --- | --- | --- |
| Catálogo de productos | `dangling` | 9999 | 9999 |
| Carrito de la compra y proceso de pago | `grounded` | 9 | — |

Resumen: `{'lines': 2, 'grounded': 1, 'dangling': 1, 'insufficient_context': 0, 'grounding_rate': 0.5, 'is_valid': False}`

