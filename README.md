# FlightGear-Knowledge-Assistant

Asistente Multiagente de Gestión del Conocimiento para Soporte Técnico Aeronáutico

## Descripción

FlightGear-Knowledge-Assistant es un asistente basado en Inteligencia Artificial Generativa diseñado para apoyar a ingenieros de soporte técnico en la resolución de incidencias relacionadas con software de simulación de vuelo.

La solución combina una arquitectura multiagente, técnicas de Retrieval Augmented Generation (RAG) y búsqueda semántica para facilitar el acceso a documentación técnica e incidencias históricas.

## Demo

Aplicación desplegada:

https://iaproject-pi.vercel.app?_vercel_share=G3wzYxFTqXq1SWvS04LK16FwrgzMDRcA

## Problema abordado

Los equipos de soporte técnico gestionan información distribuida entre múltiples fuentes:

- Incidencias históricas
- Documentación técnica
- Correos electrónicos
- Sistemas de seguimiento de errores

Esta fragmentación dificulta la reutilización del conocimiento acumulado y aumenta el tiempo necesario para diagnosticar problemas ya resueltos anteriormente.

FlightGear-Knowledge-Assistant busca centralizar dicho conocimiento y proporcionar respuestas contextualizadas basadas en información verificable.


### Flujo de procesamiento

1. Recepción de la consulta.
2. Validación mediante guardrails.
3. Clasificación por tipo de consulta.
4. Generación de estrategia por un agente especializado.
5. Recuperación de información mediante RAG.
6. Generación de respuesta final.
7. Registro de trazas mediante Langfuse.

![Arquitectura](./FlightGear%20Knowledge%20Assistant/docs/images/architecture.png)

## Funcionalidades principales

### Arquitectura multiagente

El sistema incorpora tres agentes especializados:

- Historical Search
- Problem Analysis
- Documentation Query

y un agente final encargado de generar la respuesta utilizando la información recuperada.

### Recuperación aumentada mediante RAG

El sistema consulta dos fuentes de conocimiento independientes:

- Base histórica de incidencias.
- Documentación técnica vectorizada.

### Seguridad

Se han implementado mecanismos de protección frente a:

- Jailbreak
- Prompt Injection
- Contenido NSFW
- Información personal identificable (PII)
- Consultas fuera del dominio funcional

### Evaluación

La calidad del sistema se ha evaluado mediante DeepEval utilizando las métricas:

- Faithfulness
- Answer Relevancy
- Contextual Precision
- Contextual Recall

### Monitorización

La solución incorpora trazabilidad de consultas y respuestas mediante Langfuse.

## Stack tecnológico

| Componente | Tecnología |
|------------|------------|
| Orquestación | n8n |
| Modelos LLM | GPT-4o, GPT-4.1-mini, GPT-4.1-nano |
| Embeddings | text-embedding-3-small |
| Base vectorial | Supabase + pgvector |
| Almacenamiento documental | Google Drive |
| Frontend | HTML, CSS y JavaScript |
| Despliegue | Vercel |
| Evaluación | DeepEval |
| Observabilidad | Langfuse |
| Control de versiones | GitHub |

## Resultados de evaluación

| Métrica | Resultado |
|----------|----------|
| Faithfulness | 0.95 |
| Answer Relevancy | 0.88 |
| Contextual Precision | 0.45 |
| Contextual Recall | 0.81 |

Los resultados muestran una elevada fidelidad al contexto recuperado y una buena capacidad de respuesta, identificándose la precisión del retrieval como principal línea de mejora.

## Estructura del repositorio

```text
FlightGear-Knowledge-Assistant/
│
├── README.md
│
├── docs/
│   ├── memoria.pdf
│   ├── deepeval-results.html
│   └── images/
│
├── evaluation/
│
└── workflows/
```

## Líneas de evolución

- Uso de modelos open-source self-hosted.
- Despliegue completamente on-premise.
- Incorporación de métricas avanzadas de observabilidad.
- Monitorización de costes y latencias.
- Optimización del retrieval mediante técnicas de re-ranking.
- Evaluación continua del sistema.

## Autora

**Belén Quintas Herraiz**

Máster en IA Generativa (2025-2026)

Proyecto Final
