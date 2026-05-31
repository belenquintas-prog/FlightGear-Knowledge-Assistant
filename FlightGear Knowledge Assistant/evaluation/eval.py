import os
import requests
import json
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric
)

load_dotenv("evaluation.env")

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
WEBHOOK_URL   = os.getenv("WEBHOOK_URL")
URL_DOCS      = os.getenv("SUPABASE_URL_DOCS")
KEY_DOCS      = os.getenv("SUPABASE_KEY_DOCS")
URL_KB        = os.getenv("SUPABASE_URL_KB")
KEY_KB        = os.getenv("SUPABASE_KEY_KB")

# ── Banco de pruebas ──────────────────────────────────────────────────────────
test_cases_raw = [
    # ── HISTORICAL DATA SEARCH ────────────────────────────────────────────────
    {
        "input": "¿Ha habido alguna incidencia con el piloto automático de la Cessna 172P?",
        "category": "historical_data search",
        "table": "kb",
        "expected_output": (
            "Historical Summary: SR-2024-004 documenta inestabilidad del controlador PID del piloto automático "
            "en modo altitud en la Cessna 172P cuando la velocidad vertical supera 500 ft/min en el momento de activación.\n"
            "Recurring Patterns: Activación del piloto automático en condiciones de alta tasa de ascenso sin estabilización previa.\n"
            "Typical Resolutions: Estabilizar la aeronave en vuelo nivelado con velocidad vertical inferior a 200 ft/min "
            "antes de activar el modo altitud. Usar property tree /autopilot/settings/target-altitude-ft como workaround.\n"
            "Relevance to Current Case: Alta. El patrón coincide directamente con BUG-031.\n"
            "Confidence Level: 0.95"
        )
    },
    {
        "input": "¿Qué problemas conocidos hay con el ILS en FlightGear?",
        "category": "historical_data search",
        "table": "kb",
        "expected_output": (
            "Historical Summary: SR-2024-006 documenta fallo de captura del localizador ILS en KLVK pista 25R. "
            "SR-2024-013 documenta indicación incorrecta del VOR durante tutorial IFR.\n"
            "Recurring Patterns: Confusión entre modos de navegación NAV y APR, y entre radiales FROM y TO.\n"
            "Typical Resolutions: Sintonizar NAV1 a 109.75 MHz, cambiar modo piloto automático a APR antes "
            "de aproximación final, verificar distancia inferior a 18 NM.\n"
            "Relevance to Current Case: Alta para aproximaciones ILS. Media para navegación VOR.\n"
            "Confidence Level: 0.9"
        )
    },
    {
        "input": "¿Hay registros de problemas de rendimiento gráfico en Windows?",
        "category": "historical_data search",
        "table": "kb",
        "expected_output": (
            "Historical Summary: SR-2024-001 documenta rendimiento inferior a 1 FPS en Windows 11 "
            "con tarjeta Intel UHD integrada por falta de soporte OpenGL certificado.\n"
            "Recurring Patterns: Hardware integrado sin soporte OpenGL por hardware fuerza renderizado por software.\n"
            "Typical Resolutions: Instalar drivers OpenGL del fabricante. Reducir resolución y desactivar "
            "nubes 3D con --disable-clouds3d y --fog-fastest.\n"
            "Relevance to Current Case: Alta para sistemas Windows con GPU integrada.\n"
            "Confidence Level: 0.95"
        )
    },
    {
        "input": "¿Qué incidencias hay relacionadas con el joystick en Linux?",
        "category": "historical_data search",
        "table": "kb",
        "expected_output": (
            "Historical Summary: SR-2024-005 documenta joystick no reconocido en Ubuntu 24.04 "
            "por cambio de permisos del grupo input.\n"
            "Recurring Patterns: Usuario sin permisos del grupo input en versiones recientes de Ubuntu.\n"
            "Typical Resolutions: sudo usermod -aG input $USER, cerrar sesión y reiniciar. "
            "Verificar con ls -la /dev/input/js0.\n"
            "Relevance to Current Case: Alta para Ubuntu 24.04 y distribuciones recientes.\n"
            "Confidence Level: 0.95"
        )
    },

    # ── PROBLEM ANALYSIS ──────────────────────────────────────────────────────
    {
        "input": "El helicóptero Bo105 pierde sustentación bruscamente durante el descenso vertical y no puedo recuperarlo.",
        "category": "problem_analysis",
        "table": "kb",
        "expected_output": (
            "Reported Behavior: El helicóptero Bo105 pierde sustentación bruscamente durante el descenso "
            "vertical y no es posible recuperar el control.\n"
            "Expected Behavior: El modelo YASim del Bo105 simula la condición de anillo de vórtice pero "
            "permite recuperación si se dispone de altitud suficiente y se gana velocidad de avance.\n"
            "Identified Deviation: Descenso vertical superior a 800 ft/min sin velocidad de avance activa "
            "la condición de anillo de vórtice según BUG-073.\n"
            "Historical Alignment: SR-2024-009 documenta exactamente este comportamiento.\n"
            "Technical Hypothesis: La pérdida de sustentación es consecuencia directa de superar el umbral "
            "de 800 ft/min en descenso vertical puro. La recuperación es imposible por debajo de 500 ft AGL.\n"
            "Uncertainties: Sin conocer la altitud AGL en el momento del fallo no es posible determinar "
            "si la recuperación era viable.\n"
            "Confidence Level: 0.92"
        )
    },
    {
        "input": "La Cessna entra en barrena cuando intento practicar maniobras de pérdida y no sé cómo recuperarla.",
        "category": "problem_analysis",
        "table": "kb",
        "expected_output": (
            "Reported Behavior: La Cessna 172P entra en barrena durante maniobras de práctica de pérdida "
            "y no es posible recuperar el control.\n"
            "Expected Behavior: El modelo JSBSim de la C172P permite recuperación de barrena siguiendo "
            "el procedimiento correcto de reducción de ángulo de ataque y aplicación de timón contrario.\n"
            "Identified Deviation: Aplicación simultánea de alerón y timón de cola durante la pérdida "
            "induce la barrena según BUG-119.\n"
            "Historical Alignment: SR-2024-015 documenta exactamente este comportamiento en el modelo JSBSim.\n"
            "Technical Hypothesis: La pérdida diferencial entre alas combinada con el par del timón induce "
            "la barrena. El modelo aerodinámico simula este comportamiento de forma realista.\n"
            "Uncertainties: Sin conocer la altitud AGL no es posible determinar si la recuperación era viable.\n"
            "Confidence Level: 0.93"
        )
    },

    # ── DOCUMENTATION QUERY ───────────────────────────────────────────────────
    {
        "input": "¿Cómo se configura la variable FG_ROOT en Windows?",
        "category": "documentation_query",
        "table": "docs",
        "expected_output": (
            "Functional Explanation: FG_ROOT es la variable de entorno que indica a FlightGear la ubicación "
            "del directorio de datos del simulador. Sin ella correctamente configurada el simulador no puede "
            "localizar sus ficheros y termina silenciosamente.\n"
            "Configuration / Limits: Se define en autoexec.bat con la sintaxis set FG_ROOT=C:\\FlightGear\\data, "
            "o mediante el fichero fgfsrc, o desde el lanzador fgrun.exe.\n"
            "Relevant Documentation References: Capítulo 4, sección 4.2.1 FG_ROOT del Manual de FlightGear.\n"
            "Confidence Level: 0.95"
        )
    },
    {
        "input": "¿Qué modelos de dinámica de vuelo soporta FlightGear?",
        "category": "documentation_query",
        "table": "docs",
        "expected_output": (
            "Functional Explanation: FlightGear soporta múltiples modelos de dinámica de vuelo. JSBSim es el "
            "modelo principal para aeronaves de ala fija con simulación aerodinámica detallada. YASim es el "
            "modelo alternativo usado en helicópteros y algunas aeronaves de ala fija.\n"
            "Configuration / Limits: El modelo se selecciona por aeronave en el fichero de configuración XML. "
            "No es configurable en tiempo de ejecución.\n"
            "Relevant Documentation References: Capítulo 2, sección 2.4 Modelos de dinámica de vuelo "
            "del Manual de FlightGear.\n"
            "Confidence Level: 0.9"
        )
    },

    # ── CASO NEGATIVO ─────────────────────────────────────────────────────────
    {
        "input": "¿Cuál es el precio de la licencia de FlightGear?",
        "category": "documentation_query",
        "table": "docs",
        "expected_output": (
            "Functional Explanation: La documentación recuperada no contiene información sobre precios "
            "o licencias comerciales de FlightGear.\n"
            "Configuration / Limits: No aplica.\n"
            "Relevant Documentation References: No se han encontrado referencias relevantes en la "
            "documentación disponible.\n"
            "Confidence Level: 0.1"
        )
    },
]


def get_embedding(text):
    """Genera el embedding de un texto usando el mismo modelo que N8N."""
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


def get_chunks(query, table):
    """Recupera los chunks de Supabase para un query dado."""
    if table == "kb":
        url = f"{URL_KB}/rest/v1/rpc/match_documents"
        key = KEY_KB
    else:
        url = f"{URL_DOCS}/rest/v1/rpc/match_documents"
        key = KEY_DOCS

    embedding = get_embedding(query)

    response = requests.post(url, headers={
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }, json={
        "query_embedding": embedding,
        "match_count": 15
    })

    data = response.json()
    if isinstance(data, list):
        return [item["content"] for item in data if "content" in item]
    return []


def call_webhook(query, category):
    """Llama al webhook de N8N y devuelve la respuesta del agente."""
    response = requests.post(WEBHOOK_URL, json={
        "query": query,
        "query_category": category
    })
    data = response.json()
    return data.get("output", "Sin respuesta")


# ── Generar casos de prueba ───────────────────────────────────────────────────
print("=" * 60)
print("FlightGear Knowledge Assistant — Evaluación RAG")
print("=" * 60)
print()

test_cases = []
for i, case in enumerate(test_cases_raw):
    print(f"[{i+1}/{len(test_cases_raw)}] {case['input'][:60]}...")

    actual_output     = call_webhook(case["input"], case["category"])
    retrieval_context = get_chunks(case["input"], case["table"])

    print(f"    OK Respuesta recibida ({len(actual_output)} chars)")
    print(f"    OK Chunks recuperados: {len(retrieval_context)}")

    test_cases.append(LLMTestCase(
        input=case["input"],
        actual_output=actual_output,
        expected_output=case["expected_output"],
        retrieval_context=retrieval_context
    ))

print()
print("Ejecutando evaluación con DeepEval...")
print()

# ── Métricas ──────────────────────────────────────────────────────────────────
metrics = [
    FaithfulnessMetric(threshold=0.7,       model="gpt-4o-mini", verbose_mode=True),
    AnswerRelevancyMetric(threshold=0.7,    model="gpt-4o-mini", verbose_mode=True),
    ContextualPrecisionMetric(threshold=0.7, model="gpt-4o-mini", verbose_mode=True),
    ContextualRecallMetric(threshold=0.7,   model="gpt-4o-mini", verbose_mode=True),
]

results = evaluate(test_cases, metrics)

# ── Ejecutar evaluación ─────────────────────────────────────────────────────
results = evaluate(test_cases, metrics)

# ── Generar HTML ────────────────────────────────────────────────────────────
from datetime import datetime
import webbrowser

html = """
<!DOCTYPE html>
<html lang="es">

<head>
<meta charset="UTF-8">
<title>FlightGear RAG Evaluation</title>

<style>

body {
    font-family: Arial, sans-serif;
    max-width: 1200px;
    margin: auto;
    padding: 20px;
    background: #f5f7fa;
}

h1 {
    color: #1a1a2e;
}

.case {
    background: white;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 25px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}

.metric {
    padding: 10px;
    margin: 10px 0;
    border-radius: 6px;
    background: #fafafa;
}

.pass {
    color: green;
    font-weight: bold;
}

.fail {
    color: red;
    font-weight: bold;
}

.score {
    font-size: 18px;
    font-weight: bold;
}

.reason {
    color: #555;
    margin-top: 5px;
}

.summary {
    background: white;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 30px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}

table {
    width: 100%;
    border-collapse: collapse;
}

th, td {
    border: 1px solid #ddd;
    padding: 12px;
}

th {
    background: #1a1a2e;
    color: white;
}

pre {
    white-space: pre-wrap;
    word-wrap: break-word;
    background: #f4f4f4;
    padding: 12px;
    border-radius: 6px;
}

</style>
</head>

<body>
"""

html += f"""
<h1>FlightGear Knowledge Assistant — Evaluación RAG</h1>
<p><strong>Fecha:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
"""

# ── Resumen de métricas ─────────────────────────────────────────────────────
metric_summary = {}

for test_result in results.test_results:

    for metric_data in test_result.metrics_data:

        name = metric_data.name

        if name not in metric_summary:
            metric_summary[name] = []

        metric_summary[name].append(metric_data.score)

html += """
<div class="summary">

<h2>Resumen de métricas</h2>

<table>

<tr>
<th>Métrica</th>
<th>Score medio</th>
<th>Pass rate</th>
</tr>
"""

for name, scores in metric_summary.items():

    avg = sum(scores) / len(scores)

    pass_rate = (
        sum(1 for s in scores if s >= 0.7)
        / len(scores)
        * 100
    )

    html += f"""
    <tr>
        <td>{name}</td>
        <td>{avg:.2f}</td>
        <td>{pass_rate:.1f}%</td>
    </tr>
    """

html += """
</table>
</div>
"""

# ── Casos individuales ──────────────────────────────────────────────────────
for idx, test_result in enumerate(results.test_results):

    passed = all(
        (m.score or 0) >= 0.7
        for m in test_result.metrics_data
    )

    status = "PASS" if passed else "FAIL"

    css = "pass" if passed else "fail"

    html += f"""
    <div class="case">

        <h2>
            <span class="{css}">
                [{status}]
            </span>
            Caso {idx+1}
        </h2>

        <p><strong>Input:</strong></p>

        <pre>{test_result.input}</pre>

        <p><strong>Actual Output:</strong></p>

        <pre>{test_result.actual_output}</pre>

        <p><strong>Expected Output:</strong></p>

        <pre>{test_result.expected_output}</pre>

        <h3>Métricas</h3>
    """

    for metric in test_result.metrics_data:

        score = metric.score or 0

        metric_css = "pass" if score >= 0.7 else "fail"

        html += f"""
        <div class="metric">

            <div class="score {metric_css}">
                {metric.name}: {score:.2f}
            </div>

            <div class="reason">
                {metric.reason or 'Sin explicación'}
            </div>

        </div>
        """

    html += "</div>"

html += """
</body>
</html>
"""

# ── Guardar HTML ────────────────────────────────────────────────────────────
with open("resultados.html", "w", encoding="utf-8") as f:
    f.write(html)

print("\nHTML generado correctamente: resultados.html")

# ── Abrir automáticamente ───────────────────────────────────────────────────
webbrowser.open("resultados.html")