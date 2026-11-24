import streamlit as st
from openai import OpenAI
import os

# Leer GROQ_API_KEY (soporta st.secrets y variables de entorno)
try:
    api_key = st.secrets.get("GROQ_API_KEY")
except Exception:
    api_key = None

if not api_key:
    api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("No se encontró GROQ_API_KEY en st.secrets ni en las variables de entorno")

# ¡Punto clave! Especificar la base_url de Groq
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=api_key
)

PROMPTS_AUDIENCIA = {
    "Dirección": (
        "Eres analista de riesgo senior. Explica la decisión de crédito a la dirección en 2-3 frases. "
        "Enfócate en riesgos, impacto y criterios de aprobación. Sin jerga técnica excesiva."
    ),
    "Negocio": (
        "Eres analista de negocio. Explica la decisión al área comercial en 3-4 frases. "
        "Indica la lógica general y las 2-3 variables más relevantes. Sugiere cómo integrar el resultado."
    ),
    "Técnico": (
        "Eres científico de datos. Explica al equipo técnico en 4-5 frases. "
        "Incluye probabilidad, contribuciones SHAP y el tipo de modelo. Sé preciso con los números."
    ),
    "Persona afectada": (
        "Eres asesor de atención al cliente. Explica a la persona solicitante en 3-4 frases, lenguaje claro, sin jerga. "
        "Indica qué factores influyeron, qué dato puede corregir y dónde solicitar revisión. "
        "No uses 'SHAP' ni 'probabilidad'."
    ),
}


def generar_explicacion(resultado: dict, audiencia: str) -> str:
    system_prompt = PROMPTS_AUDIENCIA[audiencia]
    evidencia = {
        "decision": resultado["decision"],
        "probabilidad": resultado["probabilidad"],
        "contribuciones_shap": resultado["contribuciones_shap"],
        "datos_solicitud": resultado["datos_solicitud"],
        "modelo": resultado["modelo"],
        "modelo_tipo": resultado["modelo_tipo"],
    }
    user_prompt = (
        "Genera la explicación basándote EXCLUSIVAMENTE en esta evidencia. "
        "No inventes razones no respaldadas por las contribuciones SHAP.\n\n"
        f"Evidencia:\n{json.dumps(evidencia, indent=2, ensure_ascii=False)}"
    )

    # Modelo gratuito de Groq: Llama 3.3 70B (buen equilibrio calidad/velocidad)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=400,
    )
    return response.choices[0].message.content.strip()


def validar_fidelidad(explicacion: str, resultado: dict) -> dict:
    variables = list(resultado["contribuciones_shap"].keys())
    texto = explicacion.lower().replace("_", " ")
    mencionadas = [v for v in variables if v.lower().replace("_", " ") in texto]
    return {
        "variables_reales": variables,
        "variables_mencionadas": mencionadas,
        "fidelidad_ok": len(mencionadas) > 0,
        "advertencia": None if mencionadas else "La explicación no menciona ninguna variable del modelo.",
    }


def ejecutar_agente(solicitud: dict, audiencia: str, nombre_modelo: str) -> dict:
    from model_utils import evaluar_solicitud

    resultado = evaluar_solicitud(solicitud, nombre_modelo)
    explicacion = generar_explicacion(resultado, audiencia)
    validacion = validar_fidelidad(explicacion, resultado)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "audiencia": audiencia,
        "solicitud": solicitud,
        "decision": resultado["decision"],
        "probabilidad": resultado["probabilidad"],
        "contribuciones_shap": resultado["contribuciones_shap"],
        "explicacion_generada": explicacion,
        "validacion": validacion,
        "modelo": resultado["modelo"],
        "modelo_tipo": resultado["modelo_tipo"],
        "version_modelo": resultado["version_modelo"],
        "fuentes": resultado["fuentes"],
        "hash_evidencia": resultado["hash_evidencia"],
        "modelo_llm": "llama-3.3-70b-versatile",
        "feedback": None,
        "apelacion": None,
    }
