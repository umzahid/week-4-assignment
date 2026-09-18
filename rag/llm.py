import ollama

MODEL = "llama3.2:1b"


class OllamaError(Exception):
    pass


def generate(prompt: str, system: str = "") -> str:
    try:
        response = ollama.generate(
            model=MODEL, prompt=prompt, system=system, options={"temperature": 0}
        )
    except Exception as exc:
        raise OllamaError(
            f"Could not reach Ollama or model '{MODEL}' isn't available. "
            f"Run 'ollama serve' and 'ollama pull {MODEL}'. Details: {exc}"
        ) from exc
    return response["response"]
