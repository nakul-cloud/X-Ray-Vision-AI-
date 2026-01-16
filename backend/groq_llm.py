# grok_llm.py

from groq import Groq
import os


class GroqLLM:
    """
    Wrapper for GROQ LLM inference.
    Works with RAG pipeline and accepts custom prompts.
    """

    def __init__(self, model_name="openai/gpt-oss-120b"):
        self.api_key = os.getenv("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError("[ERROR] Missing GROQ_API_KEY in environment variables")

        self.model = model_name

        # Init client
        self.client = Groq(api_key=self.api_key)

        print(f"[INFO] Groq LLM loaded using model: {self.model}")

    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.1):
        """
        Generate a response using the chosen Groq model.
        Optimized for medical RAG (low temperature).
        ✔ Retries once on empty response
        ✔ Returns clear error messages
        """
        if not prompt or not prompt.strip():
             return "[ERROR] Empty prompt provided to LLM."

        retries = 1
        last_error = None
        
        for attempt in range(retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.9,
                )

                # Groq returns: response.choices[0].message.content
                content = response.choices[0].message.content
                
                if content and content.strip():
                    return content.strip()
                else:
                    print(f"[WARN] Groq returned empty content (Attempt {attempt+1}/{retries+1})")
                    
            except Exception as e:
                print(f"[ERROR] Groq LLM Error (Attempt {attempt+1}): {str(e)}")
                last_error = str(e)

        # If we failed after retries
        return f"[ERROR] Groq generation failed. Details: {last_error}"
