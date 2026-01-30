from openai import OpenAI
import os
from .config import DATABRICKS_TOKEN, DATABRICKS_HOST, DATABRICKS_MODEL_ENDPOINT


# Construct the proper serving endpoint URL
if DATABRICKS_HOST and DATABRICKS_MODEL_ENDPOINT:
    base_url = f"{DATABRICKS_HOST.rstrip('/')}/serving-endpoints"
else:
    base_url = "https://adb-5732085104630262.2.azuredatabricks.net/serving-endpoints"


client = OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url=base_url
)


def generate_answer(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model="databricks-meta-llama-3-1-8b-instruct",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert information extraction assistant. Extract and provide answers from the given context. Never refuse to answer if the information exists in the context, regardless of how the question is phrased. Only say you don't know when the information is genuinely absent."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=1024,
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        error_msg = str(e)

        # If Databricks token is invalid, use fallback mode
        if "403" in error_msg or "Invalid access token" in error_msg:
            print("\n Databricks token invalid. Using fallback mode (context only).\n")
            return generate_fallback_answer(prompt)

        return f"Error generating answer: {error_msg}"


def generate_fallback_answer(prompt: str) -> str:
    """Simple fallback when Databricks is unavailable - extracts context from prompt"""
    # Extract the context section from the prompt
    if "CONTEXT:" in prompt and "QUESTION:" in prompt:
        context_start = prompt.find("CONTEXT:") + len("CONTEXT:")
        question_start = prompt.find("QUESTION:")
        context = prompt[context_start:question_start].strip()
        question = prompt[question_start + len("QUESTION:"):].replace("ANSWER:", "").strip()

        # Return the full context with key information highlighted
        lines = context.split('\n')
        relevant_lines = [line for line in lines if line.strip()][:10]  # First 10 non-empty lines
        return "Based on the documents:\n\n" + "\n".join(relevant_lines)

    return "Unable to generate answer. Please check your Databricks credentials."