
from google import genai
from src.config import settings


def test_gemini():

    
    if not settings.api_key:
        raise ValueError("GEMINI_API_KEY is missing in .env file")

   
    client = genai.Client(
        api_key=settings.api_key
    )

    
    response = client.models.generate_content(
        model=settings.model,
        contents="Hello Gemini! Please introduce yourself."
    )

    
    print("\nGemini Response:")
    print(response.text)


if __name__ == "__main__":
    test_gemini()
