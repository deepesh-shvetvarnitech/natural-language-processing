import asyncio
import httpx

from src.config import settings


async def check_models():

    url = settings.base_url.rstrip("/") + "/models"

    headers = {
        "Authorization": f"Bearer {settings.api_key}"
    }

    async with httpx.AsyncClient(timeout=30) as client:

        response = await client.get(
            url,
            headers=headers
        )

        response.raise_for_status()

        data = response.json()

        available_models = [
            model["id"].replace("models/", "")
            for model in data["data"]
        ]

        print("CONFIGURED MODEL:", settings.model)

        print("\nAVAILABLE MODELS:")

        for model in available_models:
            print(model)

        if settings.model in available_models:
            print("\nMODEL AVAILABLE")
        else:
            print("\nMODEL NOT AVAILABLE")


asyncio.run(check_models())