import asyncio
import os
from dotenv import load_dotenv
import chat

load_dotenv()

if __name__ == "__main__":
    asyncio.run(chat.run(
        model_name=os.getenv("GEMMA_MODEL", "gemma-4"),
        app_name=os.getenv("APP_NAME", "gemma-adk-agent"),
    ))
