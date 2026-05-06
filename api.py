
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import json

# Import core components from the existing agent project
# Assuming agent.py, chat.py, and related modules are accessible
try:
    from chat import Runner, InMemorySessionService, USER_ID, SESSION_ID
    from agent import root_agent, ensure_model_loaded
    import logger
    import time
except ImportError as e:
    print(f"Error importing agent components: {e}")
    exit(1)

app = FastAPI(title="Gemma ADK Chat Agent API")

# Pydantic schema for incoming messages
class ChatRequest(BaseModel):
    message: str

# Global state initialization (can be refined for production)
session_service = InMemorySessionService()
async def initialize_agent():
    await session_service.create_session(
        app_name="portfolio_app", user_id=USER_ID, session_id=SESSION_ID
    )
    ensure_model_loaded()
    global runner
    runner = Runner(agent=root_agent, app_name="portfolio_app", session_service=session_service)

# Initialize the agent once when the server starts
@app.on_event("startup")
async def startup_event():
    await initialize_agent()
    print("Agent initialized successfully for API.")


def _unwrap_tool_result(response: dict) -> str:
    if not response:
        return ""
    val = response.get("output") or response.get("result") or response
    if isinstance(val, dict):
        val = val.get("output") or val.get("result") or str(val)
    return str(val)

async def stream_chat_response(runner: Runner, user_input: str):
    """
    Generator function to stream the agent's response.
    This function adapts the logic from chat.py's _send to yield chunks.
    """
    message = {"role": "user", "parts": [{ "text": user_input }]}
    
    # Simplified state tracking for streaming
    response_buf = ""
    thinking_buf = ""
    is_final = False
    
    # Using an async generator to yield chunks
    async for event in runner.run_async(
        user_id=USER_ID, session_id=SESSION_ID, new_message=message
    ):
        # Tool calls/responses (can be logged or streamed separately if needed)
        for fc in event.get_function_calls():
            yield json.dumps({"type": "tool_call", "name": fc.name, "args": fc.args}) + "\n"
        for fr in event.get_function_responses():
            result = _unwrap_tool_result(fr.response)
            yield json.dumps({"type": "tool_result", "name": fr.name, "result": result}) + "\n"

        if event.content and event.content.parts:
            # Check for thinking part
            think_text = "".join(
                p.text for p in event.content.parts
                if p.text and getattr(p, "thought", False)
            )
            if think_text:
                thinking_buf += think_text
                yield json.dumps({"type": "thinking", "text": think_text}) + "\n"

            # Check for response part
            resp_text = "".join(
                p.text for p in event.content.parts
                if p.text and not getattr(p, "thought", False)
            )

            if resp_text and event.partial:
                response_buf += resp_text
                # Yield partial response immediately
                yield json.dumps({"type": "chunk", "text": resp_text}) + "\n"

        if event.is_final_response():
            is_final = True
            # Ensure any remaining thoughts/chunks are yielded
            if thinking_buf:
                yield json.dumps({"type": "thinking", "text": thinking_buf}) + "\n"
            
            # Final yield of the complete response
            if response_buf:
                yield json.dumps({"type": "chunk", "text": response_buf}) + "\n"
                
            break
            
# The main streaming endpoint
@app.post("/api/v1/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Accepts a message and streams the response chunk-by-chunk.
    """
    try:
        # The runner object is initialized globally on startup
        return StreamingResponse(
            stream_chat_response(runner, request.message), 
            media_type="text/event-stream"
        )
    except Exception as e:
        print(f"API Error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

