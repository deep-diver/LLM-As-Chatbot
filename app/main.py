import time
import json
import asyncio

from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

characters_per_second = 5
app = FastAPI()

origins = [
    "http://localhost:8080",
    "https://localhost:8080",
    "https://giftup.web.app/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def generate():
    number_of_yields = len(text) // characters_per_second
    for i in range(number_of_yields):
        from_index = i*characters_per_second
        to_index = i*characters_per_second+characters_per_second
        await asyncio.sleep(0.1)
        yield "data: " + json.dumps(text[from_index:to_index]) + "\n\n"

    if len(text) % characters_per_second != 0:
        await asyncio.sleep(0.1)
        yield "data: " + json.dumps(text[number_of_yields*characters_per_second:]) + "\n\n"

    yield "[DONE]"

@app.get("/echo")
async def echo1(text: str):
    return StreamingResponse(generate(), media_type="text/event-stream")
  
@app.post("/echo")
async def echo2(text: str):
    return StreamingResponse(generate(), media_type="text/event-stream")  
