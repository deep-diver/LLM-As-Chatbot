import time
import json
import asyncio
from functools import partial

from fastapi import FastAPI, Response, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import firebase_admin
from firebase_admin import credentials
from firebase_admin import auth

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

cred = credentials.Certificate("firebase_book.json")
firebase_admin.initialize_app(cred)

def verify_login(id_token):
    try:
        decoded_token = auth.verify_id_token(id_token)
        return decoded_token
    except Exception as e:
        print("Failed to verify token", e)
        return False

def check_auth(request: Request):
    # Check for authorization header
    authorization_header = request.headers.get("Authorization")
    if authorization_header:
        id_token = authorization_header.replace("Bearer ", "")
        try:
            verified = verify_login(id_token) # Verify login
            if not verified:
                raise HTTPException(status_code=403, detail="403")
        except Exception as e:
            raise HTTPException(status_code=401, detail="401")
    else:
        # Return 402 Payment Required with message
        raise HTTPException(status_code=402, detail="402")

async def generate(text):
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
async def echo1(text: str, request: Request):
    check_auth(request)
    
    gen = partial(generate, text)

    return StreamingResponse(gen(), media_type="text/event-stream")
  
@app.post("/echo")
async def echo2(text: str, request: Request):
    check_auth(request)
    
    gen = partial(generate, text)
    
    return StreamingResponse(gen(), media_type="text/event-stream")  
