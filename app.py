from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field
from typing import Optional
from chatbot import ChatBot
import logging
from config import settings
import time
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chatbot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Request model
class ChatRequest(BaseModel):
    message: str = Field(..., example="Hello there!", description="User's input message")
    context: Optional[dict] = Field(None, example={"last_intent": "greeting"}, description="Conversation context")

# Response model
class ChatResponse(BaseModel):
    response: str = Field(..., example="Hello! How can I help you?", description="Chatbot's response")
    processing_time: str = Field(..., example="0.045s", description="Time taken to process the request")
    status: str = Field(..., example="success", description="Request status")

app = FastAPI(
    title="Chatbot API",
    version="1.0.0",
    description="An intelligent chatbot API with natural language processing capabilities",
    contact={
        "name": "Kasun Vimarshana",
        "email": "kasunvmail@gmail.com"
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    },
    docs_url="/docs",  # Explicitly enable Swagger at /docs
    redoc_url="/redoc" # Also enable ReDoc alternative
)

# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize chatbot
chatbot = ChatBot()

@app.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Process chat message",
    description="Processes user input and returns an intelligent response",
    response_description="Contains the chatbot response and processing metrics",
    tags=["Chat Operations"],
    responses={
        200: {"description": "Successful response"},
        400: {"description": "Invalid input or empty message"},
        500: {"description": "Internal server error"}
    }
)
async def chat_endpoint(request: ChatRequest):
    """
    Process user message and return chatbot response
    
    - **message**: User input text (required)
    - **context**: Optional conversation context
    
    Returns structured response with:
    - Generated response text
    - Processing time
    - Request status
    """
    try:
        # data = await request.json()
        # message = data.get("message", "").strip()
        message = request.message
        
        if not message:
            raise HTTPException(status_code=400, detail="Empty message")
        
        start_time = time.perf_counter()
        response = chatbot.get_response(message)
        processing_time = time.perf_counter() - start_time
        
        return {
            "response": response,
            "processing_time": f"{processing_time:.3f}s",
            "status": "success"
        }
    
    except ValueError as e:
        logger.warning(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Server error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )
    
@app.get("/health", include_in_schema=False)
async def health_check():
    """Health check endpoint for monitoring"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
