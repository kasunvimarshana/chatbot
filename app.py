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
from pyngrok import ngrok, conf
import os
from datetime import datetime
import threading

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

# Initialize FastAPI app
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
    docs_url="/docs",
    redoc_url="/redoc"
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

# Ngrok configuration
class NgrokManager:
    def __init__(self):
        self.public_url = None
        self.tunnel = None
        # Configure ngrok
        conf.get_default().region = "eu"  # or "us", "eu", "ap", "au", "sa"
        # Set your auth token if available
        if hasattr(settings, 'NGROK_AUTH_TOKEN'):
            ngrok.set_auth_token(settings.NGROK_AUTH_TOKEN)

    def start_tunnel(self, port=8000):
        try:
            self.tunnel = ngrok.connect(
                port,
                proto="http",
                bind_tls=True,
                # For paid accounts you can add:
                # subdomain="your-subdomain",
                # hostname="your-custom-domain.com",
                # auth="username:password",
            )
            self.public_url = self.tunnel.public_url
            logger.info(f"Ngrok tunnel created: {self.public_url}")
            return self.public_url
        except Exception as e:
            logger.error(f"Failed to create ngrok tunnel: {str(e)}")
            raise

    def close_tunnel(self):
        if self.tunnel:
            ngrok.disconnect(self.tunnel.public_url)
            logger.info("Ngrok tunnel closed")

    def get_public_url(self):
        return self.public_url

# Initialize ngrok manager
ngrok_manager = NgrokManager()

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
        start_time = time.perf_counter()
        message = request.message.strip()
        context = request.context or {}
        
        if not message:
            raise HTTPException(status_code=400, detail="Empty message")
        
        # Process the message
        response = chatbot.get_response(message, context)
        processing_time = time.perf_counter() - start_time
        
        # Log the interaction
        logger.info(f"Processed message: '{message}' in {processing_time:.3f}s")
        
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
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime": time.time() - app.startup_time
    }

@app.get("/ngrok-info", include_in_schema=False)
async def get_ngrok_info():
    """Get ngrok tunnel information"""
    return {
        "public_url": ngrok_manager.get_public_url(),
        "tunnels": [str(t) for t in ngrok.get_tunnels()]
    }

# Add startup and shutdown events
@app.on_event("startup")
async def startup_event():
    app.startup_time = time.time()
    # Start ngrok in a separate thread to not block the application
    threading.Thread(target=start_ngrok_tunnel).start()

@app.on_event("shutdown")
async def shutdown_event():
    ngrok_manager.close_tunnel()
    logger.info("Application shutdown complete")

def start_ngrok_tunnel():
    try:
        public_url = ngrok_manager.start_tunnel(8000)
        # Update OpenAPI servers
        app.servers = [{"url": public_url, "description": "ngrok tunnel"}]
        logger.info(f"Swagger UI available at: {public_url}/docs")
    except Exception as e:
        logger.error(f"Failed to start ngrok: {str(e)}")

if __name__ == "__main__":
    # Start Uvicorn server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
        reload=False,
        workers=1
    )

