from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from datetime import datetime
import os
import time
import logging
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from typing import Dict
import requests
from PIL import Image, UnidentifiedImageError
import io
from dotenv import load_dotenv
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app setup
app = FastAPI()

# Setup directories
UPLOAD_DIR = "static/generated"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Setup template and static directories
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Database setup
Base = declarative_base()
engine = create_engine('sqlite:///image_generation.db')
SessionLocal = sessionmaker(bind=engine)

# Configuration
load_dotenv()

CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
CLOUDFLARE_AI_GATEWAY_ID = os.getenv("CLOUDFLARE_AI_GATEWAY_ID")
CLOUDFLARE_API_URL = f"https://gateway.ai.cloudflare.com/v1/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_AI_GATEWAY_ID}"
API_KEY = os.getenv("CLOUDFLARE_API_KEY")

# Predefined prompts and models
PROMPTS = [
    "Create macro photos of the intricate patterns on butterfly wings, showcasing the beauty of nature",
    "A 3d render of a futuristic cityscape with flying cars and neon lights, inspired by cyberpunk aesthetics",
    "Create a portrait close up of a woman with a striking hairstyle, she has bright blue hair and a nose ring. It should be in 8k with cinematic and volumetric lighting",
    "Generate photos of a pair of noise cancelling headphones in various settings, highlighting their design, comfort, and sound quality",
    "Create visuals of a luxury car driving through scenic landscapes, highlighting its design and performance",
    "Create food photography of fish and chips in the style of Marc Haydon, showcasing the capabilities of the Hasselblad camera",
    "A big neon sign that says 'Welcome to the Future' in a futuristic city setting, with flying cars and tall skyscrapers",
]

AI_MODELS = [
    "@cf/lykon/dreamshaper-8-lcm",
    "@cf/black-forest-labs/flux-1-schnell",
    "@cf/stabilityai/stable-diffusion-xl-base-1.0",
    "@cf/bytedance/stable-diffusion-xl-lightning"
]

# Updated Database model
class ImageGeneration(Base):
    __tablename__ = "image_generations"
    
    id = Column(Integer, primary_key=True)
    prompt = Column(String)
    model = Column(String)
    generation_time = Column(Float)
    rating = Column(Float, nullable=True)
    image_path = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Add unique constraint for prompt + model combination
    __table_args__ = (UniqueConstraint('prompt', 'model', name='unique_prompt_model'),)

# Create database tables
Base.metadata.drop_all(engine)  # Drop existing tables to handle schema changes
Base.metadata.create_all(engine)

async def call_cloudflare_api(prompt: str, model: str) -> bytes:
    """
    Call the Cloudflare AI API to generate an image.
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "cf-aig-cache-ttl": "0"
    }
    
    payload = [{
        "provider": "workers-ai",
        "endpoint": model,
        "headers": headers,
        "query": {
            "prompt": prompt
        }
    }]
    
    try:
        response = requests.post(CLOUDFLARE_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        if response.headers.get("Content-Type") == "application/json":
            response_json = response.json()
            if "result" in response_json and "image" in response_json["result"]:
                image_base64 = response_json["result"]["image"]
                return base64.b64decode(image_base64)
            else:
                raise ValueError("Unexpected response format from API")
        else:
            return response.content
    
    except Exception as e:
        logger.error(f"API call failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

async def generate_image(prompt: str, model: str) -> Dict:
    """
    Generate an image using the Cloudflare AI API and save it locally.
    """
    start_time = time.time()
    
    # Generate image through API
    image_data = await call_cloudflare_api(prompt, model)
    
    try:
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_data))
    except UnidentifiedImageError:
        raise HTTPException(status_code=500, detail="Failed to identify image file")
    
    # Save image locally
    timestamp = int(time.time())
    filename = f"{timestamp}_{model.split('/')[-1]}.png"
    filepath = os.path.join(UPLOAD_DIR, filename)
    image.save(filepath)
    
    generation_time = time.time() - start_time
    relative_path = f"/static/generated/{filename}"
    
    return {
        "image_url": relative_path,
        "generation_time": generation_time
    }

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the main dashboard page"""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prompts": enumerate(PROMPTS),
            "models": AI_MODELS
        }
    )

@app.post("/generate")
async def generate(prompt_index: int = Form(...)):
    """Handle image generation requests for all models"""
    try:
        if prompt_index not in range(len(PROMPTS)):
            raise HTTPException(status_code=400, detail="Invalid prompt index")
        
        prompt = PROMPTS[prompt_index]
        results = []
        
        # Generate images for all models
        for model in AI_MODELS:
            result = await generate_image(prompt, model)
            
            # Check for existing record
            db = SessionLocal()
            existing_record = db.query(ImageGeneration).filter_by(
                prompt=prompt,
                model=model
            ).first()
            
            if existing_record:
                # Update existing record
                existing_record.generation_time = result["generation_time"]
                existing_record.image_path = result["image_url"]
                existing_record.created_at = datetime.utcnow()
            else:
                # Create new record
                db_record = ImageGeneration(
                    prompt=prompt,
                    model=model,
                    generation_time=result["generation_time"],
                    image_path=result["image_url"]
                )
                db.add(db_record)
            
            db.commit()
            
            results.append({
                "model": model,
                "image_url": result["image_url"],
                "generation_time": result["generation_time"]
            })
            
            db.close()
        
        return results
    
    except Exception as e:
        logger.error(f"Error generating images: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rate")
async def rate_image(
    prompt: str = Form(...),
    model: str = Form(...),
    rating: float = Form(...)
):
    """Update the rating for a specific image generation"""
    if rating < 0 or rating > 5:
        raise HTTPException(status_code=400, detail="Rating must be between 0 and 5")
    
    db = SessionLocal()
    try:
        record = db.query(ImageGeneration).filter_by(
            prompt=prompt,
            model=model
        ).first()
        
        if not record:
            raise HTTPException(status_code=404, detail="Image generation not found")
        
        record.rating = rating
        db.commit()
        return {"status": "success"}
    
    finally:
        db.close()

@app.get("/metrics")
async def get_metrics():
    """Get metrics for dashboard visualizations"""
    db = SessionLocal()
    try:
        results = db.query(ImageGeneration).all()
        total_generations = len(results)
        average_generation_time = sum(r.generation_time for r in results) / total_generations if total_generations > 0 else 0
        rated_results = [r for r in results if r.rating is not None]
        average_rating = sum(r.rating for r in rated_results) / len(rated_results) if rated_results else 0

        metrics = {
            "total_generations": total_generations,
            "average_generation_time": average_generation_time,
            "average_rating": average_rating,
            "model_usage": {},
            "prompt_usage": {}
        }
        
        for model in AI_MODELS:
            model_results = [r for r in results if r.model == model]
            if model_results:
                rated_model_results = [r for r in model_results if r.rating is not None]
                metrics["model_usage"][model] = {
                    "count": len(model_results),
                    "avg_time": sum(r.generation_time for r in model_results) / len(model_results),
                    "avg_rating": sum(r.rating for r in rated_model_results) / len(rated_model_results) if rated_model_results else 0
                }
        
        return metrics
    finally:
        db.close()

@app.get("/images")
async def get_images():
    """Get all generated images grouped by prompt"""
    db = SessionLocal()
    try:
        results = db.query(ImageGeneration).order_by(ImageGeneration.created_at.desc()).all()
        
        # Group images by prompt
        grouped_results = {}
        for result in results:
            if result.prompt not in grouped_results:
                grouped_results[result.prompt] = []
            
            grouped_results[result.prompt].append({
                "url": result.image_path,
                "model": result.model,
                "generation_time": result.generation_time,
                "rating": result.rating,
                "created_at": result.created_at.isoformat()
            })
        
        return grouped_results
    finally:
        db.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)