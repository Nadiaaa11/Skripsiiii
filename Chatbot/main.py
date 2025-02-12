from collections import defaultdict
import http
import shutil
from openai import OpenAI
from fastapi import FastAPI, Form, Request, File, UploadFile, WebSocket, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from slugify import slugify
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, func, or_
from databases import Database
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.future import select
from fastapi.middleware.cors import CORSMiddleware
from markdown import markdown
import os
import re
import uuid
import requests
import pymysql
import aiomysql
import logging
from scipy import spatial
import pandas as pd
import spacy
import cloudinary
import cloudinary.uploader
import logging

openai = OpenAI(
    api_key="sk-V4UGt-FNIde85D7t0zrvZbVG5eolZDsE8awXTAuJYgT3BlbkFJToj--_okCjQAcwzYu4ZC6JDX8kznTJhzruBhL9Q5YA"
)

client = OpenAI()

app = FastAPI()

app.mount("/static", StaticFiles(directory="Chatbot/static"), name="static")

templates = Jinja2Templates(directory="Chatbot/templates")

UPLOAD_DIR = "static/uploads"

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "jfif"}

Base = declarative_base()

DATABASE_URL = "mysql+aiomysql://root:@localhost:3306/ecommerce"

engine = create_async_engine(DATABASE_URL, echo=True)

database = Database(DATABASE_URL)

class Product(Base):
    __tablename__ = "product"
    product_id = Column(Integer, primary_key=True, index=True)
    product_name = Column(String(255), nullable=False)
    product_detail = Column(String(255), nullable=False)
    product_price = Column(Float, nullable=False)
    product_seourl = Column(String(255), nullable=False)

class ProductPhoto(Base):  
    __tablename__ = "product_photo"
    productphoto_id = Column(Integer, primary_key=True)
    product_id = Column(Integer, nullable=False)
    productphoto_path = Column(String(255), nullable=False)
    productphoto_order = Column(Integer, nullable=False)

class ProductVariant(Base):
    __tablename__ = "product_variant"
    variant_id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, nullable=False)
    size = Column(String(50), nullable=False)
    color = Column(String(50), nullable=False)
    stock = Column(Integer, nullable=False)
    product_price = Column(Float, nullable=False)

async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_db():
    async with AsyncSession(engine) as session:  # Updated to use AsyncSession
        yield session
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.embeddings.create(input=[text], engine=model)['data'][0]['embedding']

def cosine_similarity(a, b):
    return 1 - spatial.distance.cosine(a, b)

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

nlp = spacy.load("en_core_web_sm")

def extract_keywords_from_ai_response(ai_response: str):
    doc = nlp(ai_response)  # Use spaCy for noun phrases and entities
    keywords = set(chunk.text for chunk in doc.noun_chunks)  # Extract noun chunks
    keywords.update(ent.text for ent in doc.ents)  # Add named entities

    # Use regex as a fallback to extract any remaining capitalized phrases
    regex_keywords = re.findall(r'\b[A-Z][a-z]*(?:\s[A-Z][a-z]*)*\b', ai_response)
    keywords.update(regex_keywords)
    
    return list(keywords)

def render_markdown(text: str) -> str:
    html_content = markdown(text)
    return html_content

stop_words = [
    "a", "an", "and", "are", "as", "at", "be", "but", "by", 
    "for", "if", "in", "into", "is", "it", "its", "of", "on", 
    "or", "so", "such", "that", "the", "their", "then", 
    "there", "these", "they", "this", "to", "too", "was", 
    "will", "with", "you", "your", "do", "does", "did", 
    "have", "has", "having", "we", "us", "our", "ours", 
    "I", "me", "my", "mine", "he", "him", "his", "she", 
    "her", "hers", "itself", "themselves", "yourself", 
    "yourselves", "those", "from", "which", "or", "any", 
    "all", "some", "each", "every", "one", "once", "while", 
    "when", "where", "how", "what", "why", "about", "like", 
    "over", "under", "more", "less", "up", "down", "out", 
    "around", "just", "only", "even", "always", "never", 
    "not", "can", "could", "should", "would", "might", 
    "must", "shall", "may", "perhaps", "often", "sometimes", 
    "always", "usually"
]

async def fetch_products_from_db(db: AsyncSession, keywords: list[str]):
    relevant_keywords = [kw for kw in keywords if len(kw) > 2 and kw.lower() not in stop_words]

    query = (
        select(Product, ProductVariant, ProductPhoto)
        .join(ProductVariant, Product.product_id == ProductVariant.product_id)
        .join(ProductPhoto, Product.product_id == ProductPhoto.product_id)
        .where(
            or_(*[Product.product_name.ilike(f"%{keyword}%") for keyword in relevant_keywords]),
                ProductVariant.stock > 0  # Ensure product is available in stock
        )
        .order_by(func.rand())
        .limit(10)
    )
    
    result = await db.execute(query)
    products = result.fetchall()

    product_list = []
    seen_products = set()

    for product, variant, photo in products:
        if product.product_id not in seen_products:
            product_data = {
                "product_id": product.product_id,
                "product": product.product_name,
                "description": product.product_detail,
                "price": variant.product_price,
                "size": variant.size,
                "color": variant.color,
                "stock": variant.stock,
                "link": f"http://localhost/e-commerce-main/product-{product.product_seourl}-{product.product_id}",
                "photo": photo.productphoto_path
            }
            product_list.append(product_data)
            seen_products.add(product.product_id)

    return pd.DataFrame(product_list)

def search_products_in_dataframe(keywords, product_df):
    pattern = '|'.join(map(re.escape, keywords))
    matching_products = product_df[product_df['product'].str.contains(pattern, case=False, na=False)]
    return matching_products

@app.get("/", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cloudinary.config(
    cloud_name="dn0xl1q3g",
    api_key="252519847388784",
    api_secret="pzLNZgLzfMQ9bmwiIRoyjRFqqkU"
)

def upload_to_cloudinary(file_location):
    try:
        response = cloudinary.uploader.upload(file_location, folder="uploads/")
        return response['url']
    except Exception as e:
        logging.error(f"Cloudinary upload error: {e}")
        return None

logging.basicConfig(level=logging.DEBUG)

@app.post("/upload/")
async def upload(user_input: str = Form(None), file: UploadFile = None):
    if user_input:
        print("User input", user_input)
    else:
        user_input = None
    
    try:
        if file:
            logging.info(f"File received: {file.filename}")
            # Save file with unique name in the upload directory
            file_extension = file.filename.split(".")[-1].lower()
            if file_extension not in ALLOWED_EXTENSIONS:
                raise HTTPException(status_code=400, detail="Invalid file type.")

            unique_id = uuid.uuid4()
            sanitized_filename = slugify(file.filename.rsplit(".", 1)[0], lowercase=False)
            unique_filename = f"{unique_id}_{sanitized_filename}.{file_extension}"
            file_location = os.path.join(UPLOAD_DIR, unique_filename)

            with open(file_location, "wb+") as file_object:
                file_object.write(await file.read())

            image_url = upload_to_cloudinary(file_location)

            if image_url:
                return JSONResponse(content={"success": True, "file_url": image_url})
            else:
                return JSONResponse(content={"success": False, "error": "Failed to upload image."})

            # # Return the file URL
            # return JSONResponse(content={"success": True, "file_url": f"static/uploads/{unique_filename}"})

        elif user_input:
            return JSONResponse(content={"success": True})

        # If neither input nor file is present
        return JSONResponse(content={"success": False, "error": "No input or file received"})

    except Exception as e:
        logging.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred during file upload.")

chat_responses = []

async def is_small_talk(input_text):
    greetings = ["hello", "hi", "hey", "hi there", "hello there", "good morning", "good afternoon", "good evening", "selamat pagi", "pagi", "selamat siang", "siang", "malam", "selamat malam"]
    return input_text.lower() in greetings or re.match(r"^\s*(hi|hello|hey)\s*$", input_text, re.IGNORECASE)

sessions = defaultdict(dict)

@app.websocket("/ws")
async def chat(websocket: WebSocket, db: AsyncSession = Depends(get_db)):
    print("Attempting WebSocket connection...")
    await websocket.accept()
    print("WebSocket connection accepted.")

    message_objects = []

    message_objects.append = [{
        "role": "system",
        "content": (
            "You are a fashion consultant. Your task is to provide detailed fashion recommendations "
            "for users based on their appearance and style preferences. Respond in a friendly, natural tone "
            "and avoid using structured JSON or code format. Instead, communicate recommendations in conversational sentences.\n\n"
            "When giving recommendations, mention specific clothing items and how they would suit the user's attributes, "
            "such as height, weight, and skin tone. Use descriptive phrases and consider mentioning outfit ideas for casual occasions, "
            "unless the user specifies a different occasion.\n\n"
            "make each clothing item as a bold text and for the explanation make it as a paragraph and in different line from the title, for new or different item make it in different line."
            "Example response:\n"
            "For casual wear, here are some styles that would look great on you:  \n"
            "- **Cropped Tees or Tank Tops**  \n– Perfect for a laid-back look, and they can help accentuate your waist and balance out your height.  \n"
            "- **Oversized T-shirts**  \n– A relaxed fit in earthy or jewel tones would be flattering. You could tuck them into high-waisted jeans or shorts to add some shape.  \n"
            "- **Off-Shoulder Tops**  \n– These are cute and can show off some skin while keeping things casual. They also come in various styles, like long-sleeved or with ruffles, which add a nice detail.  \n"
            "- **Button-Up Shirts (Linen or Cotton)**  \n– A lightweight button-up, especially in tan, beige, or pastel shades, could be a staple. You can wear it loose or tied at the waist.  \n"
            "- **Graphic Tees**  \n– These are always in style and add personality to any casual outfit. You could go for retro or minimalist designs in colors that complement your skin tone.  \n\n"
            "Ask the user if these styles align with their preferences or if they have any specific style they would like to focus on."
            "Do not mention any specific brand of clothing"
            "Always ask for user opinion after each suggestion"
        )
    }]


    while True:            
        user_input = await websocket.receive_text()

        if await is_small_talk(user_input):
            ai_response = "Hello! How can i assist you with fashion recommendation today?"
            await websocket.send_text(ai_response)
            continue

        message_objects.append({
            "role": "user",
            "content": user_input,
        })

        ai_response = None

        try:
            if user_input.startswith("http://") or user_input.startswith("https://"):
                clothing_features = await analyze_uploaded_image(user_input)

                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": clothing_features}],
                    temperature=0.5
                )

                ai_response = response.choices[0].message.content.strip()

                message_objects.append({
                    "role": "assistant",
                    "content": ai_response
                })

                ai_response_html = render_markdown(ai_response)
                await websocket.send_text(ai_response_html)
                
                keywords = extract_keywords_from_ai_response(clothing_features)
        
            else:
                # Get OpenAI response using message_objects
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=message_objects,
                    temperature=0.5
                )

                ai_response = response.choices[0].message.content.strip()

                message_objects.append({
                    "role": "assistant",
                    "content": ai_response
                })

                ai_response_html = render_markdown(ai_response)
                await websocket.send_text(ai_response_html)
                # Fetch products from the database based on the user input
                keywords = extract_keywords_from_ai_response(ai_response)

            # Fetch products based on the user input
            recommended_products = await fetch_products_from_db(db, keywords)

            # Prepare the assistant's product recommendation response
            if not recommended_products.empty:
                recommendation_message = f"\n**I found these products I would recommend: **\n"
                for _, row in recommended_products.iterrows():
                    recommendation_message += (
                        f"\n![Product Image]({row['photo']})\n"
                        f"\n- **{row['product']}** for IDR{row['price']}\n"
                        f"*{row['description']}*\n"
                        f" - Available in these sizes: **{row['size']}**\n"
                        f" - <a href='{row['link']}' target='_top'>View Product</a>\n"
                    )
            else:
                recommendation_message = "Sorry, I couldn't find any products that match your description."
            
            recommendation_message_html = render_markdown(recommendation_message)
            await websocket.send_text(recommendation_message_html)

            print("Raw AI Response:", ai_response)
            print("Length of AI Response:", len(ai_response))
            print("AI Response HTML:", ai_response_html)
            print("Recommendation message HTML:", recommendation_message_html)
            print("Keywords:", keywords)

        except Exception as e:
            await websocket.send_text(f'Error: {str(e)}')
            break

async def analyze_uploaded_image(image_url: str):
    try:

        if not image_url:
            return "Error: Could not upload the image for analysis."
        
        print(f"Analyzing image at URL: {image_url}")

        response = client.chat.completions.create (
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an AI fashion consultant. When given an image URL, "
                        "please analyze the clothing item in the image, describe the colors, patterns, "
                        "style, and any notable characteristics of the clothing."
                        "Do not give any styling tips for the image"
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Please analyze the clothing item in this image. "
                                "Describe the colors, patterns, style or anyhting else that you can identify. "
                                "Do not give any styling tips for the image uploaded."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                            },
                        },
                    ] ,
                },
            ],
            max_tokens=500,
            temperature=0.5
        )
        
        analyze_features = response.choices[0].message.content
        return analyze_features
    
    except Exception as e:
        print(f"Error during image analysis: {e}")
        return f"Error: {str(e)}"
    
def clean_ai_response(ai_response: str) -> str:
    cleaned_response = re.sub(r'[^a-zA-Z0-9\s,.?!-]', '', ai_response)
    cleaned_response = re.sub(r'\s*-\s*', '-', ai_response)
    cleaned_response = re.sub(r'\s+', ' ', cleaned_response).strip()
    return cleaned_response.strip()