from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import shutil
import os

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Update as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dummy implementations — replace with your logic
def transform_new_website_chunk(input, filename=None):
    if filename:
        return f"Transformed from file: {filename}"
    return f"Transformed from URL: {input}"

def generate_components(prompt):
    return f"Generated UI components from: {prompt}"

# ----------------------
# 1. Transform from URL
# ----------------------
class URLInput(BaseModel):
    url: str

@app.post("/transform/url")
async def transform_from_url(data: URLInput):
    transformed = transform_new_website_chunk(data.url)
    return {"transformed_content": transformed}

# -----------------------
# 2. Transform from File
# -----------------------
@app.post("/transform/file")
async def transform_from_file(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    class Doc:
        def __init__(self, content, filename):
            self.content = content
            self.metadata = {"filename": filename}

    with open(temp_path, "r", encoding="utf-8") as f:
        content = f.read()

    doc = Doc(content, file.filename)
    result = transform_new_website_chunk(doc, filename=doc.metadata["filename"])

    os.remove(temp_path)
    return {"transformed_content": result}

# -------------------------------
# 3. Generate Components from Prompt
# -------------------------------


class PromptInput(BaseModel):
    prompt: List[str]

@app.post("/generate/components")
async def generate_from_prompt(data: PromptInput):
    result = generate_components(data.prompt)
    return {"generated_components": result}
