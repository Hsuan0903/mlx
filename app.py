from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from mlx_vlm import load, generate, stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from PIL import Image
import requests
from pydantic import BaseModel
from io import BytesIO
import time
import os

# 初始化 FastAPI 應用
app = FastAPI()

# 設置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # 允許前端來源
    allow_credentials=True,
    allow_methods=["*"],  # 允許所有 HTTP 方法（GET, POST 等）
    allow_headers=["*"],  # 允許所有頭部
)


# 全局變量，用於存儲模型、處理器、配置和系統提示
model = None
processor = None
config = None
system_prompt = "Describe this image."
formatted_prompt = None

class PromptRequest(BaseModel):
    prompt: str

class GenerateRequest(BaseModel):
    image_path: str

class StreamResponse(BaseModel):
    text: str

# 端點 1：加載模型
@app.post("/load_model")
async def load_model(model_path: str = "mlx-community/Qwen2.5-VL-3B-Instruct-8bit"):
    global model, processor, config
    try:
        start_time = time.time()
        model, processor = load(model_path)
        config = load_config(model_path)
        end_time = time.time()
        return JSONResponse(content={"message": "Model loaded successfully", "time_taken": end_time - start_time})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 端點 2：設置系統提示
@app.post("/set_prompt")
async def set_prompt(request: PromptRequest):
    global system_prompt, formatted_prompt
    system_prompt = request.prompt
    formatted_prompt = apply_chat_template(processor, config, system_prompt, num_images=1)
    return JSONResponse(content={"message": "System prompt set successfully", "prompt": system_prompt})

# 端點 3：輸入圖像
@app.post("/input_image")
async def input_image(file: UploadFile = File(None), url: str = None):
    if file is None and url is None:
        raise HTTPException(status_code=400, detail="Please provide either an image file or a URL.")
    
    try:
        if file:
            # 從上傳的文件讀取圖像
            image_data = await file.read()
            image = Image.open(BytesIO(image_data))
        elif url:
            # 從 URL 下載圖像
            response = requests.get(url)
            image = Image.open(BytesIO(response.content))
        
        # 保存圖像到臨時文件
        image_path = "temp_image.jpg"

        image.save(image_path)
        return JSONResponse(content={"message": "Image received", "image_path": image_path})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 端點 4：生成輸出
@app.post("/generate")
async def generate_output(request: GenerateRequest):
    global model, processor, config, system_prompt
    if model is None or processor is None or config is None:
        raise HTTPException(status_code=400, detail="Model not loaded. Please load the model first.")
    
    try:
        start_time = time.time()
        
        # 生成輸出
        output = generate(model, processor, formatted_prompt, [request.image_path], verbose=False)
        end_time = time.time()
        
        # 刪除臨時圖像文件
        if os.path.exists(request.image_path):
            os.remove(request.image_path)
        
        return JSONResponse(content={"output": output, "time_taken": end_time - start_time})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 端點 5：流式生成輸出
@app.post("/stream_generate")
async def stream_generate_output(request: GenerateRequest):
    global model, processor, config, system_prompt
    if model is None or processor is None or config is None:
        raise HTTPException(status_code=400, detail="Model not loaded. Please load the model first.")
    
    async def generate_stream():
        start_time = time.time()
        for chunk in stream_generate(model, processor, formatted_prompt, [request.image_path], verbose=False):
            yield f'{{"text": "{chunk.text}"}}\n'  # Send each chunk as a separate JSON line
        end_time = time.time()
        yield f'{{"time_taken": {end_time - start_time}}}\n'
        
        if os.path.exists(request.image_path):
            os.remove(request.image_path)

    return StreamingResponse(generate_stream(), media_type="application/x-ndjson")
# 運行伺服器
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)