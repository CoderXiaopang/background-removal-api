"""
背景移除 FastAPI 服务

依赖安装:
pip install fastapi uvicorn pillow torch torchvision kornia transformers modelscope requests

或使用 requirements.txt:
fastapi>=0.104.0
uvicorn>=0.24.0
pillow>=10.0.0
torch>=2.0.0
torchvision>=0.15.0
kornia>=0.7.0
transformers>=4.30.0
modelscope>=1.9.0
requests>=2.31.0
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import torch
from torchvision import transforms
from modelscope import AutoModelForImageSegmentation
import base64
import io
from typing import Dict

app = FastAPI(title="背景移除 API", version="1.0.0")

# 初始化模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

# 加载模型
model = AutoModelForImageSegmentation.from_pretrained(
    'AI-ModelScope/RMBG-2.0', 
    trust_remote_code=True
).eval().to(device)

# 数据预处理设置
image_size = (1024, 1024)
transform_image = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 请求体模型
class ImageRequest(BaseModel):
    image_base64: str

# 响应体模型
class ImageResponse(BaseModel):
    result_image_base64: str
    mask_image_base64: str

@app.post("/segment", response_model=ImageResponse)
async def segment_image(request: ImageRequest) -> Dict[str, str]:
    """
    移除图像背景的 API 端点
    
    Args:
        request: 包含 base64 编码图像的请求
    
    Returns:
        包含结果图像和 mask 图像的 base64 编码字典
    """
    try:
        # 1. 解码 base64 图像
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        original_size = image.size
        
        # 2. 预处理图像
        input_images = transform_image(image).unsqueeze(0).to(device)
        
        # 3. 模型推理
        with torch.no_grad():
            preds = model(input_images)[-1].sigmoid().cpu()
        
        # 4. 后处理 - 生成 mask
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(original_size)
        
        # 5. 生成无背景图像（PNG 格式，带 alpha 通道）
        result_image = image.copy()
        result_image.putalpha(mask)
        
        # 6. 将结果转换为 base64
        # 结果图像（PNG 格式，带透明背景）
        result_buffer = io.BytesIO()
        result_image.save(result_buffer, format='PNG')
        result_base64 = base64.b64encode(result_buffer.getvalue()).decode('utf-8')
        
        # Mask 图像（灰度图）
        mask_buffer = io.BytesIO()
        mask.save(mask_buffer, format='PNG')
        mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode('utf-8')
        
        return {
            "result_image_base64": result_base64,
            "mask_image_base64": mask_base64
        }
    
    except base64.binascii.Error:
        raise HTTPException(status_code=400, detail="无效的 base64 编码")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理图像时出错: {str(e)}")

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "device": device,
        "model_loaded": model is not None
    }

if __name__ == "__main__":
    import uvicorn
    # 启动服务，监听 6001 端口
    uvicorn.run(app, host="0.0.0.0", port=6001)
