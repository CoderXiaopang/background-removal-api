# 背景移除 API 服务

基于 FastAPI 和 PyTorch 的图像背景移除服务，使用 RMBG-2.0 模型。

## 功能特点

- 🚀 基于 FastAPI 的高性能 API 服务
- 🤖 使用先进的 RMBG-2.0 模型进行背景移除
- 🖼️ 支持 base64 编码的图像输入输出
- 🐳 Docker 容器化部署
- 🔧 支持 CUDA GPU 加速

## 快速开始

### 使用 Docker Compose

```bash
# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

### 本地开发

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
uvicorn server:app --host 0.0.0.0 --port 6001
```

## API 文档

服务启动后，可以访问：
- API 文档: http://localhost:6001/docs
- 健康检查: http://localhost:6001/health

### 主要端点

#### POST /segment
移除图像背景

**请求体:**
```json
{
  "image_base64": "base64编码的图像数据"
}
```

**响应:**
```json
{
  "result_image_base64": "移除背景后的图像(base64)",
  "mask_image_base64": "遮罩图像(base64)"
}
```

#### GET /health
健康检查端点

## 技术栈

- **后端框架**: FastAPI
- **深度学习**: PyTorch, torchvision, transformers
- **图像处理**: Pillow, kornia
- **模型**: ModelScope (RMBG-2.0)
- **容器化**: Docker, Docker Compose

## 系统要求

- Python 3.8+
- CUDA 12.4+ (可选，用于GPU加速)
- Docker & Docker Compose (容器化部署)

## 许可证

MIT License