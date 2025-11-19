# 使用 PyTorch CUDA 镜像
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# 设置工作目录
WORKDIR /app

# 设置清华源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 复制依赖文件
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY server.py .

# 暴露端口
EXPOSE 6001

# 启动命令
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "6001"]