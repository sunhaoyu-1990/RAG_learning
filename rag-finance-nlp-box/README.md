# RAG 金融 NLP 项目

一个基于 RAG (Retrieval-Augmented Generation) 架构的金融自然语言处理（NLP）项目，旨在提供专业的金融术语标准化、缩写扩展和拼写纠正功能。前端使用 React 构建，后端使用 Python 和 FastAPI 构建。

## 功能特点

- **术语标准化**: 将非标准的金融术语转换为统一的、标准的术语。
- **缩写扩展**: 自动识别并扩展金融文本中的缩写词。
- **拼写纠正**: 对文本中的拼写错误进行检测和纠正。
- **可配置模型**: 支持通过界面切换和配置不同的语言模型和向量模型。

## 技术栈

### 前端

- **框架**: React
- **路由**: React Router
- **样式**: Tailwind CSS
- **组件库**: Headless UI
- **图标**: Lucide React

### 后端

- **Web 框架**: FastAPI
- **AI / NLP**: LangChain, Hugging Face Transformers, PyTorch, Sentence-Transformers
- **大语言模型 (LLM)**: OpenAI, Ollama
- **向量数据库**: Milvus
- **图数据库**: Neo4j
- **数据处理**: Pandas, NumPy

## 项目结构

```
rag-finance-nlp-box/
├── backend/
│   ├── main.py             # FastAPI 应用入口
│   ├── services/           # 业务逻辑服务
│   │   ├── abbr_service.py
│   │   ├── corr_service.py
│   │   └── std_service.py
│   ├── utils/              # 工具模块
│   │   ├── embedding_factory.py
│   │   └── embedding_config.py
│   ├── tools/
│   │   └── create_milvus_db.py # 创建 Milvus 数据库的脚本
│   └── data/               # 数据文件
├── frontend/
│   ├── src/
│   │   ├── index.js          # React 应用入口
│   │   ├── App.js            # 主应用组件
│   │   ├── components/       # 可复用组件
│   │   │   ├── Sidebar.js
│   │   │   └── shared/
│   │   │       └── ModelOptions.js
│   │   └── pages/            # 页面组件
│   │       ├── AbbrPage.js
│   │       ├── CorrPage.js
│   │       └── StdPage.js
│   ├── package.json        # 前端依赖和脚本
│   └── tailwind.config.js  # Tailwind CSS 配置
├── requirements.txt      # 后端 Python 依赖
└── README.md
```

## 开始使用

### 1. 环境准备

- 克隆仓库
- 安装 Python 依赖 (建议在虚拟环境_中进行):
  ```bash
  pip install -r requirements.txt
  ```
- 安装前端依赖:
  ```bash
  cd frontend
  npm install
  ```

### 2. 数据库准备

- 根据 `backend/tools/create_milvus_db.py` 脚本中的指引，准备和初始化 Milvus 向量数据库。

### 3. 运行服务

- **启动后端服务**:
  在项目根目录下运行:
  ```bash
  uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
  ```
- **启动前端应用**:
  在 `frontend` 目录下运行:
  ```bash
  npm start
  ```
- 在浏览器中打开 `http://localhost:3000`

## 参与贡献

1.  Fork 本仓库
2.  创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3.  提交您的更改 (`git commit -m '添加一些特性'`)
4.  推送到分支 (`git push origin feature/AmazingFeature`)
5.  开启一个 Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 LICENSE 文件了解详情