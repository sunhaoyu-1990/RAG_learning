from pymilvus import model
from pymilvus import MilvusClient
import pandas as pd
from tqdm import tqdm
import logging
from dotenv import load_dotenv
load_dotenv()
import torch    
from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema
import os

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 初始化 OpenAI 嵌入函数
embedding_function = model.dense.SentenceTransformerEmbeddingFunction(
            model_name='BAAI/bge-m3',
            # model_name='jinaai/jina-embeddings-v3',
            device='cuda:0' if torch.cuda.is_available() else 'cpu',
            trust_remote_code=True
        )
# embedding_function = model.dense.OpenAIEmbeddingFunction(model_name='text-embedding-3-large')

# 文件路径
file_path = "/home/train/rag-finance-nlp-box/backend/data/万条金融标准术语.csv"
db_path = "/home/train/rag-finance-nlp-box/backend/db/finance_bge_m3.db"

# 确保数据库目录存在
db_dir = os.path.dirname(db_path)
if db_dir and not os.path.exists(db_dir):
    os.makedirs(db_dir)
    logging.info(f"Created directory: {db_dir}")

# 连接到 Milvus
client = MilvusClient(db_path)

collection_name = "finance_terms_bge_m3"

# 加载数据
logging.info("Loading data from CSV")
df = pd.read_csv(file_path, header=None, names=['term', 'source'], dtype=str).fillna("NA")

# 获取向量维度（使用一个样本文档）
sample_doc = "Sample Text"
sample_embedding = embedding_function([sample_doc])[0]
vector_dim = len(sample_embedding)

# 构造Schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
    FieldSchema(name="term", dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=50),
    FieldSchema(name="input_file", dtype=DataType.VARCHAR, max_length=500),
]
schema = CollectionSchema(fields, 
                          "Financial Terms", 
                          enable_dynamic_field=True)

# 如果集合不存在，创建集合
if not client.has_collection(collection_name):
    client.create_collection(
        collection_name=collection_name,
        schema=schema
    )
    logging.info(f"Created new collection: {collection_name}")

# # 在创建集合后添加索引
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="vector",  # 指定要为哪个字段创建索引，这里是向量字段
    index_type="AUTOINDEX",  # 使用自动索引类型，Milvus会根据数据特性选择最佳索引
    metric_type="COSINE",  # 使用余弦相似度作为向量相似度度量方式
    params={"nlist": 1024}  # 索引参数：nlist表示聚类中心的数量，值越大检索精度越高但速度越慢
)

client.create_index(
    collection_name=collection_name,
    index_params=index_params
)

# 批量处理
batch_size = 1024

for start_idx in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
    end_idx = min(start_idx + batch_size, len(df))
    batch_df = df.iloc[start_idx:end_idx]

    # 准备文档
    docs = batch_df['term'].tolist()

    # 生成嵌入
    try:
        embeddings = embedding_function(docs)
        logging.info(f"Generated embeddings for batch {start_idx // batch_size + 1}")
    except Exception as e:
        logging.error(f"Error generating embeddings for batch {start_idx // batch_size + 1}: {e}")
        continue

    # 准备数据
    data = [
        {
            "vector": embeddings[idx],
            "term": str(row['term']),
            "source": str(row['source']),
            "input_file": file_path
        } for idx, (_, row) in enumerate(batch_df.iterrows())
    ]

    # 插入数据
    try:
        res = client.insert(
            collection_name=collection_name,
            data=data
        )
        logging.info(f"Inserted batch {start_idx // batch_size + 1}, result: {res}")
    except Exception as e:
        logging.error(f"Error inserting batch {start_idx // batch_size + 1}: {e}")

logging.info("Insert process completed.")

query = "A-Share"
query_embeddings = embedding_function([query])


# 搜索余弦相似度最高的
search_result = client.search(
    collection_name=collection_name,
    data=[query_embeddings[0].tolist()],
    limit=5,
    output_fields=["term", 
                   "source" 
                   ]
)
logging.info(f"Search result for '{query}': {search_result}")

# 查询所有匹配的实体
query_result = client.query(
    collection_name=collection_name,
    filter="term == 'A-Shares'",
    output_fields=["term", 
                   "source"
                   ],
    limit=5
)
logging.info(f"Query result for term == 'A-Shares': {query_result}")
