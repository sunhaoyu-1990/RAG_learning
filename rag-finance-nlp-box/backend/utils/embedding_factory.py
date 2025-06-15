import dotenv
dotenv.load_dotenv()
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
import boto3
import os
from utils.embedding_config import EmbeddingProvider, EmbeddingConfig

class EmbeddingFactory:
    @staticmethod
    def create_embedding_function(config: EmbeddingConfig):
        if config.provider == EmbeddingProvider.OPENAI:
            return OpenAIEmbeddings(
                model=config.model_name,
                openai_api_key=os.getenv('OPENAI_API_KEY')
            )
            
        elif config.provider == EmbeddingProvider.HUGGINGFACE:
            return HuggingFaceEmbeddings(
                model_name=config.model_name
            )
            
        raise ValueError(f"Unsupported embedding provider: {config.provider}")