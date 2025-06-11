"""
自定义数据处理管道
功能：实现一个可配置的文档加载、解析和切块框架。

该脚本包含三个核心功能：
1.  Load & Parse File (加载与解析文件):
    - 使用 `unstructured` 库，支持多种文件格式 (PDF, DOCX, MD, TXT 等)。
    - 能够解析复杂文档，如 PDF 和 Markdown，并提取其中的特殊结构。
    - 表格解析：将文档中的表格提取并转换为 Markdown 或 HTML 格式的文本。
    - 图片解析 (OCR)：(可选，需额外配置) 能够提取嵌入图片中的文字内容。
    - 生成的每个文档元素都包含丰富的元数据，如文件来源、页码、元素类型等。

2.  Chunk File (文件切块):
    - 对加载和解析后的文本内容进行切块。
    - 支持多种切块策略，借鉴了 LangChain 和 LlamaIndex 的思想：
        - 'recursive': 递归字符切分，能适应不同类型的文本。
        - 'character': 普通字符切分。
        - 'code': 针对特定编程语言的语法感知切分。
        - 'semantic' (via unstructured): `unstructured` 的解析过程本身就是一种语义切分，
          因为它按标题、段落、表格等文档固有结构进行分离。
    - 用户可以自定义切块大小 (chunk_size) 和重叠部分 (chunk_overlap)。

3.  Save to JSON (保存为JSON):
    - 将处理和切块后的文本块保存为统一格式的 JSON 文件。
    - 每个 JSON 对象包含文本内容 (`page_content`) 和元数据 (`metadata`)，
      为后续的 Embedding 和向量存储做好准备。

使用前置依赖:
- `langchain`: `pip install langchain`
- `unstructured`: `pip install "unstructured[all-docs]"` (安装所有文档格式的支持)
- OCR (可选):
    - `pip install "unstructured[ocr]"`
    - 需要在系统中安装 Tesseract OCR 引擎。
      - Windows: https://github.com/tesseract-ocr/tessdoc
      - macOS: `brew install tesseract`
      - Linux (Ubuntu): `sudo apt-get install tesseract-ocr`
- 代码切块依赖:
    - `pip install beautifulsoup4` (用于HTML)
    - `pip install "lark"` (用于Markdown)

函数命名遵循小写+下划线的蛇形命名法。
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Literal

# LangChain 相关导入
from langchain.docstore.document import Document
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    Language,
)

# Unstructured 相关导入
from unstructured.partition.auto import partition
from unstructured.documents.elements import Element

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- 1. Load & Parse File ---

def load_and_parse_file(file_path: str, **unstructured_kwargs) -> List[Document]:
    """
    加载并解析单个文件，将其转换为 LangChain Document 对象列表。
    这个过程利用 unstructured 库，同时完成了加载和智能解析（例如，提取表格和图片内容）。

    Args:
        file_path (str): 要处理的文件路径。
        **unstructured_kwargs: 传递给 `unstructured.partition` 函数的额外参数。
            例如:
            - strategy (str): 解析策略 ('auto', 'hi_res', 'fast')。'hi_res' 对PDF效果好。
            - ocr_languages (str): OCR 语言，如 'chi_sim+eng' 用于中英文。
            - extract_images_in_pdf (bool): 是否提取 PDF 中的图片 (需要OCR支持来转为文字)。
            - infer_table_structure (bool): 是否推断表格结构并转为HTML。

    Returns:
        List[Document]: 从文件中解析出的元素列表，每个元素是一个 Document 对象。
    """
    if not os.path.exists(file_path):
        logging.error(f"文件未找到: {file_path}")
        return []

    logging.info(f"开始使用 unstructured 解析文件: {file_path}")
    
    # 默认开启表格结构推断
    unstructured_kwargs.setdefault('infer_table_structure', True)
    
    try:
        elements: List[Element] = partition(filename=file_path, **unstructured_kwargs)  # partition可以解析.pdf,.docx,.md,.txt等文件
    except Exception as e:
        logging.error(f"使用 unstructured 解析文件 '{file_path}' 时出错: {e}")
        return []

    documents = []
    for element in elements:
        # 表格元素，元数据中会包含 HTML 表示
        if "unstructured.documents.elements.Table" in str(type(element)):
            metadata = element.metadata.to_dict()
            # unstructured 在某些情况下会将表格内容直接放入 text 字段
            # 如果 text 字段为空，则使用 HTML 表示
            page_content = element.text
            if not page_content.strip() and metadata.get('text_as_html'):
                 page_content = metadata['text_as_html']
            
            # 清理元数据，避免冗余
            metadata.pop('text_as_html', None)
        else:
            page_content = element.text
            metadata = element.metadata.to_dict()

        # 清理元数据中的空值
        metadata = {k: v for k, v in metadata.items() if v is not None}
        
        # 确保来源信息正确
        metadata['source'] = metadata.get('filename', os.path.basename(file_path))

        doc = Document(page_content=page_content, metadata=metadata)
        documents.append(doc)
    
    logging.info(f"文件解析完成，共得到 {len(documents)} 个文档元素。")
    return documents


# --- 2. Chunk File ---

ChunkingStrategy = Literal["recursive", "character", "code"]

def chunk_documents(
    documents: List[Document],
    strategy: ChunkingStrategy = "recursive",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    code_language: Optional[Language] = None
) -> List[Document]:
    """
    根据指定的策略对 Document列表进行切块。

    Args:
        documents (List[Document]): 待切块的文档列表。
        strategy (ChunkingStrategy): 切块策略。
        chunk_size (int): 每个块的最大大小。
        chunk_overlap (int): 块之间的重叠大小。
        code_language (Optional[Language]): 如果策略是 'code'，则需要指定编程语言。

    Returns:
        List[Document]: 切块后的文档列表。
    """
    logging.info(f"开始切块，策略: {strategy}，块大小: {chunk_size}，重叠: {chunk_overlap}")

    if strategy == "recursive":
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )
    elif strategy == "character":
        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )
    elif strategy == "code":
        if code_language is None:
            raise ValueError("使用 'code' 策略时必须提供 'code_language' 参数。")
        text_splitter = RecursiveCharacterTextSplitter.from_language(
            language=code_language,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    else:
        raise ValueError(f"未知的切块策略: {strategy}")

    chunked_docs = text_splitter.split_documents(documents)
    logging.info(f"切块完成，共生成 {len(chunked_docs)} 个文本块。")
    return chunked_docs


# --- 3. Save to JSON ---

def save_docs_to_json(documents: List[Document], output_path: str):
    """
    将 Document 列表保存为 JSON 文件。
    每个文档块存为一个 JSON 对象，包含 page_content 和 metadata。

    Args:
        documents (List[Document]): 要保存的文档列表。
        output_path (str): 输出的 JSON 文件路径。
    """
    docs_as_dicts = [
        {"page_content": doc.page_content, "metadata": doc.metadata}
        for doc in documents
    ]
    
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(docs_as_dicts, f, ensure_ascii=False, indent=4)
        logging.info(f"处理结果已成功保存到: {output_path}")
    except Exception as e:
        logging.error(f"保存 JSON 文件时出错: {e}")


def main():
    """
    主函数，演示整个处理流程。
    """
    # --- 配置区 ---
    # 输入文件路径 (请根据您的文件位置修改)
    # 示例1: 一个复杂的PDF文档
    INPUT_FILE = os.path.join("..", "90-文档-Data", "RAG", "LLM-RAG.pdf")
    # 示例2: Python源代码文件
    # INPUT_FILE = "05-LlamaIndex-语义分块.py" 
    
    # 输出文件路径
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
    filename = os.path.splitext(os.path.basename(INPUT_FILE))[0]

    # --- 流程控制 ---
    
    # 1. 加载与解析
    # 对于 PDF，使用 'hi_res' 策略效果更好，但可能需要更多计算资源
    # 如果要进行OCR，请取消注释 ocr_languages 并确保 Tesseract 已安装
    logging.info("--- 步骤 1: 加载与解析文件 ---")
    parsed_docs = load_and_parse_file(
        INPUT_FILE, 
        strategy="fast" # 可选 'hi_res', 'fast', 'auto'
        # ocr_languages="eng", # 示例: 英文OCR
        # extract_images_in_pdf=True
    )
    if not parsed_docs:
        logging.warning("文件解析后没有内容，程序终止。")
        return
    # 保存未经切块的解析结果，用于调试或作为 'semantic' 切块的结果
    save_docs_to_json(
        parsed_docs,
        os.path.join(OUTPUT_DIR, f"{filename}_parsed_only.json")
    )


    # 2. 按不同策略切块并保存
    logging.info("\n--- 步骤 2: 按不同策略进行文件切块 ---")
    
    # 策略 A: 递归字符切块 (通用)
    chunked_recursive_docs = chunk_documents(
        parsed_docs,
        strategy="recursive",
        chunk_size=500,
        chunk_overlap=50
    )
    save_docs_to_json(
        chunked_recursive_docs,
        os.path.join(OUTPUT_DIR, f"{filename}_chunked_recursive.json")
    )

    # 策略 B: 代码切块 (如果文件是代码)
    if INPUT_FILE.endswith(".py"):
        logging.info("\n检测到Python文件，执行代码切块...")
        chunked_code_docs = chunk_documents(
            parsed_docs, # 对于代码，可以直接用 TextLoader 加载，但这里为了统一流程也用 unstructured
            strategy="code",
            code_language=Language.PYTHON,
            chunk_size=800,
            chunk_overlap=100
        )
        save_docs_to_json(
            chunked_code_docs,
            os.path.join(OUTPUT_DIR, f"{filename}_chunked_code.json")
        )
    
    print(f"\n处理完成！所有输出文件已保存在 '{OUTPUT_DIR}' 文件夹中。")


if __name__ == "__main__":
    main() 