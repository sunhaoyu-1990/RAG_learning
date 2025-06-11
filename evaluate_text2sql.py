import os
import sys
import json
import logging
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import nltk
import importlib

# 下载 NLTK 的 'punkt' 数据（如果尚未下载）
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# 1. 设置项目根目录，以便正确导入模块
# D:\BaiduSyncdisk\shy_product\rag-in-action
# 脚本位于 D:\BaiduSyncdisk\shy_product\rag-in-action\09-系统评估-Evaluation
# 需要导入的模块位于 D:\BaiduSyncdisk\shy_product\rag-in-action\05-检索前处理-PreRetrieval\01-查询构建\Text2SQL\Sakila
# 首先，将项目根目录添加到 sys.path
# rag-in-action
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 05-检索前处理-PreRetrieval\01-查询构建\Text2SQL\Sakila
module_path = os.path.join(project_root, "05-检索前处理-PreRetrieval", "01-查询构建", "Text2SQL", "Sakila")
sys.path.insert(0, module_path)

# 2. 导入目标函数
try:
    text2sql_module = importlib.import_module("05-text2sql-rag-v2-ok")
    text2sql = text2sql_module.text2sql
except ImportError as e:
    logging.error(f"无法导入 'text2sql' 函数: {e}")
    sys.exit(1)

# 3. 日志配置
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# 4. 评估函数
def normalize_sql(sql):
    """规范化SQL语句：小写，去除首尾空格"""
    return sql.lower().strip()

def calculate_token_recall(reference, candidate):
    """计算基于词符的召回率"""
    ref_tokens = set(word_tokenize(reference))
    can_tokens = set(word_tokenize(candidate))
    if not ref_tokens:
        return 1.0 if not can_tokens else 0.0
    
    true_positives = len(ref_tokens.intersection(can_tokens))
    return true_positives / len(ref_tokens)

def evaluate_text2sql():
    """评估 text2sql RAG 系统的性能"""
    # 加载测试数据
    test_data_path = os.path.join(project_root, "90-文档-Data", "sakila", "q2sql_pairs.json")
    try:
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    except FileNotFoundError:
        logging.error(f"测试数据文件未找到: {test_data_path}")
        return

    generated_sqls = []
    reference_sqls = []
    
    print(f"开始评估，共 {len(test_data)} 条测试数据...")

    # 遍历测试集
    for item in tqdm(test_data, desc="评估进度"):
        question = item['question']
        ground_truth_sql = normalize_sql(item['sql'])
        
        try:
            # 获取模型生成的SQL，不执行
            generated_sql = text2sql(question, execute=False)
            generated_sql = normalize_sql(generated_sql)
        except Exception as e:
            logging.error(f"处理问题 '{question}' 时发生错误: {e}")
            generated_sql = "" # 如果出错，视为空字符串
            
        generated_sqls.append(generated_sql)
        reference_sqls.append(ground_truth_sql)

    # 5. 计算指标
    exact_matches = 0
    bleu_scores = []
    recall_scores = []

    for ref, gen in zip(reference_sqls, generated_sqls):
        # 精确匹配率
        if ref == gen:
            exact_matches += 1
        
        # BLEU 分数
        ref_tokens = word_tokenize(ref)
        gen_tokens = word_tokenize(gen)
        bleu = sentence_bleu([ref_tokens], gen_tokens)
        bleu_scores.append(bleu)
        
        # 召回率
        recall = calculate_token_recall(ref, gen)
        recall_scores.append(recall)

    # 6. 打印评估结果
    print("\n--- 评估结果 ---")
    
    accuracy = (exact_matches / len(test_data)) * 100
    print(f"准确率 (Exact Match Accuracy): {accuracy:.2f}%")
    
    avg_bleu = (sum(bleu_scores) / len(bleu_scores))
    print(f"平均 BLEU 分数: {avg_bleu:.4f}")
    
    avg_recall = (sum(recall_scores) / len(recall_scores))
    print(f"平均词符召回率 (Token Recall): {avg_recall:.4f}")
    print("------------------")

# 7. 程序入口
if __name__ == "__main__":
    evaluate_text2sql() 