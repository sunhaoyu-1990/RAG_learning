import React, { useState } from 'react';
import { EmbeddingOptions, TextInput } from '../components/shared/ModelOptions';

const StdPage = () => {
  const [input, setInput] = useState('');
  const [result, setResult] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  
  const [embeddingOptions, setEmbeddingOptions] = useState({
    provider: 'huggingface',
    model: 'BAAI/bge-m3',
    dbName: 'finance_bge_m3',
    collectionName: 'finance_terms_bge_m3'
  });

  const handleEmbeddingOptionChange = (e) => {
    const { name, value } = e.target;
    setEmbeddingOptions(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async () => {
    setIsLoading(true);
    setError('');
    setResult('');
    try {
      const response = await fetch('http://192.168.0.75:8000/api/std', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          text: input, 
          embeddingOptions 
        }),
      });
      const data = await response.json();
      setResult(JSON.stringify(data, null, 2));
    } catch (error) {
      console.error('Error:', error);
      setError(`An error occurred: ${error.message}`);
    }
    setIsLoading(false);
  };

  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold mb-6">金融术语标准化 📚</h1>
      <div className="grid grid-cols-1 gap-6">
        {/* 左侧面板：文本输入和嵌入选项 */}
        <div className="bg-white shadow-md rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4">输入金融术语</h2>
          <TextInput
            value={input}
            onChange={(e) => setInput(e.target.value)}
            rows={4}
            placeholder="请输入需要标准化的金融术语..."
          />
          
          <EmbeddingOptions options={embeddingOptions} onChange={handleEmbeddingOptionChange} />

          <button
            onClick={handleSubmit}
            className="bg-purple-500 text-white px-4 py-2 rounded-md hover:bg-purple-600 w-full mt-4"
            disabled={isLoading}
          >
            {isLoading ? '处理中...' : '标准化术语'}
          </button>
        </div>
      </div>
      
      {/* 结果显示区域 */}
      {(error || result) && (
        <div className="mt-6">
          {error && (
            <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 mb-6" role="alert">
              <p className="font-bold">错误：</p>
              <p>{error}</p>
            </div>
          )}
          {result && (
            <div className="bg-green-100 border-l-4 border-green-500 text-green-700 p-4 mb-6" role="alert">
              <p className="font-bold">结果：</p>
              <pre>{result}</pre>
            </div>
          )}
        </div>
      )}

    </div>
  );
};

export default StdPage;