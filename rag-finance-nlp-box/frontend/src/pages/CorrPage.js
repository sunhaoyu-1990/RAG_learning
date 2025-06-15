import React, { useState } from 'react';
import { LLMOptions, TextInput } from '../components/shared/ModelOptions';

const CorrPage = () => {
  const [input, setInput] = useState('');
  const [result, setResult] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  
  // LLM options
  const [llmOptions, setLlmOptions] = useState({
    provider: 'openai',
    model: 'gpt-4o-mini'
  });

  const handleLlmOptionChange = (e) => {
    const { name, value } = e.target;
    setLlmOptions(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async () => {
    setIsLoading(true);
    setResult('');
    try {
      const response = await fetch('http://192.168.0.75:8000/api/corr', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: input,
          llmOptions
        }),
      });
      const data = await response.json();
      setResult(JSON.stringify(data, null, 2));
    } catch (error) {
      console.error('Error:', error);
      setResult('处理请求时发生错误。');
    }
    setIsLoading(false);
  };

  return (
    <div className="max-w-6xl mx-auto">
      <h1 className="text-3xl font-bold mb-6">金融记录纠错 📝</h1>
      
      <div className="grid grid-cols-3 gap-6">
        {/* Left panel: Text inputs */}
        <div className="col-span-2 bg-white shadow-md rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4">输入金融记录</h2>
          <TextInput
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="请输入需要纠错的金融记录..."
          />

          <button
            onClick={handleSubmit}
            className="bg-red-500 text-white px-4 py-2 rounded-md hover:bg-red-600 w-full"
            disabled={isLoading}
          >
            {isLoading ? '处理中...' : '纠正记录'}
          </button>
        </div>

        {/* Right panel: Options */}
        <div className="bg-white shadow-md rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4">选项</h2>
          
          {/* LLM Options */}
          <LLMOptions options={llmOptions} onChange={handleLlmOptionChange} />
        </div>
      </div>

      {/* Results */}
      {result && (
        <div className="mt-6">
          <div className="bg-blue-100 border-l-4 border-blue-500 text-blue-700 p-4 mb-6" role="alert">
            <p className="font-bold">结果：</p>
            <pre className="whitespace-pre-wrap">{result}</pre>
          </div>
        </div>
      )}

    </div>
  );
};

export default CorrPage;