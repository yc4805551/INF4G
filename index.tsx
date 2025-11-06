import React, { useState, useEffect, useRef, useCallback } from 'react';
import ReactDOM from 'react-dom/client';
import { GoogleGenAI, Type, GenerateContentResponse } from "@google/genai"; // (这个现在可以移除了，但保留也无妨)
import mammoth from 'mammoth';
import { marked } from 'marked';

// --- 关键修改：定义你的 API 后端地址 ---
// PROD = 生产环境 (GitHub Pages), DEV = 开发环境 (npm run dev)
// VITE_API_BASE_URL 将从 GitHub Secrets 注入
const API_BASE_URL = import.meta.env.PROD 
    ? import.meta.env.VITE_API_BASE_URL 
    : 'http://localhost:5000'; // 假设你的本地 app.py 运行在 5000 端口
// --- 修改结束 ---

// FIX: Modified debounce to return a function with a `clearTimeout` method to cancel pending calls.
const debounce = <F extends (...args: any[]) => any>(func: F, waitFor: number) => {
  let timeout: ReturnType<typeof setTimeout> | null = null;
  const debounced = (...args: Parameters<F>): void => {
    if (timeout) {
      clearTimeout(timeout);
    }
    timeout = setTimeout(() => func(...args), waitFor);
  };

  debounced.clearTimeout = () => {
    if (timeout) {
      clearTimeout(timeout);
      timeout = null;
    }
  };
  return debounced;
};

// Define structures for the analysis modes
interface NoteAnalysis {
  organizedText: string;
  userThoughts: string;
}

// Updated AuditIssue to support dynamic checklists and explanations
interface AuditIssue {
  problematicText: string;
  suggestion: string;
  checklistItem: string; // The rule from the checklist that was violated
  explanation: string; // Explanation of the issue
}

interface WritingSuggestion {
    originalText: string;
    revisedText: string;
    explanation: string;
}

interface Source {
  source_file: string;
  content_chunk: string;
  score: number;
}

// New interface for strongly typing roaming results
interface RoamingResultItem {
  source: string;
  relevantText: string;
  conclusion: string;
}

type NoteChatMessage = {
  role: 'user' | 'model';
  text: string;
  isError?: boolean;
  sources?: Source[];
  isComplete?: boolean;
};

type ModelProvider = 'gemini' | 'openai' | 'deepseek' | 'ali';
type ChatMessage = {
  role: 'user' | 'model';
  parts: { text: string }[];
  resultType?: 'notes';
  resultData?: NoteAnalysis;
};

// 移除了 getModelConfig (不再需要)

interface AuditResult {
    issues: AuditIssue[];
    error?: string;
    rawResponse?: string;
}

type AuditResults = {
    [key in ModelProvider]?: AuditResult
};

// --- 关键修改：重写 callGenerativeAi ---
// 它现在调用你的 Render 后端，而不是 Gemini
// (你需要一个非流式的 /api/generate 端点)
const callGenerativeAi = async (provider: ModelProvider, systemInstruction: string, userPrompt: string, jsonResponse: boolean, mode: 'notes' | 'audit' | 'roaming' | 'writing' | null, history: ChatMessage[] = []) => {
  
  const response = await fetch(`${API_BASE_URL}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
          provider, // 'gemini', 'openai' etc.
          systemInstruction,
          userPrompt,
          history,
          jsonResponse, // 告诉你的后端是否需要 JSON 格式
          mode
      })
  });

  if (!response.ok) {
      const errorText = await response.text().catch(() => response.statusText);
      let errorJson;
      try { errorJson = JSON.parse(errorText); } catch(e) {/* ignored */}
      throw new Error(`后端 API 错误: ${errorJson?.error || errorJson?.details || errorText}`);
  }

  const responseText = await response.text();
  return responseText;
};
// --- 修改结束 ---

// --- 关键修改：重写 callGenerativeAiStream ---
// 它现在调用你的 Render 后端
const callGenerativeAiStream = async (
    provider: ModelProvider,
    systemInstruction: string,
    userPrompt: string,
    history: ChatMessage[],
    onChunk: (textChunk: string) => void,
    onComplete: () => void,
    onError: (error: Error) => void
) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/generate-stream`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
              provider,
              systemInstruction,
              userPrompt,
              history
          })
      });

      if (!response.ok) {
          const errorText = await response.text().catch(() => response.statusText);
          let errorJson;
          try { errorJson = JSON.parse(errorText); } catch(e) {/* ignored */}
          throw new Error(`后端 API 错误: ${errorJson?.error || errorJson?.details || errorText}`);
      }

      if (!response.body) {
          throw new Error("Response body is null");
      }
      
      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          
          const textChunk = decoder.decode(value, { stream: true });
          onChunk(textChunk); // 直接将 Python 传来的块发给 UI
      }
      
      onComplete();
  } catch (error: any) {
      if (error instanceof TypeError && error.message.toLowerCase().includes('failed to fetch')) {
          onError(new Error(`网络请求失败。无法连接到你的后端服务 (${API_BASE_URL})。请确认它正在运行。`));
      } else {
          onError(error);
      }
  }
};
// --- 修改结束 ---

const ThoughtsInputModal = ({
  isOpen,
  onClose,
  onSubmit,
}: {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (thoughts: string) => void;
}) => {
  const [thoughts, setThoughts] = useState('');

  if (!isOpen) return null;

  const handleSubmit = () => {
    onSubmit(thoughts);
    setThoughts(''); // Reset for next time
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <h2>输入我的想法</h2>
        <p>在整理笔记前，您可以输入任何相关的想法、问题或待办事项。AI 会将这些内容与您的笔记一并智能整理。</p>
        <textarea
          className="modal-textarea"
          rows={5}
          value={thoughts}
          onChange={(e) => setThoughts(e.target.value)}
          placeholder="例如：这个概念需要进一步查证，下周三前完成..."
          autoFocus
        />
        <div className="modal-actions">
          <button className="btn btn-secondary" onClick={onClose}>
            取消
          </button>
          <button className="btn btn-primary" onClick={handleSubmit}>
            开始整理
          </button>
        </div>
      </div>
    </div>
  );
};

// 移除了 ApiKeyModal (不再需要)

const HomeInputView = ({
  inputText,
  setInputText,
  onOrganize,
  onAudit,
  selectedModel,
  setSelectedModel,
  isProcessing,
  knowledgeBases,
  isKbLoading,
  kbError,
  selectedKnowledgeBase,
  setSelectedKnowledgeBase,
  onKnowledgeChat,
  onWriting,
}: {
  inputText: string;
  setInputText: React.Dispatch<React.SetStateAction<string>>;
  onOrganize: () => void;
  onAudit: () => void;
  selectedModel: ModelProvider;
  setSelectedModel: (model: ModelProvider) => void;
  isProcessing: boolean;
  knowledgeBases: { id: string; name: string }[];
  isKbLoading: boolean;
  kbError: string | null;
  selectedKnowledgeBase: string | null;
  setSelectedKnowledgeBase: (id: string) => void;
  onKnowledgeChat: () => void;
  onWriting: () => void;
}) => {
    const lastPastedText = useRef('');
    const fileInputRef = useRef<HTMLInputElement>(null);

    useEffect(() => {
        const handleFocus = async () => {
            if (document.hasFocus()) {
                try {
                    const text = await navigator.clipboard.readText();
                    if (text && text !== lastPastedText.current && text !== inputText) {
                        setInputText(prev => prev ? `${prev}\n\n${text}` : text);
                        lastPastedText.current = text;
                    }
                } catch (err) {
                    console.log('Clipboard permission denied, or clipboard is empty.');
                }
            }
        };

        window.addEventListener('focus', handleFocus);

        return () => {
            window.removeEventListener('focus', handleFocus);
        };
    }, [inputText, setInputText]);

  const processFile = async (file: File) => {
    if (!file) return;
    const reader = new FileReader();

    reader.onload = async (event) => {
        const fileContent = event.target?.result;
        let text = '';
        if (file.name.endsWith('.docx')) {
            try {
                const result = await mammoth.extractRawText({ arrayBuffer: fileContent as ArrayBuffer });
                text = result.value;
            } catch (err) {
                console.error("Error reading docx file", err);
                alert("无法解析 DOCX 文件。");
                return;
            }
        } else {
            text = fileContent as string;
        }
        setInputText(prev => prev ? `${prev}\n\n--- ${file.name} ---\n${text}` : text);
    };
    
    if (file.name.endsWith('.docx')) {
        reader.readAsArrayBuffer(file);
    } else if (file.name.endsWith('.txt') || file.name.endsWith('.md')) {
        reader.readAsText(file);
    } else {
        alert("不支持的文件类型。请上传 .txt, .md 或 .docx 文件。");
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
        processFile(e.target.files[0]);
        e.target.value = '';
    }
  };

  const handleUploadClick = () => {
      fileInputRef.current?.click();
  };

  const handleDragOver = (e: React.DragEvent<HTMLTextAreaElement>) => {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.add('drag-over');
  };

  const handleDragLeave = (e: React.DragEvent<HTMLTextAreaElement>) => {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.remove('drag-over');
  };

  const handleDrop = async (e: React.DragEvent<HTMLTextAreaElement>) => {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.remove('drag-over');

    if (e.dataTransfer.files?.[0]) {
      await processFile(e.dataTransfer.files[0]);
      e.dataTransfer.clearData();
    }
  };

  const modelProviders: ModelProvider[] = ['gemini', 'openai', 'deepseek', 'ali'];

  return (
    <>
        <div className="home-grid-layout">
            <div className="home-panel">
                <h2>工作区</h2>
                <textarea
                    className="text-area"
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                    placeholder="在此处输入或拖放 .txt, .md, .docx 文件...&#10;从别处复制后，返回此页面可自动粘贴"
                    disabled={isProcessing}
                    style={{flexGrow: 1}}
                />
                 <input
                    type="file"
                    ref={fileInputRef}
                    style={{ display: 'none' }}
                    accept=".txt,.md,.docx,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    onChange={handleFileChange}
                />
                <div className="utility-btn-group">
                    <button className="btn btn-secondary" onClick={() => setInputText('')} disabled={!inputText || isProcessing}>
                        清空内容
                    </button>
                    <button className="btn btn-secondary" onClick={handleUploadClick} disabled={isProcessing}>
                        上传文件
                    </button>
                </div>
            </div>
            <div className="home-panel">
                <h2>全局配置</h2>
                <div className="config-group">
                    <h4>选择模型</h4>
                    <div className="model-selector-group">
                        {modelProviders.map(model => (
                            <button
                                key={model}
                                className={`model-btn ${selectedModel === model ? 'active' : ''}`}
                                onClick={() => setSelectedModel(model)}
                                disabled={isProcessing}
                            >
                                {model}
                            </button>
                        ))}
                    </div>
                </div>
                <div className="config-group">
                    <h4>选择知识库</h4>
                    {isKbLoading && <div className="spinner-container" style={{padding: '10px 0'}}><p>正在加载知识库...</p></div>}
                    {kbError && <div className="error-message" style={{textAlign: 'left'}}>{kbError}</div>}
                    {!isKbLoading && !kbError && (
                        knowledgeBases.length > 0 ? (
                            <div className="kb-selector-group">
                                {knowledgeBases.map(kb => (
                                    <button
                                        key={kb.id}
                                        className={`kb-selector-btn ${selectedKnowledgeBase === kb.id ? 'active' : ''}`}
                                        onClick={() => setSelectedKnowledgeBase(kb.id)}
                                        disabled={isProcessing}
                                    >
                                        {kb.name}
                                    </button>
                                ))}
                            </div>
                        ) : (
                            <p className="instruction-text">未找到可用的知识库。请检查后端服务和 Milvus 连接。</p>
                        )
                    )}
                </div>
                <div className="config-group">
                    <h4>API Keys</h4>
                <p className="instruction-text">API Keys 已移至后端服务器安全管理。</p>
                </div>
            </div>
        </div>
        <div className="home-actions-bar">
            <button className="action-btn" onClick={onOrganize} disabled={!inputText || isProcessing}>
                1. 整理笔记
            </button>
            <button className="action-btn" onClick={onAudit} disabled={!inputText || isProcessing}>
                2. 审阅文本
            </button>
            <button className="action-btn" onClick={onKnowledgeChat} disabled={!inputText || isProcessing || !selectedKnowledgeBase}>
                3. 知识库对话
            </button>
            <button className="action-btn" onClick={onWriting} disabled={isProcessing}>
                4. 沉浸式写作
            </button>
        </div>
    </>
  );
};


const NoteAnalysisView = ({
  analysisResult,
  isLoading: isInitialLoading,
  error,
  provider,
  originalText,
  selectedKnowledgeBaseId,
  knowledgeBases
}: {
  analysisResult: NoteAnalysis | null;
  isLoading: boolean;
  error: string | null;
  provider: ModelProvider;
  originalText: string;
  selectedKnowledgeBaseId: string | null;
  knowledgeBases: { id: string; name: string }[];
}) => {
  const [consolidatedText, setConsolidatedText] = useState('');
  
  // State for Chat
  const [chatHistory, setChatHistory] = useState<NoteChatMessage[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [isChatLoading, setIsChatLoading] = useState(false);
  const chatHistoryRef = useRef<HTMLDivElement>(null);

  // State for Roaming Notes
  const [isRoaming, setIsRoaming] = useState(false);
  const [roamingResult, setRoamingResult] = useState<RoamingResultItem[] | null>(null);
  const [roamingError, setRoamingError] = useState<string | null>(null);


  useEffect(() => {
    if (analysisResult) {
      const fullText = `【整理后】\n${analysisResult.organizedText}\n\n---\n\n【我的想法】\n${analysisResult.userThoughts}\n\n---\n\n【原文】\n${originalText}`;
      setConsolidatedText(fullText);
      // Initialize chat with a welcome message
      setChatHistory([{ role: 'model', text: '您好！您可以针对这篇笔记进行提问、要求修改，或者探讨更多想法。' }]);
    }
  }, [analysisResult, originalText]);

  useEffect(() => {
    if (chatHistoryRef.current) {
        chatHistoryRef.current.scrollTop = chatHistoryRef.current.scrollHeight;
    }
  }, [chatHistory]);

  const handleExportTXT = () => {
    if (!analysisResult) return;

    // Part 1: Main Content
    let content = `【笔记工作台】\n\n【整理后】\n${analysisResult.organizedText}\n\n---\n\n【我的想法】\n${analysisResult.userThoughts}`;
    
    // Part 2: Roaming Result
    if (roamingResult && roamingResult.length > 0) {
      content += `\n\n---\n\n【笔记漫游】`;
      roamingResult.forEach((result: RoamingResultItem, index: number) => {
        content += `\n\n--- 漫游结果 ${index + 1} ---\n`;
        content += `来源: ${result.source}\n\n`;
        content += `关联原文:\n${result.relevantText}\n\n`;
        content += `联想结论:\n${result.conclusion}`;
      });
    }
    
    // Part 3: Original Text
    content += `\n\n---\n\n【原文】\n${originalText}`;

    // Part 4: Chat History
    const chatContent = chatHistory.map(msg => {
        // Skip the initial welcome message from the model
        if (msg