
import React, { useState, useEffect, useRef, useCallback } from 'react';
import ReactDOM from 'react-dom/client';
import mammoth from 'mammoth';
import { marked } from 'marked';
import { GoogleGenAI } from '@google/genai';


const API_BASE_URL = import.meta.env?.PROD
  ? `${import.meta.env.VITE_API_BASE_URL || ''}/api`
  : '/proxy-api';//doujunhao- 设置了后端连接

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

type ModelProvider = 'gemini' | 'openai' | 'deepseek' | 'ali' | 'depOCR' | 'doubao';
type ChatMessage = {
  role: 'user' | 'model';
  parts: { text: string }[];
  resultType?: 'notes';
  resultData?: NoteAnalysis;
};

type ExecutionMode = 'backend' | 'frontend';

// State for multi-model audit results
// FIX: Defined an interface for a single audit result to provide strong typing
// for what was previously an anonymous object structure, resolving 'unknown' type errors.
interface AuditResult {
    issues: AuditIssue[];
    error?: string;
    rawResponse?: string;
}

type AuditResults = {
    [key in ModelProvider]?: AuditResult
};

const frontendApiConfig: Record<string, {
    apiKey?: string;
    endpoint?: string;
    model?: string;
}> = {
    gemini: {
        apiKey: import.meta.env?.VITE_GEMINI_API_KEY,
        model: 'gemini-2.5-flash',
    },
    openai: {
        apiKey: import.meta.env?.VITE_OPENAI_API_KEY,
        endpoint: import.meta.env?.VITE_OPENAI_TARGET_URL ? `${import.meta.env.VITE_OPENAI_TARGET_URL}/v1/chat/completions` : undefined,
        model: import.meta.env?.VITE_OPENAI_MODEL,
    },
    deepseek: {
        apiKey: import.meta.env?.VITE_DEEPSEEK_API_KEY,
        endpoint: import.meta.env?.VITE_DEEPSEEK_ENDPOINT,
        model: import.meta.env?.VITE_DEEPSEEK_MODEL,
    },
    ali: {
        apiKey: import.meta.env?.VITE_ALI_API_KEY,
        endpoint: import.meta.env?.VITE_ALI_ENDPOINT || (import.meta.env?.VITE_ALI_TARGET_URL ? `${import.meta.env.VITE_ALI_TARGET_URL}/v1/chat/completions` : undefined),
        model: import.meta.env?.VITE_ALI_MODEL,
    },
    depOCR: {
        apiKey: import.meta.env?.VITE_DEPOCR_API_KEY,
        endpoint: import.meta.env?.VITE_DEPOCR_ENDPOINT,
        model: import.meta.env?.VITE_DEPOCR_MODEL,
    },
    doubao: {
        apiKey: import.meta.env?.VITE_DOUBAO_API_KEY,
        endpoint: import.meta.env?.VITE_DOUBAO_ENDPOINT,
        model: import.meta.env?.VITE_DOUBAO_MODEL,
    },
};

async function callOpenAiCompatibleApi(
    apiKey: string,
    endpoint: string,
    model: string,
    systemInstruction: string,
    userPrompt: string,
    history: ChatMessage[],
    jsonResponse: boolean,
    images?: { base64: string, mimeType: string }[],
) {
    const userMessageContent: any[] = [{ type: 'text', text: userPrompt }];
    if (images && images.length > 0) {
        images.forEach(image => {
            userMessageContent.push({
                type: 'image_url',
                image_url: { url: `data:${image.mimeType};base64,${image.base64}` }
            });
        });
    }

    const messages = [
        { role: 'system', content: systemInstruction },
        ...history.map(h => ({
            role: h.role === 'model' ? 'assistant' : 'user',
            content: h.parts[0].text
        })),
        { role: 'user', content: userMessageContent }
    ];

    const body: any = {
        model,
        messages,
        stream: false,
    };
    if (images && images.length > 0) {
        body.max_tokens = 4096;
    }

    if (jsonResponse) {
        body.response_format = { type: 'json_object' };
    }

    const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${apiKey}`
        },
        body: JSON.stringify(body)
    });

    if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`API Error: ${response.status} ${response.statusText} - ${errorText}`);
    }

    const result = await response.json();
    return result.choices[0].message.content;
}

async function callOpenAiCompatibleApiStream(
    apiKey: string,
    endpoint: string,
    model: string,
    systemInstruction: string,
    userPrompt: string,
    history: ChatMessage[],
    onChunk: (textChunk: string) => void,
    onComplete: () => void,
    onError: (error: Error) => void,
) {
    try {
        const messages = [
            { role: 'system', content: systemInstruction },
            ...history.map(h => ({
                role: h.role === 'model' ? 'assistant' : 'user',
                content: h.parts[0].text
            })),
            { role: 'user', content: userPrompt }
        ];

        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${apiKey}`
            },
            body: JSON.stringify({
                model,
                messages,
                stream: true,
            })
        });

        if (!response.ok || !response.body) {
            const errorText = await response.text().catch(() => `Status: ${response.status}`);
            throw new Error(`Streaming Error: ${errorText}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || ''; // Keep the last, possibly incomplete line

            for (const line of lines) {
                if (line.trim().startsWith('data: ')) {
                    const dataStr = line.substring(6).trim();
                    if (dataStr === '[DONE]') {
                        onComplete();
                        return;
                    }
                    try {
                        const data = JSON.parse(dataStr);
                        const content = data.choices?.[0]?.delta?.content;
                        if (content) {
                            onChunk(content);
                        }
                    } catch (e) {
                        console.error("Error parsing stream data chunk:", dataStr, e);
                    }
                }
            }
        }
        onComplete();
    } catch (error: any) {
        onError(error);
    }
}

const callGenerativeAi = async (
    provider: ModelProvider,
    executionMode: ExecutionMode,
    systemInstruction: string,
    userPrompt: string,
    jsonResponse: boolean,
    mode: 'notes' | 'audit' | 'roaming' | 'writing' | 'ocr' | null,
    history: ChatMessage[] = [],
    images?: { base64: string, mimeType: string }[],
) => {
    if (executionMode === 'frontend') {
        const config = frontendApiConfig[provider];
        if (!config.model) {
            throw new Error(`Frontend Direct mode for ${provider} is not configured: model is missing.`);
        }

        if (provider === 'gemini') {
            if (!config.apiKey) {
                throw new Error(`Frontend Direct mode for ${provider} is not configured. Please set VITE_GEMINI_API_KEY in your environment.`);
            }
            const ai = new GoogleGenAI({ apiKey: config.apiKey });

            const userParts: any[] = [{ text: userPrompt }];
            if (images && images.length > 0) {
                const imageParts = images.map(img => ({
                    inlineData: {
                        mimeType: img.mimeType,
                        data: img.base64,
                    }
                }));
                userParts.unshift(...imageParts);
            }
            const fullContents = [...history, { role: 'user', parts: userParts }];


            const response = await ai.models.generateContent({
                model: (images && images.length > 0) ? 'gemini-2.5-flash' : config.model, // Use vision model if image is present
                contents: fullContents as any, // Cast to any to align with SDK expectations
                config: {
                    systemInstruction: systemInstruction,
                    responseMimeType: jsonResponse ? 'application/json' : undefined
                }
            });
            return response.text;
        } else { // OpenAI-compatible
            if (!config.apiKey) {
                throw new Error(`Frontend Direct mode for ${provider} is not configured. Please set VITE_${provider.toUpperCase()}_API_KEY in your environment.`);
            }
            if (!config.endpoint) {
                throw new Error(`Frontend Direct mode for ${provider} is not configured. Please set the endpoint URL in your environment.`);
            }
            return callOpenAiCompatibleApi(
                config.apiKey,
                config.endpoint,
                config.model,
                systemInstruction,
                userPrompt,
                history,
                jsonResponse,
                images,
            );
        }

    } else { // Backend mode
        const retries = 2; // 1 initial attempt + 2 retries
        for (let i = 0; i <= retries; i++) {
            try {
                const response = await fetch(`${API_BASE_URL}/generate`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ provider, systemInstruction, userPrompt, jsonResponse, mode, history, images })
                });
                
                if (!response.ok) {
                    const errorText = await response.text().catch(() => response.statusText);
                    let userFriendlyError = `后端代理服务出错 (状态码: ${response.status})。请检查后端服务日志。`;
                    if (response.status >= 500 && response.status < 600) {
                        userFriendlyError += ` 这可能是由于后端无法连接到上游AI服务导致的。`;
                    }
                    console.error("Backend raw error:", errorText);
                    throw new Error(userFriendlyError);
                }
                return await response.text();

            } catch (error) {
                console.error(`Attempt ${i + 1} failed for ${provider}:`, error);
                if (i === retries) {
                    if (error instanceof TypeError && error.message.toLowerCase().includes('failed to fetch')) {
                        throw new Error(`网络请求失败。无法连接到后端服务(${API_BASE_URL})。请检查网络连接、VPN配置或确认后端服务正在运行。`);
                    }
                    throw error; // Re-throw the last error
                }
                await new Promise(res => setTimeout(res, 1000));
            }
        }
        throw new Error('All retry attempts failed.');
    }
};

// New function for streaming chat responses
const callGenerativeAiStream = async (
    provider: ModelProvider,
    executionMode: ExecutionMode,
    systemInstruction: string,
    userPrompt: string,
    history: ChatMessage[],
    onChunk: (textChunk: string) => void,
    onComplete: () => void,
    onError: (error: Error) => void,
    thinkingBudget?: number,
) => {
    try {
        if (executionMode === 'frontend') {
            const config = frontendApiConfig[provider];
            if (!config.model) {
                throw new Error(`Frontend Direct mode for ${provider} is not configured. Model is missing.`);
            }
            
            if (provider === 'gemini') {
                if (!config.apiKey) {
                    throw new Error(`Frontend Direct mode for ${provider} is not configured. Please set VITE_GEMINI_API_KEY in your environment.`);
                }
                const ai = new GoogleGenAI({ apiKey: config.apiKey });
                const fullContents = [...history, { role: 'user', parts: [{ text: userPrompt }] }];
                
                const streamResult = await ai.models.generateContentStream({
                    model: config.model,
                    contents: fullContents as any, // Cast to any to align with SDK
                    config: { systemInstruction: systemInstruction }
                });

                for await (const chunk of streamResult) {
                    onChunk(chunk.text);
                }
                onComplete();
            } else { // OpenAI-compatible
                if (!config.apiKey) {
                    throw new Error(`Frontend Direct mode for ${provider} is not configured. Please set VITE_${provider.toUpperCase()}_API_KEY in your environment.`);
                }
                if (!config.endpoint) {
                    throw new Error(`Frontend Direct mode for ${provider} is not configured. Please set the endpoint URL in your environment.`);
                }
                await callOpenAiCompatibleApiStream(
                    config.apiKey,
                    config.endpoint,
                    config.model,
                    systemInstruction,
                    userPrompt,
                    history,
                    onChunk,
                    onComplete,
                    onError
                );
            }

        } else { // Backend mode
            const response = await fetch(`${API_BASE_URL}/generate-stream`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ provider, systemInstruction, userPrompt, history, thinkingBudget })
            });

            if (!response.ok || !response.body) {
                const errorText = await response.text().catch(() => `Status: ${response.status}`);
                throw new Error(`后端流式传输错误: ${errorText}`);
            }
            
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                onChunk(decoder.decode(value, { stream: true }));
            }
            onComplete();
        }
    } catch (error: any) {
        if (error instanceof TypeError && error.message.toLowerCase().includes('failed to fetch')) {
            onError(new Error(`网络请求失败。无法连接到后端服务(${API_BASE_URL})。请检查网络连接、VPN配置或确认后端服务正在运行。`));
        } else {
            onError(error);
        }
    }
};

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
  onTextRecognition,
  executionMode,
  setExecutionMode,
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
  onTextRecognition: () => void;
  executionMode: ExecutionMode;
  setExecutionMode: (mode: ExecutionMode) => void;
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

  const modelProviders: ModelProvider[] = ['gemini', 'openai', 'deepseek', 'ali', 'depOCR', 'doubao'];

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
                    <h4>执行模式</h4>
                    <div className="model-selector-group">
                        <button
                            className={`model-btn ${executionMode === 'backend' ? 'active' : ''}`}
                            onClick={() => setExecutionMode('backend')}
                            disabled={isProcessing}
                        >
                            后端代理
                        </button>
                        <button
                            className={`model-btn ${executionMode === 'frontend' ? 'active' : ''}`}
                            onClick={() => setExecutionMode('frontend')}
                            disabled={isProcessing}
                        >
                            前端直连
                        </button>
                    </div>
                    {executionMode === 'frontend' && (
                        <p className="instruction-text">前端直连模式将直接在浏览器中调用 AI 服务。请确保已在环境中配置了相应模型的 API Keys。</p>
                    )}
                </div>
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
                3. 内参对话
            </button>
            <button className="action-btn" onClick={onWriting} disabled={isProcessing}>
                4. 沉浸写作
            </button>
            <button className="action-btn" onClick={onTextRecognition} disabled={isProcessing}>
                5. 文本识别
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
  knowledgeBases,
  executionMode,
}: {
  analysisResult: NoteAnalysis | null;
  isLoading: boolean;
  error: string | null;
  provider: ModelProvider;
  originalText: string;
  selectedKnowledgeBaseId: string | null;
  knowledgeBases: { id: string; name: string }[];
  executionMode: ExecutionMode;
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
        if (msg.role === 'model' && msg.text.startsWith('您好！您可以针对这篇笔记进行提问')) {
            return '';
        }
        const role = msg.role === 'user' ? 'User' : 'Model';
        return `[${role}]\n${msg.text}`;
    }).filter(Boolean).join('\n\n');

    if (chatContent) {
        content += `\n\n---\n\n【多轮问答】\n${chatContent}`;
    }

    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `笔记整理与讨论 - ${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleStartRoaming = async () => {
    if (!selectedKnowledgeBaseId || !analysisResult) {
        if (!selectedKnowledgeBaseId) {
            alert("请返回首页选择一个知识库以开始笔记漫游。");
        }
        return;
    }

    setIsRoaming(true);
    setRoamingError(null);
    setRoamingResult(null);

    try {
        // Step 1: Call local backend to get relevant context
        const backendResponse = await fetch(`${API_BASE_URL}/find-related`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: analysisResult.organizedText,
                collection_name: selectedKnowledgeBaseId,
                top_k: 3
            })
        });

        if (!backendResponse.ok) {
            const errorText = await backendResponse.text().catch(() => backendResponse.statusText);
            let errorJson;
            if (errorText) {
                try {
                    errorJson = JSON.parse(errorText);
                } catch (e) { /* Not JSON */ }
            }
            throw new Error(`知识库查询失败: ${errorJson?.error || errorText}`);
        }

        const responseText = await backendResponse.text();
        if (!responseText) {
            throw new Error("知识库查询返回为空。");
        }
        
        let backendData;
        try {
            backendData = JSON.parse(responseText);
        } catch (e: any) {
            console.error('Error parsing backend JSON:', responseText);
            throw new Error(`Backend returned invalid JSON: ${e.message}`);
        }
        
        if (backendData.error) {
            throw new Error(`知识库返回错误: ${backendData.error}`);
        }
        
        const sources: Source[] = backendData.related_documents || [];

        if (sources.length === 0) {
            setRoamingError("在知识库中未找到足够相关的内容来进行漫游联想。");
            setIsRoaming(false);
            return;
        }

        // Step 2: Call Generative AI for each source to create a conclusion
        const systemInstruction = `You are an AI assistant skilled at synthesizing information. Based on a user's note and a relevant passage from their knowledge base, create an "Associative Conclusion" connecting the two ideas. Your entire response must be a JSON object with one key: "conclusion" (your generated associative summary).`;
        
        const roamingPromises = sources.map(async (source: Source) => {
            const userPrompt = `[Relevant Passage from Knowledge Base]:\n${source.content_chunk}\n\n[User's Original Note]:\n${analysisResult.organizedText}`;
            const genAiResponseText = await callGenerativeAi(provider, executionMode, systemInstruction, userPrompt, true, 'roaming');
            const result = JSON.parse(genAiResponseText.replace(/```json\n?|\n?```/g, ''));

            if (!result.conclusion) {
                throw new Error("AI model did not return a valid conclusion for one of the documents.");
            }
            
            return {
                source: source.source_file,
                relevantText: source.content_chunk,
                conclusion: result.conclusion,
            };
        });

        const newRoamingResults = await Promise.all(roamingPromises);
        setRoamingResult(newRoamingResults);

    } catch (err: any) {
        setRoamingError(`笔记漫游失败: ${err.message}`);
    } finally {
        setIsRoaming(false);
    }
  };
  
  const handleSendChatMessage = async (e?: React.FormEvent) => {
      e?.preventDefault();
      if (!chatInput.trim() || isChatLoading || !analysisResult) return;

      const newUserMessage: NoteChatMessage = { role: 'user', text: chatInput };
      const currentChatHistory = [...chatHistory, newUserMessage];
      setChatHistory(currentChatHistory);
      setChatInput('');
      setIsChatLoading(true);

      const systemInstruction = `You are a helpful assistant. The user has just organized a note and wants to discuss it. The note's organized content is provided below. Your role is to answer questions, help refine the text, or brainstorm ideas based on this note. Be helpful and conversational.\n\n--- NOTE START ---\n${analysisResult.organizedText}\n--- NOTE END ---`;
      
      const chatHistoryForApi = currentChatHistory
        .slice(0, -1) // Exclude the user message we just added
        .filter(msg => !(msg.role === 'model' && msg.text.startsWith('您好！您可以针对这篇笔记进行提问'))) // Exclude the initial UI-only message
        .map(msg => ({ 
            role: msg.role as 'user' | 'model',
            parts: [{ text: msg.text }]
        }));

      const modelResponse: NoteChatMessage = { role: 'model', text: '' };
      setChatHistory(prev => [...prev, modelResponse]);

      try {
          await callGenerativeAiStream(
              provider,
              executionMode,
              systemInstruction,
              chatInput,
              chatHistoryForApi,
              (chunk) => {
                  setChatHistory(prev => {
                      const newHistory = [...prev];
                      if(newHistory.length > 0) {
                        newHistory[newHistory.length - 1].text += chunk;
                      }
                      return newHistory;
                  });
              },
              () => {
                  setIsChatLoading(false);
              },
              (error) => {
                  setChatHistory(prev => {
                      const newHistory = [...prev];
                      if(newHistory.length > 0) {
                        newHistory[newHistory.length - 1].text = `抱歉，出错了: ${error.message}`;
                        newHistory[newHistory.length - 1].isError = true;
                      }
                      return newHistory;
                  });
                  setIsChatLoading(false);
              }
          );
      } catch (error: any) {
           setChatHistory(prev => {
              const newHistory = [...prev];
              if(newHistory.length > 0) {
                newHistory[newHistory.length - 1].text = `抱歉，出错了: ${error.message}`;
                newHistory[newHistory.length - 1].isError = true;
              }
              return newHistory;
          });
          setIsChatLoading(false);
      }
  };

  if (isInitialLoading) {
      return (
          <div className="spinner-container">
              <div className="spinner large"></div>
              <p style={{ marginTop: '16px', color: '#a0a0a0' }}>正在整理，请稍候...</p>
          </div>
      );
  }
  if (error) {
      return <div className="error-message" style={{ textAlign: 'left', whiteSpace: 'pre-wrap' }}>{error}</div>;
  }
  if (!analysisResult) {
      return <div className="large-placeholder">分析结果将显示在此处。</div>;
  }

  return (
      <div className="note-analysis-layout">
        <div className="note-content-panel">
            <h2 style={{textTransform: 'capitalize'}}>笔记工作台 (由 {provider} 模型生成)</h2>
            <div className="note-content-scrollable-area">
                <textarea
                    readOnly
                    className="text-area consolidated-note-display"
                    value={consolidatedText}
                ></textarea>
                <div className="content-section" style={{padding: '16px', backgroundColor: 'var(--background-color)'}}>
                    <h3>笔记漫游</h3>
                    {!roamingResult && !isRoaming && !roamingError && <p className="instruction-text">如需基于笔记内容进行关联联想，请在首页选择知识库后，点击下方“开始笔记漫游”按钮。</p>}
                    {isRoaming && <div className="spinner-container" style={{padding: '20px 0'}}><div className="spinner"></div></div>}
                    {roamingError && <div className="error-message">{roamingError}</div>}
                    {roamingResult && (
                        <div className="roaming-results-container">
                            {roamingResult.map((result: RoamingResultItem, index: number) => (
                                <div key={index} className="roaming-result">
                                    <p><strong>来源 ({index + 1}):</strong> {result.source}</p>
                                    <p><strong>关联原文:</strong> {result.relevantText}</p>
                                    <p><strong>联想结论:</strong> {result.conclusion}</p>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>
            <div className="card-bottom-actions">
                <div className="button-group">
                    <button className="btn btn-secondary" onClick={handleStartRoaming} disabled={isRoaming || !selectedKnowledgeBaseId}>
                        {isRoaming ? '漫游中...' : `开始笔记漫游 (使用 ${provider})`}
                    </button>
                </div>
                <div className="button-group" style={{marginLeft: 'auto'}}>
                    <button className="btn btn-secondary" onClick={handleExportTXT}>导出 TXT</button>
                </div>
            </div>
        </div>
        <div className="note-chat-panel">
            <h2>多轮问答</h2>
            <div className="kb-chat-history" ref={chatHistoryRef}>
                {chatHistory.map((msg, index) => (
                    <div key={index} className={`kb-message ${msg.role} ${msg.isError ? 'error' : ''}`}>
                        <p>{msg.text}</p>
                    </div>
                ))}
                {isChatLoading && chatHistory[chatHistory.length - 1]?.role === 'model' && !chatHistory[chatHistory.length - 1]?.text && (
                    <div className="spinner-container" style={{padding: '10px 0'}}><div className="spinner"></div></div>
                )}
            </div>
            <form className="chat-input-form" onSubmit={handleSendChatMessage}>
                <textarea
                    className="chat-input"
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSendChatMessage(); }}}
                    placeholder="针对笔记提问..."
                    rows={2}
                    disabled={isChatLoading}
                />
                <button type="submit" className="btn btn-primary send-btn" disabled={isChatLoading || !chatInput.trim()}>发送</button>
            </form>
        </div>
      </div>
  );
};

const parseJsonResponse = <T,>(responseText: string): { data: T | null, error?: string, rawResponse?: string } => {
    let parsedData: T | null = null;
    let jsonString = responseText.trim();

    const tryParse = (str: string): T | null => {
        try {
            const fixedStr = str.replace(/,\s*([}\]])/g, '$1');
            const result = JSON.parse(fixedStr);
            return result as T;
        } catch {
            try {
                const result = JSON.parse(str);
                return result as T;
            } catch {
                return null;
            }
        }
    };

    parsedData = tryParse(jsonString);

    if (!parsedData) {
        const markdownMatch = jsonString.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
        if (markdownMatch && markdownMatch[1]) {
            parsedData = tryParse(markdownMatch[1].trim());
        }
    }

    if (!parsedData) {
        const firstBrace = jsonString.indexOf('{');
        const lastBrace = jsonString.lastIndexOf('}');
        const firstBracket = jsonString.indexOf('[');
        const lastBracket = jsonString.lastIndexOf(']');

        let startIndex = -1, endIndex = -1;
        
        if (firstBracket !== -1 && lastBracket > firstBracket) {
            startIndex = firstBracket;
            endIndex = lastBracket;
        } else if (firstBrace !== -1 && lastBrace > firstBrace) {
            startIndex = firstBrace;
            endIndex = lastBrace;
        }
        
        if (startIndex !== -1) {
            parsedData = tryParse(jsonString.substring(startIndex, endIndex + 1));
        }
    }

    if (parsedData === null) {
        const lowercasedResponse = responseText.toLowerCase();
        if (Array.isArray([] as T) && (lowercasedResponse.includes('no issues found') || lowercasedResponse.includes('没有发现') || lowercasedResponse.includes('未发现'))) {
            return { data: [] as T };
        }
        return { 
            data: null, 
            error: "未能将模型响应解析为有效的JSON。", 
            rawResponse: responseText 
        };
    }
    return { data: parsedData };
};


const parseAuditResponse = (responseText: string): { issues: AuditIssue[], error?: string, rawResponse?: string } => { 
  // 1. 'parseError' 和 'parsedRawResponse' 仅在 parseJsonResponse 本身失败时才会被设置 
  const { data, error: parseError, rawResponse: parsedRawResponse } = parseJsonResponse<unknown>(responseText); 

  // 2. 如果 parseJsonResponse 失败，则将其错误和原始响应向上传递 
  if (!data) { 
    return { issues: [], error: parseError, rawResponse: parsedRawResponse }; 
  }

  let issuesArray: AuditIssue[] = [];

  // 3. 检查 data 是否直接就是一个数组 
  if (Array.isArray(data)) {
    issuesArray = data as AuditIssue[];
  }
  // 4. 否则，检查 data 是否是一个包含 'issues' 数组的对象 
  else if (typeof data === 'object' && data !== null && 'issues' in data && Array.isArray((data as { issues: any }).issues)) {
    issuesArray = (data as { issues: AuditIssue[] }).issues;
  }
  // 5. 【关键修复】如果两种都不是，说明格式错误 
  else { 
    // 我们现在将原始的 'responseText' 作为 rawResponse 传递回去，以便调试 
    return { issues: [], error: "模型返回了意外的 JSON 格式 (既不是数组，也不是包含 'issues' 的对象)。", rawResponse: responseText }; 
  }

  // 6. 现在我们安全地筛选 issuesArray 
  const validIssues = issuesArray.filter(issue =>
    issue &&
    typeof issue.problematicText === 'string' &&
    typeof issue.suggestion === 'string' &&
    typeof issue.checklistItem === 'string' &&
    typeof issue.explanation === 'string' &&
    issue.problematicText.trim()
  );
  return { issues: validIssues };
};

const AuditView = ({
    initialText,
    selectedModel,
    executionMode,
} : {
    initialText: string;
    selectedModel: ModelProvider;
    executionMode: ExecutionMode;
}) => {
    const [text] = useState(initialText);
    const [auditResults, setAuditResults] = useState<AuditResults>({});
    const [isLoading, setIsLoading] = useState(false);
    const [checklist, setChecklist] = useState<string[]>([
        '全文错别字',
        '全文中文语法问题',
        '文中逻辑不合理的地方',
        '学术名词是否前后一致'
    ]);
    const [selectedIssueId, setSelectedIssueId] = useState<string | null>(null);
    const textDisplayRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (selectedIssueId && textDisplayRef.current) {
            const element = textDisplayRef.current.querySelector(`[data-issue-id="${selectedIssueId}"]`);
            if (element) {
                element.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        }
    }, [selectedIssueId]);

    const handleChecklistItemChange = (index: number, value: string) => {
        const newChecklist = [...checklist];
        newChecklist[index] = value;
        setChecklist(newChecklist);
    };

    const addChecklistItem = () => setChecklist([...checklist, '']);
    const removeChecklistItem = (index: number) => setChecklist(checklist.filter((_, i) => i !== index));

    const handleAudit = async () => {
        setIsLoading(true);
        setAuditResults({});
        setSelectedIssueId(null);
        const model = selectedModel;

        const systemInstruction = `You are a professional editor. Analyze the provided text based ONLY on the rules in the following checklist. For each issue you find, return a JSON object with "problematicText" (the exact, verbatim text segment from the original), "suggestion" (your proposed improvement), "checklistItem" (the specific rule from the checklist that was violated), and "explanation" (a brief explanation of why it's a problem). Your entire response MUST be a single JSON array of these objects, or an empty array [] if no issues are found.

[Checklist]:
- ${checklist.filter(item => item.trim()).join('\n- ')}
`;
        const userPrompt = `[Text to Audit]:\n\n${text}`;
        
        try {
            const responseText = await callGenerativeAi(model, executionMode, systemInstruction, userPrompt, true, 'audit');
            const { issues, error, rawResponse } = parseAuditResponse(responseText);
            setAuditResults({ [model]: { issues, error, rawResponse } });
            
        } catch (err: any) {
            console.error(`Error auditing with ${model}:`, err);
            setAuditResults({ [model]: { issues: [], error: err.message } });
        } finally {
            setIsLoading(false);
        }
    };

    const handleAuditAll = async () => {
        setIsLoading(true);
        setAuditResults({});
        setSelectedIssueId(null);

        const allModels: ModelProvider[] = ['gemini', 'openai', 'deepseek', 'ali', 'depOCR', 'doubao'];

        const systemInstruction = `You are a professional editor. Analyze the provided text based ONLY on the rules in the following checklist. For each issue you find, return a JSON object with "problematicText" (the exact, verbatim text segment from the original), "suggestion" (your proposed improvement), "checklistItem" (the specific rule from the checklist that was violated), and "explanation" (a brief explanation of why it's a problem). Your entire response MUST be a single JSON array of these objects, or an empty array [] if no issues are found.

[Checklist]:
- ${checklist.filter(item => item.trim()).join('\n- ')}
`;
        const userPrompt = `[Text to Audit]:\n\n${text}`;

        const auditPromises = allModels.map(model => 
            callGenerativeAi(model, executionMode, systemInstruction, userPrompt, true, 'audit')
        );
        
        const results = await Promise.allSettled(auditPromises);
        
        const newAuditResults: AuditResults = {};
        results.forEach((result, index) => {
            const model = allModels[index];
            if (result.status === 'fulfilled') {
                const { issues, error, rawResponse } = parseAuditResponse(result.value);
                newAuditResults[model] = { issues, error, rawResponse };
            } else {
                newAuditResults[model] = { issues: [], error: (result.reason as Error).message };
            }
        });

        setAuditResults(newAuditResults);
        setIsLoading(false);
    };

    // FIX: Explicitly cast the result of Object.entries to fix type inference
    // issues where 'result' was being inferred as 'unknown'.
    const allIssuesWithIds = (Object.entries(auditResults) as [string, AuditResult | undefined][]).flatMap(([model, result]) => {
        return result?.issues?.map((issue: AuditIssue, index: number) => ({
            ...issue,
            model: model as ModelProvider,
            id: `${model}-${index}`
        })) ?? [];
    });

    const renderOriginalTextWithHighlight = () => {
        if (!text) return <div className="large-placeholder">审阅结果将显示在此处。</div>;
        const selectedIssue = selectedIssueId ? allIssuesWithIds.find(i => i.id === selectedIssueId) : null;
        if (!selectedIssue) {
            return <div className="audit-text-display">{text}</div>;
        }
        const term = selectedIssue.problematicText;
        if (!term || term.trim() === '') {
             return <div className="audit-text-display">{text}</div>;
        }
        try {
            const regex = new RegExp(`(${term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'g');
            const parts = text.split(regex);
            let firstMatch = true;
            return (
                <div className="audit-text-display">
                    {parts.map((part, index) => {
                        if (part === term) {
                            const idToAssign = firstMatch ? selectedIssue.id : undefined;
                            firstMatch = false;
                            return (
                                <span key={index} className="selected-highlight" data-issue-id={idToAssign}>
                                    {part}
                                </span>
                            );
                        }
                        return <React.Fragment key={index}>{part}</React.Fragment>;
                    })}
                </div>
            );
        } catch (e) {
            console.error("Regex error in highlighting:", e);
            return <div className="audit-text-display">{text}</div>;
        }
    };

    // FIX: Explicitly cast the result of Object.values to fix type inference
    // issues where 'res' was being inferred as 'unknown'.
    const hasAnyIssues = (Object.values(auditResults) as (AuditResult | undefined)[]).some(res => (res?.issues?.length ?? 0) > 0);
    const hasAnyErrors = (Object.values(auditResults) as (AuditResult | undefined)[]).some(res => !!(res?.error));

    return (
        <div className="audit-view-container">
            <div className="audit-config-panel">
                <h2 style={{textTransform: 'capitalize'}}>审阅清单</h2>
                <div className="checklist-editor">
                    {checklist.map((item, index) => (
                        <div key={index} className="checklist-item">
                            <input
                                type="text"
                                value={item}
                                onChange={(e) => handleChecklistItemChange(index, e.target.value)}
                                placeholder={`规则 #${index + 1}`}
                                disabled={isLoading}
                            />
                            <button onClick={() => removeChecklistItem(index)} disabled={isLoading}>-</button>
                        </div>
                    ))}
                    <button className="btn btn-secondary" onClick={addChecklistItem} disabled={isLoading}>+ 添加规则</button>
                </div>
                <div className="audit-button-group">
                     <button className="btn btn-primary audit-start-btn" onClick={handleAudit} disabled={isLoading || !text}>
                        {isLoading ? <span className="spinner"></span> : null}
                        {isLoading ? '审阅中...' : `审阅 (${selectedModel})`}
                    </button>
                    <button className="btn btn-primary audit-start-btn" onClick={handleAuditAll} disabled={isLoading || !text}>
                        {isLoading ? <span className="spinner"></span> : null}
                        {isLoading ? '审阅中...' : '全部模型审阅'}
                    </button>
                </div>
                <div className="audit-status-area">
                    {/* FIX: Explicitly cast the result of Object.entries to fix type inference
                    // issues where 'result' was being inferred as 'unknown'. */}
                    { (Object.entries(auditResults) as [string, AuditResult | undefined][]).map(([model, result]: [string, AuditResult | undefined]) => {
                        if (!result) return null;
                        return (
                        <div key={model} className="audit-status-item">
                            <span className={`model-indicator model-${model}`}>{model}</span>
                            {result.error 
                                ? <span className="status-error">失败: {result.error}</span>
                                : <span className="status-success">完成 ({result.issues.length}个问题)</span>
                            }
                        </div>
                    )})}
                </div>
            </div>

            <div className="audit-results-panel">
                <div className="content-section audit-original-text-section">
                    <h2>原始文本</h2>
                    <div className="original-text-container" ref={textDisplayRef}>
                       {isLoading && !Object.keys(auditResults).length ? <div className="spinner-container"><div className="spinner large"></div><p>正在调用模型，请稍候...</p></div> : renderOriginalTextWithHighlight()}
                    </div>
                </div>
                <div className="content-section audit-issues-section">
                    <h2>审核问题</h2>
                    <div className="issues-list-container">
                        {!isLoading && Object.keys(auditResults).length > 0 && !hasAnyIssues && !hasAnyErrors && <div className="large-placeholder">未发现任何问题。</div>}
                        {/* FIX: Explicitly cast the result of Object.entries to fix type inference
                        // issues where 'result' was being inferred as 'unknown'. */}
                        { (Object.entries(auditResults) as [string, AuditResult | undefined][]).map(([model, result]: [string, AuditResult | undefined]) => {
                            if (!result) return null;

                            if (result.error && result.rawResponse) {
                                return (
                                    <details key={`${model}-error`} open className="issue-group">
                                        <summary className={`issue-group-summary model-border-${model}`}>
                                            <span className={`model-indicator model-${model}`}>{model}</span> (解析失败)
                                        </summary>
                                        <div className="issue-group-content">
                                            <div className="issue-card raw-response-card">
                                                <div className="issue-card-header">原始模型响应 (Raw Model Response)</div>
                                                <div className="issue-card-body">
                                                    <pre className="raw-response-text">{result.rawResponse}</pre>
                                                </div>
                                            </div>
                                        </div>
                                    </details>
                                );
                            }

                            if (result.issues.length === 0) return null;
                            
                            return (
                                <details key={model} open className="issue-group">
                                    <summary className={`issue-group-summary model-border-${model}`}>
                                        <span className={`model-indicator model-${model}`}>{model}</span> ({result.issues.length}个问题)
                                    </summary>
                                    <div className="issue-group-content">
                                    {result.issues.map((issue: AuditIssue, index: number) => {
                                        const issueId = `${model}-${index}`;
                                        return (
                                            <div
                                                key={issueId}
                                                className={`issue-card ${selectedIssueId === issueId ? 'selected' : ''}`}
                                                onClick={() => setSelectedIssueId(issueId)}
                                                tabIndex={0}
                                                onKeyDown={(e) => { if(e.key === 'Enter' || e.key === ' ') setSelectedIssueId(issueId)}}
                                            >
                                                <div className="issue-card-header">{issue.checklistItem}</div>
                                                <div className="issue-card-body">
                                                    <p><strong>原文:</strong> {issue.problematicText}</p>
                                                    <p><strong>建议:</strong> {issue.suggestion}</p>
                                                    <p><strong>说明:</strong> {issue.explanation}</p>
                                                </div>
                                            </div>
                                        );
                                    })}
                                    </div>
                                </details>
                            );
                        })}
                    </div>
                </div>
            </div>
        </div>
    );
};

// By moving this pure function outside the component, we prevent it from being
// recreated on every render, which is a minor performance optimization.
const parseMessageText = (text: string) => {
    if (!text) return '';
    const textWithCitations = text.replace(/\[Source: (.*?)\]/g, (match, filename) => {
        return `<a href="#" class="source-citation" data-filename="${filename.trim()}">${match}</a>`;
    });
    return marked.parse(textWithCitations, { gfm: true, breaks: true }) as string;
};

const KnowledgeChatView = ({
  knowledgeBaseId,
  knowledgeBaseName,
  initialQuestion,
  provider,
  executionMode,
}: {
  knowledgeBaseId: string;
  knowledgeBaseName: string;
  initialQuestion?: string;
  provider: ModelProvider;
  executionMode: ExecutionMode;
}) => {
  const [chatHistory, setChatHistory] = useState<NoteChatMessage[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [isChatLoading, setIsChatLoading] = useState(false);
  const chatHistoryRef = useRef<HTMLDivElement>(null);
  const isInitialQuestionSent = useRef(false);

  useEffect(() => {
    setChatHistory([{ role: 'model', text: `您好！已连接到“${knowledgeBaseName}”知识库。每次提问我都会优先从知识库中寻找答案。`, isComplete: true }]);
  }, [knowledgeBaseName]);
  
  useEffect(() => {
    if (chatHistoryRef.current) {
        chatHistoryRef.current.scrollTop = chatHistoryRef.current.scrollHeight;
    }
  }, [chatHistory]);
  
  const handleExportTXT = () => {
    const content = chatHistory.map(msg => {
        if (msg.role === 'model' && msg.text.startsWith('您好！已连接到')) {
            return '';
        }
        let entry = '';
        if (msg.role === 'user') {
            entry += `[User]\n${msg.text}\n\n`;
        } else { // model
            entry += `[Model]\n${msg.text}\n`;
            if (msg.sources && msg.sources.length > 0) {
                entry += `\n--- 参考源头信息 ---\n`;
                msg.sources.forEach(source => {
                    entry += `  - 文件: ${source.source_file}\n`;
                    entry += `    Relevance: ${source.score.toFixed(2)}\n`;
                    entry += `    内容片段: "${source.content_chunk}"\n\n`;
                });
            } else {
                entry += '\n';
            }
        }
        return entry;
    }).filter(Boolean).join('---\n\n');

    if (!content.trim()) {
        alert("没有可导出的对话内容。");
        return;
    }

    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `知识库对话 - ${knowledgeBaseName} - ${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };
  
  const handleSendChatMessage = useCallback(async (e?: React.FormEvent, messageOverride?: string) => {
    e?.preventDefault();
    const messageToSend = messageOverride || chatInput;
    if (!messageToSend.trim() || isChatLoading) return;

    const newUserMessage: NoteChatMessage = { role: 'user', text: messageToSend, isComplete: true };
    setChatHistory(prev => [...prev, newUserMessage]);
    if (!messageOverride) {
      setChatInput('');
    }
    setIsChatLoading(true);
    
    const placeholderMessage: NoteChatMessage = { role: 'model', text: '', isComplete: false };
    setChatHistory(prev => [...prev, placeholderMessage]);

    try {
      // Step 1: ALWAYS query the knowledge base
      const backendResponse = await fetch(`${API_BASE_URL}/find-related`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: messageToSend, collection_name: knowledgeBaseId, top_k: 5 })
      });
      if (!backendResponse.ok) {
        const errorText = await backendResponse.text().catch(() => backendResponse.statusText);
        throw new Error(`知识库查询失败: ${errorText}`);
      }
      
      const result = await backendResponse.json();
      if (result.error) throw new Error(`知识库返回错误: ${result.error}`);

      const retrievedSources: Source[] = result.related_documents || [];
      const finalSources: Source[] | undefined = retrievedSources;
      
      const context = retrievedSources.map((s: Source) => `
<document>
  <source>${s.source_file}</source>
  <content>
    ${s.content_chunk}
  </content>
</document>
`).join('');
      
      const systemInstruction = `You are a helpful Q&A assistant. Answer the user's question based ONLY on the provided documents.
- Structure your answer clearly using Markdown formatting (like lists, bold text, etc.).
- For each piece of information or claim in your answer, you MUST cite its origin by appending "[Source: file_name.txt]" at the end of the sentence.
- You must use the exact filename from the <source> tag of the document you used.
- If the information comes from multiple sources, cite them all, like "[Source: file1.txt], [Source: file2.txt]".
- If you cannot answer the question from the documents, state that clearly. Do not use outside knowledge.`;
      
      const userPrompt = `[DOCUMENTS]${context}\n\n[USER QUESTION]\n${messageToSend}`;
      const chatHistoryForApi: ChatMessage[] = []; // No history is passed for KB-mode to force focus on provided context

      await callGenerativeAiStream(
          provider, executionMode, systemInstruction, userPrompt, chatHistoryForApi,
          (chunk) => {
              setChatHistory(prev => {
                  const newHistory = [...prev];
                  const lastMessage = newHistory[newHistory.length - 1];
                  if (lastMessage?.role === 'model') {
                      lastMessage.text += chunk;
                  }
                  return newHistory;
              });
          },
          () => { // onComplete
               setChatHistory(prev => {
                  const newHistory = [...prev];
                  const lastMessage = newHistory[newHistory.length - 1];
                  if (lastMessage?.role === 'model') {
                      lastMessage.sources = finalSources;
                      lastMessage.isComplete = true;
                  }
                  return newHistory;
              });
              setIsChatLoading(false);
          },
          (error) => { throw error; }
      );

    } catch (error: any) {
        setChatHistory(prev => {
            const newHistory = [...prev];
            const lastMessage = newHistory[newHistory.length - 1];
            if (lastMessage?.role === 'model') {
                lastMessage.text = `抱歉，处理时出错了: ${error.message}`;
                lastMessage.isError = true;
                lastMessage.isComplete = true;
            }
            return newHistory;
        });
        setIsChatLoading(false);
    }
  }, [chatInput, isChatLoading, knowledgeBaseId, provider, executionMode]);

  useEffect(() => {
    if (initialQuestion && !isInitialQuestionSent.current) {
      isInitialQuestionSent.current = true;
      handleSendChatMessage(undefined, initialQuestion);
    }
  }, [initialQuestion, handleSendChatMessage]);
  
  const handleHistoryClick = (e: React.MouseEvent<HTMLDivElement>) => {
      const target = e.target as HTMLElement;
      if (target.classList.contains('source-citation')) {
          e.preventDefault();
          const filename = target.dataset.filename;
          if (!filename) return;

          const messageElement = target.closest('.kb-message');
          if (!messageElement) return;

          const sourceItem = messageElement.querySelector(`.source-item[data-filename="${filename}"]`) as HTMLLIElement;
          
          if (sourceItem) {
              const details = sourceItem.closest('details');
              if (details && !details.open) {
                  details.open = true;
              }
              
              sourceItem.scrollIntoView({ behavior: 'smooth', block: 'center' });
              
              sourceItem.classList.add('highlighted');
              setTimeout(() => {
                  sourceItem.classList.remove('highlighted');
              }, 2500);
          }
      }
  };

  return (
    <div className="kb-view-container">
        <div className="view-header-row">
          <h2 style={{textTransform: 'capitalize'}}>知识库对话: {knowledgeBaseName} (由 {provider} 模型生成)</h2>
          <button className="btn btn-secondary" onClick={handleExportTXT} disabled={chatHistory.length <= 1}>
            导出 TXT
          </button>
        </div>
        <div className="kb-chat-history" ref={chatHistoryRef} onClick={handleHistoryClick}>
            {chatHistory.map((msg, index) => (
                <div key={index} className={`kb-message ${msg.role} ${msg.isError ? 'error' : ''}`}>
                    <div className="avatar-icon">{msg.role === 'user' ? 'U' : 'M'}</div>
                    <div className="message-content">
                        {(msg.role === 'model' && msg.isComplete)
                            ? <div dangerouslySetInnerHTML={{ __html: parseMessageText(msg.text) }} />
                            : <p>{msg.text}</p>
                        }
                        {msg.sources && msg.sources.length > 0 && (
                            <details className="source-info-box" open>
                                <summary>参考源头信息 ({msg.sources.length})</summary>
                                <ul className="source-list">
                                    {msg.sources.map((source, i) => (
                                        <li key={i} className="source-item" data-filename={source.source_file}>
                                            <div className="source-header">
                                                <span className="source-filename">{source.source_file}</span>
                                                <span className="source-score">Relevance: {source.score.toFixed(2)}</span>
                                            </div>
                                            <p className="source-chunk">"{source.content_chunk}"</p>
                                        </li>
                                    ))}
                                </ul>
                            </details>
                        )}
                    </div>
                </div>
            ))}
             {isChatLoading && chatHistory[chatHistory.length - 1]?.role === 'model' && !chatHistory[chatHistory.length - 1]?.text && (
                <div className="spinner-container" style={{padding: '10px 0'}}><div className="spinner"></div></div>
            )}
        </div>
        <form className="chat-input-form" onSubmit={handleSendChatMessage}>
            <textarea
                className="chat-input"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSendChatMessage(); }}}
                placeholder="在此输入您的问题..."
                rows={2}
                disabled={isChatLoading}
            />
            <button type="submit" className="btn btn-primary send-btn" disabled={isChatLoading || !chatInput.trim()}>发送</button>
        </form>
    </div>
  );
};

const DiffView = ({ originalText, revisedText }: { originalText: string; revisedText: string }) => {
    const diff = (a: string[], b: string[]) => {
        const matrix = Array(a.length + 1).fill(null).map(() => Array(b.length + 1).fill(0));
        for (let i = 1; i <= a.length; i++) {
            for (let j = 1; j <= b.length; j++) {
                if (a[i - 1] === b[j - 1]) {
                    matrix[i][j] = matrix[i - 1][j - 1] + 1;
                } else {
                    matrix[i][j] = Math.max(matrix[i - 1][j], matrix[i][j - 1]);
                }
            }
        }
        let i = a.length;
        let j = b.length;
        const result: { value: string; type: 'common' | 'removed' | 'added' }[] = [];
        while (i > 0 || j > 0) {
            if (i > 0 && j > 0 && a[i - 1] === b[j - 1]) {
                result.unshift({ value: a[i - 1], type: 'common' });
                i--;
                j--;
            } else if (j > 0 && (i === 0 || matrix[i][j - 1] >= matrix[i - 1][j])) {
                result.unshift({ value: b[j - 1], type: 'added' });
                j--;
            } else if (i > 0 && (j === 0 || matrix[i][j - 1] < matrix[i - 1][j])) {
                result.unshift({ value: a[i - 1], type: 'removed' });
                i--;
            } else {
                break;
            }
        }
        return result;
    };

    const diffResult = diff(originalText.split(/(\s+)/), revisedText.split(/(\s+)/));

    return (
        <div className="diff-view">
            {diffResult.map((part, index) => {
                if (part.type === 'added') {
                    return <span key={index} className="diff-add">{part.value}</span>;
                }
                if (part.type === 'removed') {
                    return <span key={index} className="diff-remove">{part.value}</span>;
                }
                return <span key={index}>{part.value}</span>;
            })}
        </div>
    );
};


const WritingView = ({
  initialText,
  onTextChange,
  selectedModel,
  selectedKnowledgeBase,
  knowledgeBases,
  executionMode,
}: {
  initialText: string;
  onTextChange: (newText: string) => void;
  selectedModel: ModelProvider;
  selectedKnowledgeBase: string | null;
  knowledgeBases: { id: string; name: string }[];
  executionMode: ExecutionMode;
}) => {
  const [text, setText] = useState(initialText);
  const [suggestions, setSuggestions] = useState<WritingSuggestion[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [styleReferenceText, setStyleReferenceText] = useState('');
  const [copyMessage, setCopyMessage] = useState<{text: string; type: 'success' | 'error'} | null>(null);

  const [kbResults, setKbResults] = useState<Source[] | null>(null);
  const [isKbSearching, setIsKbSearching] = useState(false);
  const [kbError, setKbError] = useState<string | null>(null);
  const [selectedSuggestionIndex, setSelectedSuggestionIndex] = useState<number | null>(null);

  // New state for chat
  const [chatHistory, setChatHistory] = useState<NoteChatMessage[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [isChatLoading, setIsChatLoading] = useState(false);
  const chatHistoryRef = useRef<HTMLDivElement>(null);

  const editorRef = useRef<HTMLTextAreaElement>(null);
  const styleFileRef = useRef<HTMLInputElement>(null);
  const suppressSuggestionFetch = useRef(false);
  const fetchIdRef = useRef(0);

  // For resizable panel and responsive layout
  const containerRef = useRef<HTMLDivElement>(null);
  const isResizingRef = useRef(false);
  const [panelGridStyle, setPanelGridStyle] = useState({ gridTemplateColumns: '2fr 8px 1fr' });
  const [isMobileLayout, setIsMobileLayout] = useState(window.innerWidth <= 1000);

  const debouncedResizeHandler = useRef(
    debounce(() => {
      setIsMobileLayout(window.innerWidth <= 1000);
    }, 200)
  ).current;

  useEffect(() => {
    const handler = () => debouncedResizeHandler();
    window.addEventListener('resize', handler);
    handler(); // Initial check
    return () => {
      window.removeEventListener('resize', handler);
      debouncedResizeHandler.clearTimeout();
    };
  }, [debouncedResizeHandler]);


  useEffect(() => {
    // Initialize chat with a welcome message
    setChatHistory([{ role: 'model', text: '您好！我是您的写作助手，您可以随时向我提问。' }]);
  }, []);

  useEffect(() => {
    if (chatHistoryRef.current) {
        chatHistoryRef.current.scrollTop = chatHistoryRef.current.scrollHeight;
    }
  }, [chatHistory]);

  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    isResizingRef.current = true;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
  };

  const handleMouseUp = () => {
    isResizingRef.current = false;
    document.body.style.cursor = 'default';
    document.body.style.userSelect = 'auto';
    window.removeEventListener('mousemove', handleMouseMove);
    window.removeEventListener('mouseup', handleMouseUp);
  };

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (isResizingRef.current && containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect();
      const editorPanelWidth = e.clientX - rect.left;
      // Subtract resizer width from the assistant panel width
      const assistantPanelWidth = rect.right - e.clientX - 8; 
      
      // Set min widths to prevent collapsing
      if (editorPanelWidth > 300 && assistantPanelWidth > 300) { 
        setPanelGridStyle({
          gridTemplateColumns: `${editorPanelWidth}px 8px ${assistantPanelWidth}px`
        });
      }
    }
  }, []);

  useEffect(() => {
    // Cleanup event listeners when component unmounts
    return () => {
        window.removeEventListener('mousemove', handleMouseMove);
        window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [handleMouseMove]);


  const fetchSuggestions = useCallback(debounce(async (currentText: string, styleText: string) => {
    if (currentText.trim().length < 50) { // Don't run on very short text
      setSuggestions([]);
      return;
    }
    const fetchId = ++fetchIdRef.current;
    setIsLoading(true);
    setError(null);
    setSelectedSuggestionIndex(null);

    let systemInstruction = `你是一位专业的中文写作助理。你的任务是实时帮助用户改进他们的中文写作。
- 分析所提供的文本，并找出最多6个关键的改进点。
- 保持文档原有的语调和风格。
- 针对每一条建议，提供精准的原文片段 ("originalText")、你修改后的版本 ("revisedText")，以及简明扼要的修改说明 ("explanation")。
- 你所有的输出，包括建议和说明，都必须是中文。
- 你的整个响应必须是一个JSON对象数组，每个对象包含 "originalText"、"revisedText" 和 "explanation" 这三个键。
- 如果文本写得很好，无需修改，请返回一个空数组 []。`;

    if (styleText.trim()) {
        systemInstruction += `\n\n- 重要：你必须严格遵循以下“写作风格参考”文档中的写作风格、语气和词汇。

[写作风格参考]:
---
${styleText.trim()}
---
`;
    }

    const userPrompt = `[Text for Analysis]:\n\n${currentText}`;

    try {
      const responseText = await callGenerativeAi(selectedModel, executionMode, systemInstruction, userPrompt, true, 'writing');
      
      if (fetchId !== fetchIdRef.current) return;

      const { data, error: parseError, rawResponse } = parseJsonResponse<unknown>(responseText);

      if (parseError || !data) {
        console.error("Raw response on parse error:", rawResponse);
        throw new Error(parseError || "Received invalid data from model.");
      }
      
      let suggestionsArray: WritingSuggestion[] = [];

      if (Array.isArray(data)) {
        suggestionsArray = data as WritingSuggestion[];
      }
      else if (typeof data === 'object' && data !== null && 'suggestions' in data && Array.isArray((data as { suggestions: any }).suggestions)) {
        suggestionsArray = (data as { suggestions: WritingSuggestion[] }).suggestions;
      }
      else {
        throw new Error("Model returned an unexpected JSON format (not an array, or an object with 'suggestions').");
      }

      const validSuggestions = suggestionsArray.filter(s => s.originalText && s.revisedText && s.explanation);
      setSuggestions(validSuggestions);

    } catch (err: any) {
      if (fetchId === fetchIdRef.current) {
          setError(`获取建议失败: ${err.message}`);
          setSuggestions([]);
      }
    } finally {
      if (fetchId === fetchIdRef.current) {
          setIsLoading(false);
      }
    }
  }, 1500), [selectedModel, executionMode]);

  useEffect(() => {
    onTextChange(text);
    if (suppressSuggestionFetch.current) {
        suppressSuggestionFetch.current = false;
        return;
    }
    fetchSuggestions(text, styleReferenceText);
  }, [text, styleReferenceText, fetchSuggestions, onTextChange]);

  const handleApplySuggestion = (suggestion: WritingSuggestion) => {
    suppressSuggestionFetch.current = true;
    setText(prevText => prevText.replace(suggestion.originalText, suggestion.revisedText));
    setSuggestions(prev => prev.filter(s => s !== suggestion));
    setSelectedSuggestionIndex(null);
  };
  
  const handleRefresh = () => {
      fetchSuggestions.clearTimeout();
      fetchSuggestions(text, styleReferenceText);
  };
  
  const handleSuggestionClick = (suggestion: WritingSuggestion, index: number) => {
    setSelectedSuggestionIndex(index);
    if (editorRef.current) {
        const fullText = editorRef.current.value;
        const startIndex = fullText.indexOf(suggestion.originalText);
        if (startIndex !== -1) {
            const endIndex = startIndex + suggestion.originalText.length;
            editorRef.current.focus();
            editorRef.current.setSelectionRange(startIndex, endIndex);
        }
    }
  };

  const handleCopy = () => {
    if (!text) return;

    const copyLegacy = () => {
        const textArea = document.createElement("textarea");
        textArea.value = text;
        // Make the textarea invisible and out of the viewport
        textArea.style.position = 'fixed';
        textArea.style.top = '-9999px';
        textArea.style.left = '-9999px';
        
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();

        try {
            const successful = document.execCommand('copy');
            if (successful) {
                setCopyMessage({ text: '已复制!', type: 'success' });
            } else {
                setCopyMessage({ text: '复制失败!', type: 'error' });
            }
        } catch (err) {
            console.error('Fallback copy method failed:', err);
            setCopyMessage({ text: '复制失败!', type: 'error' });
        }

        document.body.removeChild(textArea);
        setTimeout(() => setCopyMessage(null), 2000);
    };
    
    // Use modern clipboard API if available and in a secure context
    if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(() => {
            setCopyMessage({ text: '已复制!', type: 'success' });
            setTimeout(() => setCopyMessage(null), 2000);
        }).catch(err => {
            console.error('Clipboard API failed, trying fallback:', err);
            copyLegacy();
        });
    } else {
        console.warn('Clipboard API not available, using fallback.');
        copyLegacy();
    }
  };
  
  const handleKbSearch = async () => {
    if (!selectedKnowledgeBase) {
        setKbError("请返回首页选择一个知识库。");
        return;
    }
    if (text.trim().length < 20) {
        setKbError("请写入更多内容以便进行有效检索。");
        return;
    }
    setIsKbSearching(true);
    setKbError(null);
    setKbResults(null);
    try {
        const backendResponse = await fetch(`${API_BASE_URL}/find-related`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text, collection_name: selectedKnowledgeBase, top_k: 3 })
        });
        if (!backendResponse.ok) {
            const errorText = await backendResponse.text();
            throw new Error(errorText || "知识库查询失败。");
        }
        const data = await backendResponse.json();
        if (data.error) throw new Error(data.error);
        const sources = data.related_documents || [];
        setKbResults(sources);
        if (sources.length === 0) setKbError("未找到相关内容。");
    } catch (err: any) {
        setKbError(`知识库检索出错: ${err.message}`);
    } finally {
        setIsKbSearching(false);
    }
  };

  const handleSendWritingChatMessage = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!chatInput.trim() || isChatLoading) return;

    const newUserMessage: NoteChatMessage = { role: 'user', text: chatInput };
    const currentHistory = [...chatHistory, newUserMessage];
    setChatHistory(currentHistory);
    setChatInput('');
    setIsChatLoading(true);

    const systemInstruction = `You are a writing assistant. The user is currently writing the following text and has a question about it. Your role is to answer questions, help refine the text, or brainstorm ideas based on this text. Be helpful and conversational.\n\n--- CURRENT TEXT ---\n${text}\n--- END TEXT ---`;
    
    const chatHistoryForApi = currentHistory
      .slice(0, -1)
      .filter(msg => !(msg.role === 'model' && msg.text.startsWith('您好！我是您的写作助手')))
      .map(msg => ({ role: msg.role, parts: [{ text: msg.text }] }));
    
    const modelResponse: NoteChatMessage = { role: 'model', text: '' };
    setChatHistory(prev => [...prev, modelResponse]);

    try {
        await callGenerativeAiStream(
            selectedModel, executionMode, systemInstruction, chatInput, chatHistoryForApi,
            (chunk) => {
                setChatHistory(prev => {
                    const newHistory = [...prev];
                    const lastMessage = newHistory[newHistory.length - 1];
                    if (lastMessage?.role === 'model') {
                        lastMessage.text += chunk;
                    }
                    return newHistory;
                });
            },
            () => { setIsChatLoading(false); },
            (error) => {
                setChatHistory(prev => {
                    const newHistory = [...prev];
                    const lastMessage = newHistory[newHistory.length - 1];
                    if (lastMessage?.role === 'model') {
                        lastMessage.isError = true;
                        lastMessage.text = `抱歉，出错了: ${error.message}`;
                    }
                    return newHistory;
                });
                setIsChatLoading(false);
            }
        );
    } catch (error: any) {
        setChatHistory(prev => {
            const newHistory = [...prev];
            const lastMessage = newHistory[newHistory.length - 1];
            if (lastMessage?.role === 'model') {
                lastMessage.isError = true;
                lastMessage.text = `抱歉，出错了: ${error.message}`;
            }
            return newHistory;
        });
        setIsChatLoading(false);
    }
  };


  const processStyleFile = async (file: File) => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = async (event) => {
        const fileContent = event.target?.result;
        let fileText = '';
        if (file.name.endsWith('.docx')) {
            try {
                const result = await mammoth.extractRawText({ arrayBuffer: fileContent as ArrayBuffer });
                fileText = result.value;
            } catch (err) {
                setError("无法解析 DOCX 文件。");
                return;
            }
        } else {
            fileText = fileContent as string;
        }
        setStyleReferenceText(fileText);
    };
    if (file.name.endsWith('.docx')) {
        reader.readAsArrayBuffer(file);
    } else if (file.name.endsWith('.txt') || file.name.endsWith('.md')) {
        reader.readAsText(file);
    } else {
        alert("不支持的文件类型。请上传 .txt, .md 或 .docx 文件。");
    }
  };

  const handleStyleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
        processStyleFile(e.target.files[0]);
        e.target.value = '';
    }
  };

  const handleUploadStyleClick = () => {
      styleFileRef.current?.click();
  };

  return (
    <div 
        ref={containerRef} 
        className={`writing-view-container ${isMobileLayout ? 'mobile-layout' : ''}`}
        style={!isMobileLayout ? panelGridStyle : undefined}
    >
      <div className="writing-editor-panel">
         <div className="assistant-panel-header">
            <h2>沉浸式写作</h2>
            <div className="header-actions">
                {copyMessage && <span className={`copy-message ${copyMessage.type}`}>{copyMessage.text}</span>}
                <button className="btn btn-secondary" onClick={handleCopy} disabled={!text}>
                    复制全文
                </button>
            </div>
        </div>
        <textarea
          ref={editorRef}
          className="text-area writing-editor"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="在此开始写作，AI 将在您停顿时提供建议..."
        />
      </div>
      
      {!isMobileLayout && <div className="resizer" onMouseDown={handleMouseDown}></div>}

      <div className="writing-assistant-panel">
        <div className="assistant-panel-header">
            <h2 style={{textTransform: 'capitalize'}}>AI 助手 ({selectedModel})</h2>
            <button className="btn btn-secondary" onClick={handleRefresh} disabled={isLoading}>
                {isLoading ? <span className="spinner"></span> : null}
                刷新建议
            </button>
        </div>
        <div className="assistant-content">
          <div className="suggestions-container">
            {isLoading && <div className="spinner-container"><div className="spinner large" /></div>}
            {!isLoading && error && <div className="error-message">{error}</div>}
            {!isLoading && !error && suggestions.length === 0 && (
              <div className="large-placeholder">
                <p>暂无建议。</p>
                <p className="instruction-text">请继续写作，或确保文本长度超过50个字符以便AI分析。</p>
              </div>
            )}
            {!isLoading && !error && suggestions.length > 0 && (
              <div className="suggestions-list">
                {suggestions.map((s, i) => (
                  <div 
                    key={i} 
                    className={`suggestion-card ${selectedSuggestionIndex === i ? 'selected' : ''}`}
                    onClick={() => handleSuggestionClick(s, i)}
                  >
                    <div className="suggestion-body">
                      <p><strong>差异对比:</strong></p>
                      <DiffView originalText={s.originalText} revisedText={s.revisedText} />
                      <p style={{marginTop: '8px'}}><strong>说明:</strong> {s.explanation}</p>
                    </div>
                    <div className="suggestion-actions">
                      <button className="btn btn-primary" onClick={(e) => { e.stopPropagation(); handleApplySuggestion(s); }}>
                        应用此建议
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
          <div className="writing-chat-container">
              <h4>连续对话</h4>
              <div className="kb-chat-history" ref={chatHistoryRef}>
                {chatHistory.map((msg, index) => (
                    <div key={index} className={`kb-message ${msg.role} ${msg.isError ? 'error' : ''}`}>
                         <div className="message-content" style={{padding: '8px 12px', maxWidth: '100%'}}>
                            <p>{msg.text}</p>
                        </div>
                    </div>
                ))}
                {isChatLoading && chatHistory[chatHistory.length - 1]?.role === 'model' && !chatHistory[chatHistory.length - 1]?.text && (
                    <div className="spinner-container" style={{padding: '10px 0'}}><div className="spinner"></div></div>
                )}
              </div>
              <form className="chat-input-form" onSubmit={handleSendWritingChatMessage}>
                <textarea
                    className="chat-input"
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSendWritingChatMessage(); }}}
                    placeholder="就当前文本提问..."
                    rows={1}
                    disabled={isChatLoading}
                />
                <button type="submit" className="btn btn-primary send-btn" disabled={isChatLoading || !chatInput.trim()}>发送</button>
              </form>
          </div>
        </div>
        <div className="style-reference-section">
            <h4>写作风格参考 (可选)</h4>
            <textarea
                className="text-area style-reference-textarea"
                value={styleReferenceText}
                onChange={(e) => setStyleReferenceText(e.target.value)}
                placeholder="在此处粘贴范文，或上传文件，以固定AI的写作风格..."
            />
             <input
                type="file"
                ref={styleFileRef}
                style={{ display: 'none' }}
                accept=".txt,.md,.docx,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                onChange={handleStyleFileChange}
            />
            <div className="utility-btn-group" style={{justifyContent: 'flex-end'}}>
                 <button className="btn btn-secondary" onClick={handleUploadStyleClick}>
                    上传风格文件
                </button>
                <button className="btn btn-secondary" onClick={() => setStyleReferenceText('')} disabled={!styleReferenceText}>
                    清空风格参考
                </button>
            </div>
        </div>
      </div>
    </div>
  );
};

const blobToBase64 = (blob: Blob): Promise<string> => {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => {
            const base64String = reader.result?.toString().split(',')[1];
            if (base64String) {
                resolve(base64String);
            } else {
                reject(new Error("Failed to convert blob to base64"));
            }
        };
        reader.onerror = reject;
        reader.readAsDataURL(blob);
    });
};

const TextRecognitionView = ({ provider, executionMode }: { provider: ModelProvider; executionMode: ExecutionMode; }) => {
    const [files, setFiles] = useState<File[]>([]);
    const [recognizedText, setRecognizedText] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [progressMessage, setProgressMessage] = useState('');
    const [recognitionModel, setRecognitionModel] = useState<ModelProvider | null>(null);

    const [chatHistory, setChatHistory] = useState<NoteChatMessage[]>([]);
    const [chatInput, setChatInput] = useState('');
    const [isChatLoading, setIsChatLoading] = useState(false);
    const chatHistoryRef = useRef<HTMLDivElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    
    const chatModelProviders: ModelProvider[] = ['gemini', 'openai', 'deepseek', 'ali', 'doubao'];
    const [chatProvider, setChatProvider] = useState<ModelProvider>(
      chatModelProviders.includes(provider) ? provider : 'gemini'
    );

    useEffect(() => {
        setChatHistory([{ role: 'model', text: '您好！上传文档并识别后，您可以在此就识别出的文本内容进行提问。' }]);
    }, []);

    useEffect(() => {
        if (chatHistoryRef.current) {
            chatHistoryRef.current.scrollTop = chatHistoryRef.current.scrollHeight;
        }
    }, [chatHistory]);
    
    const handleClear = () => {
        setFiles([]);
        setRecognizedText('');
        setError(null);
        setProgressMessage('');
        setChatHistory([{ role: 'model', text: '您好！上传文档并识别后，您可以在此就识别出的文本内容进行提问。' }]);
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    };

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const selectedFiles = e.target.files;
        if (!selectedFiles || selectedFiles.length === 0) return;

        setRecognizedText('');
        setError(null);
        setProgressMessage('');

        const newFiles = Array.from(selectedFiles);
        // FIX: Explicitly type 'f' as File to resolve an inference error where its type was 'unknown'.
        const supportedFiles = newFiles.filter((f: File) => 
            f.type.startsWith('image/') || f.type === 'application/pdf'
        );

        if (supportedFiles.length < newFiles.length) {
            setError("已过滤不支持的文件类型。请上传 PNG, JPG, 或 PDF 文件。");
        }

        setFiles(prev => [...prev, ...supportedFiles]);
    };
    
    const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        e.stopPropagation();
        e.currentTarget.classList.add('drag-over');
    };

    const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        e.stopPropagation();
        e.currentTarget.classList.remove('drag-over');
    };

    const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        e.stopPropagation();
        e.currentTarget.classList.remove('drag-over');

        const droppedFiles = e.dataTransfer.files;
        if (!droppedFiles || droppedFiles.length === 0) return;
        
        setRecognizedText('');
        setError(null);
        setProgressMessage('');

        const newFiles = Array.from(droppedFiles);
        // FIX: Explicitly type 'f' as File to resolve an inference error where its type was 'unknown'.
        const supportedFiles = newFiles.filter((f: File) => 
            f.type.startsWith('image/') || f.type === 'application/pdf'
        );

        if (supportedFiles.length < newFiles.length) {
            setError("已过滤不支持的文件类型。请上传 PNG, JPG, 或 PDF 文件。");
        }

        setFiles(prev => [...prev, ...supportedFiles]);
        e.dataTransfer.clearData();
    };

    const pdfToImages = async (pdfFile: File): Promise<string[]> => {
        const images: string[] = [];
        const pdfJS = (window as any).pdfjsLib;
        if (!pdfJS) {
            throw new Error("PDF library is not loaded.");
        }
        pdfJS.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.5.136/pdf.worker.min.mjs`;

        const arrayBuffer = await pdfFile.arrayBuffer();
        const pdf = await pdfJS.getDocument(arrayBuffer).promise;
        const numPages = pdf.numPages;

        for (let i = 1; i <= numPages; i++) {
            setProgressMessage(`正在处理 PDF 页面 ${i} / ${numPages}...`);
            const page = await pdf.getPage(i);
            const viewport = page.getViewport({ scale: 2.0 }); // Higher scale for better OCR quality
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            canvas.height = viewport.height;
            canvas.width = viewport.width;

            if (context) {
                await page.render({ canvasContext: context, viewport: viewport }).promise;
                const blob = await new Promise<Blob | null>(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.95));
                if (blob) {
                    const base64 = await blobToBase64(blob);
                    images.push(base64);
                }
            }
        }
        return images;
    };


    const handleRecognize = async (ocrProvider: ModelProvider) => {
        if (files.length === 0) {
            setError("请先选择一个文件。");
            return;
        }
        setIsLoading(true);
        setRecognitionModel(ocrProvider);
        setError(null);
        setRecognizedText('');
        setProgressMessage('准备开始处理...');

        try {
            const imagesToProcess: { base64: string, mimeType: string }[] = [];
            
            let fileCounter = 0;
            for (const file of files) {
                fileCounter++;
                setProgressMessage(`正在处理文件 ${fileCounter} / ${files.length}: ${file.name}...`);
                if (file.type.startsWith('image/')) {
                    const base64Image = await blobToBase64(file);
                    imagesToProcess.push({ base64: base64Image, mimeType: file.type });
                } else if (file.type === 'application/pdf') {
                    const base64Images = await pdfToImages(file);
                    base64Images.forEach(b64 => {
                        imagesToProcess.push({ base64: b64, mimeType: 'image/jpeg' });
                    });
                }
            }

            if (imagesToProcess.length === 0) {
                throw new Error("没有可供识别的有效图片。");
            }
            
            setProgressMessage('正在调用 AI 模型进行识别...');
            const systemInstruction = `You are an expert Optical Character Recognition (OCR) engine. Your task is to extract any and all text from the provided image(s).
- Transcribe the text exactly as it appears.
- If multiple images are provided, treat them as pages of a single document and return the text in sequential order.
- Preserve the original line breaks and formatting as much as possible.
- Return only the extracted text, with no additional commentary, summaries, or explanations.`;
            const userPrompt = "Extract all text from the provided image(s), in order.";

            const responseText = await callGenerativeAi(
                ocrProvider,
                executionMode,
                systemInstruction,
                userPrompt,
                false,
                'ocr',
                [],
                imagesToProcess
            );

            setRecognizedText(responseText);

        } catch (err: any) {
            setError(`文本识别失败 (${ocrProvider}): ${err.message}`);
        } finally {
            setIsLoading(false);
            setRecognitionModel(null);
            setProgressMessage('');
        }
    };

    const handleSendChatMessage = async (e?: React.FormEvent) => {
        e?.preventDefault();
        if (!chatInput.trim() || isChatLoading) return;

        const newUserMessage: NoteChatMessage = { role: 'user', text: chatInput };
        const currentHistory = [...chatHistory, newUserMessage];
        setChatHistory(currentHistory);
        setChatInput('');
        setIsChatLoading(true);

        const systemInstruction = `You are a helpful assistant. The user has performed OCR on a document, and the recognized text is provided below. Your role is to answer questions, summarize, or analyze this text based on the user's request. Be helpful and conversational.\n\n--- RECOGNIZED TEXT ---\n${recognizedText}\n--- END TEXT ---`;

        const chatHistoryForApi = currentHistory
          .slice(0, -1)
          .filter(msg => !(msg.role === 'model' && msg.text.startsWith('您好！')))
          .map(msg => ({ role: msg.role as 'user' | 'model', parts: [{ text: msg.text }] }));

        const modelResponse: NoteChatMessage = { role: 'model', text: '' };
        setChatHistory(prev => [...prev, modelResponse]);

        try {
            await callGenerativeAiStream(
                chatProvider, executionMode, systemInstruction, chatInput, chatHistoryForApi,
                (chunk) => {
                    setChatHistory(prev => {
                        const newHistory = [...prev];
                        newHistory[newHistory.length - 1].text += chunk;
                        return newHistory;
                    });
                },
                () => { setIsChatLoading(false); },
                (error) => {
                    setChatHistory(prev => {
                        const newHistory = [...prev];
                        newHistory[newHistory.length - 1].isError = true;
                        newHistory[newHistory.length - 1].text = `抱歉，出错了: ${error.message}`;
                        return newHistory;
                    });
                    setIsChatLoading(false);
                }
            );
        } catch (error: any) {
            setChatHistory(prev => {
                const newHistory = [...prev];
                newHistory[newHistory.length - 1].isError = true;
                newHistory[newHistory.length - 1].text = `抱歉，出错了: ${error.message}`;
                return newHistory;
            });
            setIsChatLoading(false);
        }
    };

    return (
        <div className="ocr-view-container">
            <div className="file-upload-panel">
                <h2>1. 上传文件</h2>
                <div className="file-drop-zone" onClick={() => fileInputRef.current?.click()} onDragOver={handleDragOver} onDragLeave={handleDragLeave} onDrop={handleDrop}>
                    <input
                        type="file"
                        ref={fileInputRef}
                        style={{ display: 'none' }}
                        accept=".png,.jpg,.jpeg,.pdf"
                        onChange={handleFileChange}
                        multiple
                    />
                    {files.length > 0 ? (
                        <div className="file-list-preview">
                            <ul>
                                {files.map((f, index) => (
                                    <li key={`${f.name}-${index}`}>
                                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
                                            <path d="M14 2H6c-1.1 0-1.99.9-1.99 2L4 20c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V8l-6-6zM6 20V4h7v5h5v11H6z"></path>
                                        </svg>
                                        <span>{f.name}</span>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    ) : (
                        <p>点击或拖拽 PNG/JPG/PDF 文件到此处 (支持多文件)</p>
                    )}
                </div>
                 {files.length > 0 && <p className="instruction-text" style={{textAlign: 'center'}}>已选择 {files.length} 个文件</p>}
                <div className="utility-btn-group ocr-action-buttons">
                    <button className="btn btn-secondary" onClick={handleClear} disabled={(files.length === 0 && !recognizedText) || isLoading}>
                        清空内容
                    </button>
                    <div className="ocr-recognition-button-group">
                        <button
                            className="btn btn-primary"
                            onClick={() => handleRecognize('depOCR')}
                            disabled={files.length === 0 || isLoading}
                            title="使用专门优化的OCR模型进行识别"
                        >
                            {isLoading && recognitionModel === 'depOCR' ? 
                                <><span className="spinner"></span> {progressMessage || '识别中...'}</> : 
                                `2. 识别 (depOCR)`
                            }
                        </button>
                        {provider !== 'depOCR' && (
                            <button
                                className="btn btn-primary"
                                onClick={() => handleRecognize(provider)}
                                disabled={files.length === 0 || isLoading}
                                title={`使用全局选择的 ${provider} 模型进行多模态识别`}
                            >
                                {isLoading && recognitionModel === provider ? 
                                    <><span className="spinner"></span> {progressMessage || '识别中...'}</> :
                                    `2. 识别 (${provider})`
                                }
                            </button>
                        )}
                    </div>
                </div>
            </div>
            <div className="ocr-results-panel">
                <div className="view-header-row">
                    <h2>3. 识别结果与讨论</h2>
                    <div className="model-selector-container">
                        <span className="model-selector-label">对话模型:</span>
                        <div className="model-selector-group small">
                            {chatModelProviders.map(model => (
                                <button
                                    key={model}
                                    className={`model-btn ${chatProvider === model ? 'active' : ''}`}
                                    onClick={() => setChatProvider(model)}
                                    disabled={isChatLoading}
                                >
                                    {model}
                                </button>
                            ))}
                        </div>
                    </div>
                </div>
                <div className="ocr-result-and-chat-area">
                    <div className="ocr-result-container">
                        {isLoading && !recognizedText && (
                             <div className="spinner-container">
                                <div className="spinner large"></div>
                                <p>{progressMessage || '正在调用模型进行识别...'}</p>
                            </div>
                        )}
                        {!isLoading && !recognizedText && !error && (
                            <div className="large-placeholder">
                                <p>识别结果将显示在此处。</p>
                            </div>
                        )}
                        {error && <div className="error-message" style={{margin: '16px'}}>{error}</div>}
                        {recognizedText && <textarea className="text-area" value={recognizedText} readOnly />}
                    </div>
                    <div className="ocr-chat-container">
                        <div className="kb-chat-history" ref={chatHistoryRef}>
                            {chatHistory.map((msg, index) => (
                                <div key={index} className={`kb-message ${msg.role} ${msg.isError ? 'error' : ''}`}>
                                    <div className="message-content" style={{padding: '8px 12px', maxWidth: '100%'}}>
                                        <p>{msg.text}</p>
                                    </div>
                                </div>
                            ))}
                             {isChatLoading && chatHistory[chatHistory.length - 1]?.role === 'model' && !chatHistory[chatHistory.length - 1]?.text && (
                                <div className="spinner-container" style={{padding: '10px 0'}}><div className="spinner"></div></div>
                            )}
                        </div>
                        <form className="chat-input-form" onSubmit={handleSendChatMessage}>
                            <textarea
                                className="chat-input"
                                value={chatInput}
                                onChange={(e) => setChatInput(e.target.value)}
                                onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSendChatMessage(); }}}
                                placeholder="就识别出的文本提问..."
                                rows={1}
                                disabled={isChatLoading || !recognizedText}
                            />
                            <button type="submit" className="btn btn-primary send-btn" disabled={isChatLoading || !chatInput.trim() || !recognizedText}>发送</button>
                        </form>
                    </div>
                </div>
            </div>
        </div>
    );
};

const App = () => {
    type View = 'home' | 'notes' | 'audit' | 'knowledge-chat' | 'writing' | 'text-recognition';
    const [view, setView] = useState<View>('home');
    const [inputText, setInputText] = useState('');
    const [noteAnalysisResult, setNoteAnalysisResult] = useState<NoteAnalysis | null>(null);
    const [noteAnalysisError, setNoteAnalysisError] = useState<string | null>(null);
    const [isProcessing, setIsProcessing] = useState(false);
    const [isThoughtsModalOpen, setIsThoughtsModalOpen] = useState(false);
    
    const [selectedModel, setSelectedModel] = useState<ModelProvider>('gemini');
    const [executionMode, setExecutionMode] = useState<ExecutionMode>('backend');

    const [knowledgeBases, setKnowledgeBases] = useState<{ id: string; name: string }[]>([]);
    const [isKbLoading, setIsKbLoading] = useState(true);
    const [kbError, setKbError] = useState<string | null>(null);
    const [selectedKnowledgeBase, setSelectedKnowledgeBase] = useState<string | null>(null);
    const [initialKnowledgeChatQuestion, setInitialKnowledgeChatQuestion] = useState<string | undefined>();
    
    useEffect(() => {
        if (executionMode === 'frontend' && selectedModel !== 'gemini') {
            // If switching to frontend and a non-Gemini model was selected,
            // default back to Gemini to avoid issues before this refactor.
            // With the refactor, this line is less critical but good for safety.
            // setSelectedModel('gemini'); // This logic is now removed to allow other models
        }
    }, [executionMode, selectedModel]);

    useEffect(() => {
        const fetchKnowledgeBases = async () => {
            setIsKbLoading(true);
            setKbError(null);
            try {
                const response = await fetch(`${API_BASE_URL}/list-collections`);
                if (!response.ok) {
                    const errorText = await response.text().catch(() => response.statusText);
                    let errorJson;
                    try { errorJson = JSON.parse(errorText); } catch(e) {/* ignored */}
                    throw new Error(errorJson?.error || `获取知识库列表失败 (状态: ${response.status})`);
                }
                const data = await response.json();
                const collections: string[] = data.collections || [];
                const formattedKbs = collections.map(name => ({ id: name, name }));
                setKnowledgeBases(formattedKbs);
                // If nothing is selected, or the previously selected one no longer exists, select the first one.
                if (formattedKbs.length > 0) {
                    if (!selectedKnowledgeBase || !collections.includes(selectedKnowledgeBase)) {
                       setSelectedKnowledgeBase(formattedKbs[0].id);
                    }
                } else {
                    setSelectedKnowledgeBase(null);
                }
            } catch (error: any) {
                console.error("Failed to fetch knowledge bases:", error);
                const userFriendlyError = "无法连接到知识库服务。请检查后端服务是否正在运行，并刷新页面重试。";
                setKbError(userFriendlyError);
                setKnowledgeBases([]);
                setSelectedKnowledgeBase(null);
            } finally {
                setIsKbLoading(false);
            }
        };
        fetchKnowledgeBases();
    }, []); // Run only once on component mount


    const handleAnalysis = async (userThoughts: string) => {
        setIsProcessing(true);
        setNoteAnalysisError(null);
        setNoteAnalysisResult(null);
        setView('notes');

        const systemInstruction = `You are a note organization expert. Structure the user's fragmented notes into a coherent, organized document. Also, analyze and summarize the user's separate "thoughts" about the notes, maintaining them as a distinct section. Your response must be in JSON format with two keys: "organizedText" for the structured notes, and "userThoughts" for the processed user ideas.`;
        const userPrompt = `Here are my notes:\n\n${inputText}\n\nHere are my thoughts on these notes:\n\n${userThoughts}`;

        try {
            const responseText = await callGenerativeAi(selectedModel, executionMode, systemInstruction, userPrompt, true, 'notes');
            let result;
            try {
                result = JSON.parse(responseText);
            } catch (e: any) {
                console.error('Error parsing note analysis from AI:', responseText);
                throw new Error(`Failed to parse note analysis response: ${e.message}`);
            }
            setNoteAnalysisResult(result);
        } catch (err: any) {
            setNoteAnalysisError(`笔记整理失败: ${err.message}`);
        } finally {
            setIsProcessing(false);
        }
    };
    
    const handleTriggerOrganize = () => {
        setIsThoughtsModalOpen(true);
    };

    const handleTriggerAudit = () => {
        setView('audit');
    };
    
    const handleTriggerWriting = () => {
        setView('writing');
    };

    const handleTriggerTextRecognition = () => {
        setView('text-recognition');
    };

    const handleCloseThoughtsModal = () => {
        setIsThoughtsModalOpen(false);
    }
    
    const handleSubmitThoughts = (thoughts: string) => {
        setIsThoughtsModalOpen(false);
        handleAnalysis(thoughts);
    }

    const handleTriggerKnowledgeChat = () => {
        if (!selectedKnowledgeBase) {
            alert("请先选择一个知识库。");
            return;
        }
        if (!inputText.trim()) {
            alert("请在工作区输入您的问题。");
            return;
        }
        setInitialKnowledgeChatQuestion(inputText);
        setView('knowledge-chat');
    };

    const renderView = () => {
        switch (view) {
            case 'notes':
                return <NoteAnalysisView 
                    isLoading={isProcessing} 
                    error={noteAnalysisError} 
                    analysisResult={noteAnalysisResult} 
                    provider={selectedModel}
                    originalText={inputText}
                    selectedKnowledgeBaseId={selectedKnowledgeBase}
                    knowledgeBases={knowledgeBases}
                    executionMode={executionMode}
                />;
            case 'audit':
                 return <AuditView 
                    initialText={inputText} 
                    selectedModel={selectedModel}
                    executionMode={executionMode}
                 />;
            case 'knowledge-chat':
                if (!selectedKnowledgeBase) {
                    return <div className="error-message">错误：知识库未选择。请返回首页选择一个知识库。</div>;
                }
                return <KnowledgeChatView
                    knowledgeBaseId={selectedKnowledgeBase}
                    knowledgeBaseName={knowledgeBases.find(kb => kb.id === selectedKnowledgeBase)?.name || selectedKnowledgeBase}
                    initialQuestion={initialKnowledgeChatQuestion}
                    provider={selectedModel}
                    executionMode={executionMode}
                />;
            case 'writing':
                return <WritingView
                    initialText={inputText}
                    onTextChange={setInputText}
                    selectedModel={selectedModel}
                    selectedKnowledgeBase={selectedKnowledgeBase}
                    knowledgeBases={knowledgeBases}
                    executionMode={executionMode}
                />;
            case 'text-recognition':
                return <TextRecognitionView 
                    provider={selectedModel}
                    executionMode={executionMode}
                />;
            case 'home':
            default:
                return (
                    <HomeInputView 
                        inputText={inputText}
                        setInputText={setInputText}
                        onOrganize={handleTriggerOrganize}
                        onAudit={handleTriggerAudit}
                        selectedModel={selectedModel}
                        setSelectedModel={setSelectedModel}
                        isProcessing={isProcessing}
                        knowledgeBases={knowledgeBases}
                        isKbLoading={isKbLoading}
                        kbError={kbError}
                        selectedKnowledgeBase={selectedKnowledgeBase}
                        setSelectedKnowledgeBase={setSelectedKnowledgeBase}
                        onKnowledgeChat={handleTriggerKnowledgeChat}
                        onWriting={handleTriggerWriting}
                        onTextRecognition={handleTriggerTextRecognition}
                        executionMode={executionMode}
                        setExecutionMode={setExecutionMode}
                    />
                );
        }
    };

    const handleBackToHome = () => {
        setNoteAnalysisResult(null);
        setNoteAnalysisError(null);
        // Do not reset selectedKnowledgeBase, so it can be used again
        setInitialKnowledgeChatQuestion(undefined);
        setView('home');
    }

    return (
        <div className="main-layout">
            <div className="app-header">
                <h1>写作笔记助手</h1>
                 <div className="button-group">
                    {view !== 'home' && <button className="btn btn-secondary" onClick={handleBackToHome}>返回首页</button>}
                 </div>
            </div>

            <div className="view-container">
              {renderView()}
            </div>
            
            <ThoughtsInputModal
                isOpen={isThoughtsModalOpen}
                onClose={handleCloseThoughtsModal}
                onSubmit={handleSubmitThoughts}
             />
        </div>
    );
};

const root = ReactDOM.createRoot(document.getElementById('root')!);
root.render(<App />);