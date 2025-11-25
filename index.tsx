import React, { useState, useEffect, useRef, useCallback } from 'react';
import ReactDOM from 'react-dom/client';
import mammoth from 'mammoth';
import { marked } from 'marked';
import { GoogleGenAI } from '@google/genai';

// Helper to clean environment variables (remove accidentally added quotes/smart-quotes)
const cleanEnv = (value: string | undefined): string | undefined => {
    if (!value) return undefined;
    // Remove leading/trailing whitespace and quotes (straight or smart)
    return value.trim().replace(/^["'“]+|["'”]+$/g, '');
};

const API_BASE_URL = import.meta.env?.PROD
  ? `${cleanEnv(import.meta.env.VITE_API_BASE_URL) || ''}/api`
  : '/proxy-api';//doujunhao- 设置了后端连接

// Helper to generate user-friendly error messages for backend issues
const getBackendErrorMessage = (error: any, url: string): string => {
    // Check for Mixed Content (HTTPS frontend -> HTTP backend)
    if (typeof window !== 'undefined' && window.location.protocol === 'https:' && url.startsWith('http:')) {
        return `【安全限制警告】\n\nGitHub Pages (HTTPS) 无法连接到不安全的 HTTP 后端 (${url})。\n浏览器出于安全原因拦截了请求 (Mixed Content)。\n\n解决方案：\n1. 请将您的后端服务升级为 HTTPS。\n2. 或者切换到“前端直连”模式使用基础 AI 功能。\n3. 或者在本地环境运行前端。`;
    }
    if (error instanceof TypeError && error.message.toLowerCase().includes('failed to fetch')) {
        return `网络请求失败。无法连接到后端服务 (${url})。\n\n可能原因：\n1. 后端服务未启动。\n2. 跨域 (CORS) 配置未允许当前域名。\n3. 网络连接问题。`;
    }
    return error.message || `后端请求出错 (${url})`;
};

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
        apiKey: cleanEnv(import.meta.env?.VITE_GEMINI_API_KEY),
        model: 'gemini-2.5-flash',
    },
    openai: {
        apiKey: cleanEnv(import.meta.env?.VITE_OPENAI_API_KEY),
        // Prioritize VITE_OPENAI_ENDPOINT if available, matching usage in other providers
        endpoint: cleanEnv(import.meta.env?.VITE_OPENAI_ENDPOINT) || (cleanEnv(import.meta.env?.VITE_OPENAI_TARGET_URL) ? `${cleanEnv(import.meta.env.VITE_OPENAI_TARGET_URL)}/v1/chat/completions` : undefined),
        model: cleanEnv(import.meta.env?.VITE_OPENAI_MODEL),
    },
    deepseek: {
        apiKey: cleanEnv(import.meta.env?.VITE_DEEPSEEK_API_KEY),
        endpoint: cleanEnv(import.meta.env?.VITE_DEEPSEEK_ENDPOINT),
        model: cleanEnv(import.meta.env?.VITE_DEEPSEEK_MODEL),
    },
    ali: {
        apiKey: cleanEnv(import.meta.env?.VITE_ALI_API_KEY),
        endpoint: cleanEnv(import.meta.env?.VITE_ALI_ENDPOINT) || (cleanEnv(import.meta.env?.VITE_ALI_TARGET_URL) ? `${cleanEnv(import.meta.env.VITE_ALI_TARGET_URL)}/v1/chat/completions` : undefined),
        model: cleanEnv(import.meta.env?.VITE_ALI_MODEL),
    },
    depOCR: {
        apiKey: cleanEnv(import.meta.env?.VITE_DEPOCR_API_KEY),
        endpoint: cleanEnv(import.meta.env?.VITE_DEPOCR_ENDPOINT),
        model: cleanEnv(import.meta.env?.VITE_DEPOCR_MODEL),
    },
    doubao: {
        apiKey: cleanEnv(import.meta.env?.VITE_DOUBAO_API_KEY),
        endpoint: cleanEnv(import.meta.env?.VITE_DOUBAO_ENDPOINT),
        model: cleanEnv(import.meta.env?.VITE_DOUBAO_MODEL),
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
                    throw new Error(getBackendErrorMessage(error, API_BASE_URL));
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
        onError(new Error(getBackendErrorMessage(error, API_BASE_URL)));
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
                    {kbError && <div className="error-message" style={{textAlign: 'left', whiteSpace: 'pre-wrap'}}>{kbError}</div>}
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
        // Use setTimeout to ensure DOM has updated before scrolling
        const timer = setTimeout(() => {
             if (chatHistoryRef.current) {
                chatHistoryRef.current.scrollTop = chatHistoryRef.current.scrollHeight;
             }
        }, 100);
        return () => clearTimeout(timer);
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
        setRoamingError(getBackendErrorMessage(err, API_BASE_URL));
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
  // 5. 检查是否为单个问题对象 (Handling the specific request where model returns a single object instead of array)
  else if (typeof data === 'object' && data !== null && 'problematicText' in data && 'suggestion' in data) {
    issuesArray = [data as AuditIssue];
  }
  // 6. 如果都不是，说明格式错误 
  else { 
    // 我们现在将原始的 'responseText' 作为 rawResponse 传递回去，以便调试 
    return { issues: [], error: "模型返回了意外的 JSON 格式 (既不是数组，也不是包含 'issues' 的对象，也不是单个问题对象)。", rawResponse: responseText }; 
  }

  // 7. 现在我们安全地筛选 issuesArray 
  const validIssues = issuesArray
    .map((issue: any) => {
        if (!issue || typeof issue !== 'object') return null;
        // Even if problematicText is missing, we try to salvage what we can or skip
        if (!issue.problematicText || typeof issue.problematicText !== 'string') {
            // Some models might return just a suggestion or general advice in an object
            // If it has at least a suggestion or explanation, we can show it with a placeholder
             if ((issue.suggestion && typeof issue.suggestion === 'string') || (issue.explanation && typeof issue.explanation === 'string')) {
                 return {
                    problematicText: issue.problematicText || '(未指定文本)',
                    suggestion: typeof issue.suggestion === 'string' ? issue.suggestion : '',
                    checklistItem: typeof issue.checklistItem === 'string' ? issue.checklistItem : '通用规则',
                    explanation: typeof issue.explanation === 'string' ? issue.explanation : (typeof issue.reason === 'string' ? issue.reason : '无详细说明')
                 } as AuditIssue;
             }
             return null;
        }
        
        return {
            problematicText: issue.problematicText,
            suggestion: typeof issue.suggestion === 'string' ? issue.suggestion : '',
            checklistItem: typeof issue.checklistItem === 'string' ? issue.checklistItem : '通用规则',
            explanation: typeof issue.explanation === 'string' ? issue.explanation : (typeof issue.reason === 'string' ? issue.reason : '无详细说明')
        } as AuditIssue;
    })
    .filter((issue): issue is AuditIssue => issue !== null);
    
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