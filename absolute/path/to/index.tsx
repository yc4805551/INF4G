const handleStartRoaming = async () => {
  if (!selectedKnowledgeBaseId || !analysisResult) {
      if (!selectedKnowledgeBaseId) {
          alert("请返回首页选择一个知识库以开始笔记漫游。");
      }
      return;
  }

  setIsRoaming(true);
  setRoamingError(null);
  // 将原来的单一结果改为数组
  const [roamingResult, setRoamingResult] = useState<RoamingResult[] | null>(null);

  try {
      // Step 1: Call local backend to get relevant context - 修改top_k为3
      const backendResponse = await fetch(`${API_BASE_URL}/find-related`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
              text: analysisResult.organizedText,
              collection_name: selectedKnowledgeBaseId,
              top_k: 3 // 修改为获取3个相关文档
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
      
      // 修改：获取所有相关文档，而不只是第一个
      const relevantSources = backendData.related_documents || [];

      if (relevantSources.length === 0) {
          setRoamingError("在知识库中未找到足够相关的内容来进行漫游联想。");
          setIsRoaming(false);
          return;
      }

      // 修改：使用所有获取到的相关文档
      // 对于每个相关文档，创建一个独立的联想结论
      const roamingResults: RoamingResult[] = [];
      for (const source of relevantSources) {
          const relevantContext = source.content_chunk;
          const sourceFileName = source.source_file;

          // 为每个文档生成独立的联想结论
          const systemInstruction = `You are an AI assistant skilled at synthesizing information. Based on a user's note and a relevant passage from their knowledge base, create an "Associative Conclusion" connecting the two ideas. Your entire response must be a JSON object with one key: "conclusion" (your generated associative summary).`;
          const userPrompt = `[Relevant Passage from Knowledge Base]:\n${relevantContext}\n\n[User's Original Note]:\n${analysisResult.organizedText}`;
          
          const genAiResponseText = await callGenerativeAi(provider, systemInstruction, userPrompt, true, apiKeys, 'roaming');
          const result = JSON.parse(genAiResponseText.replace(/```json\n?|\n?```/g, ''));

          if (result.conclusion) {
              roamingResults.push({
                  source: sourceFileName,
                  relevantText: relevantContext,
                  conclusion: result.conclusion,
              });
          }
      }

      // 修改：存储多个漫游结果
      setRoamingResult(roamingResults);

  } catch (err: any) {
      setRoamingError(`笔记漫游失败: ${err.message}`);
  } finally {
      setIsRoaming(false);
  }
};

interface RoamingResult {
  source: string;
  relevantText: string;
  conclusion: string;
}

<div className="content-section" style={{padding: '16px', backgroundColor: 'var(--background-color)'}}>
    <h3>笔记漫游</h3>
    {!roamingResult && !isRoaming && !roamingError && <p className="instruction-text">如需基于笔记内容进行关联联想，请在首页选择知识库后，点击下方“开始笔记漫游”按钮。</p>}
    {isRoaming && <div className="spinner-container" style={{padding: '20px 0'}}><div className="spinner"></div></div>}
    {roamingError && <div className="error-message">{roamingError}</div>}
    {roamingResult && roamingResult.length > 0 && (
        <div className="roaming-results-container">
            {roamingResult.map((result, index) => (
                <div key={index} className="roaming-result">
                    <div className="roaming-result-header">
                        <h4>关联笔记 #{index + 1}</h4>
                        <span className="roaming-source">{result.source}</span>
                    </div>
                    <p><strong>关联原文:</strong> {result.relevantText}</p>
                    <p><strong>联想结论:</strong> {result.conclusion}</p>
                </div>
            ))}
        </div>
    )}
</div>

const handleExportTXT = () => {
  if (!analysisResult) return;

  // Part 1: Main Content
  let content = `【笔记工作台】\n\n【整理后】\n${analysisResult.organizedText}\n\n---\n\n【我的想法】\n${analysisResult.userThoughts}`;
  
  // Part 2: Roaming Result - 修改为支持多个漫游结果
  if (roamingResult && Array.isArray(roamingResult) && roamingResult.length > 0) {
    content += `\n\n---\n\n【笔记漫游】`;
    roamingResult.forEach((result, index) => {
      content += `\n\n## 关联笔记 ${index + 1}\n来源: ${result.source}\n\n关联原文:\n${result.relevantText}\n\n联想结论:\n${result.conclusion}`;
    });
  } else if (roamingResult && !Array.isArray(roamingResult)) {
    // 向后兼容：处理单一结果的情况
    content += `\n\n---\n\n【笔记漫游】\n来源: ${roamingResult.source}\n\n关联原文:\n${roamingResult.relevantText}\n\n联想结论:\n${roamingResult.conclusion}`;
  }
  
  // 其他部分保持不变
  // ... existing code ...
};