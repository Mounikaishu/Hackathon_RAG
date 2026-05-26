import React, { useState, useEffect, useRef } from 'react';

// Custom lightweight Markdown parser for formatting text, bold styling, lists, and tables
const MarkdownRenderer = ({ text }) => {
  if (!text) return null;

  // Clean up bold tags first
  let html = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

  // Detect and parse Markdown tables
  const lines = html.split('\n');
  let inTable = false;
  let tableRows = [];
  let renderedElements = [];
  let currentList = [];
  let inList = false;

  const flushList = (key) => {
    if (currentList.length > 0) {
      renderedElements.push(
        <ul key={`list-${key}`}>
          {currentList.map((li, idx) => (
            <li key={idx} dangerouslySetInnerHTML={{ __html: li }} />
          ))}
        </ul>
      );
      currentList = [];
      inList = false;
    }
  };

  const flushTable = (key) => {
    if (tableRows.length > 0) {
      renderedElements.push(
        <div className="table-responsive" key={`table-wrapper-${key}`}>
          <table>
            <thead>
              <tr>
                {tableRows[0].map((cell, idx) => (
                  <th key={idx} dangerouslySetInnerHTML={{ __html: cell }} />
                ))}
              </tr>
            </thead>
            <tbody>
              {tableRows.slice(1).map((row, rIdx) => (
                <tr key={rIdx}>
                  {row.map((cell, cIdx) => (
                    <td key={cIdx} dangerouslySetInnerHTML={{ __html: cell }} />
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
      tableRows = [];
      inTable = false;
    }
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();

    // Markdown Table Detection
    if (line.startsWith('|') || (line.includes('|') && line.split('|').length > 2)) {
      flushList(i);

      // Skip table divider rows: |---|---|
      if (line.includes('---') || line.includes('-|-')) {
        continue;
      }

      inTable = true;
      const cells = line
        .split('|')
        .map((c) => c.trim())
        .filter((c, idx, arr) => idx > 0 && idx < arr.length - 1);
      
      if (cells.length > 0) {
        tableRows.push(cells);
      }
    } 
    // Bullet List Detection
    else if (line.startsWith('* ') || line.startsWith('- ')) {
      flushTable(i);
      inList = true;
      currentList.push(line.substring(2));
    } 
    // Standard Paragraph / Text
    else {
      flushTable(i);
      flushList(i);

      if (line === '') {
        renderedElements.push(<br key={`br-${i}`} />);
      } else {
        renderedElements.push(
          <p key={`p-${i}`} dangerouslySetInnerHTML={{ __html: line }} style={{ marginBottom: '0.5rem' }} />
        );
      }
    }
  }

  // Final flushes
  flushTable('final');
  flushList('final');

  return <>{renderedElements}</>;
};

function App() {
  // Global States
  const [messages, setMessages] = useState([
    {
      sender: 'bot',
      agent: 'system_initialization',
      reason: 'User session initialized.',
      text: 'Welcome to the SVECW Placement Intelligence Assistant. Ask me anything about company eligibility profiles, interview experiences, temporal hiring trends, or distribution charts.'
    }
  ]);
  const [query, setQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [isIndexing, setIsIndexing] = useState(false);

  // Database Stats
  const [stats, setStats] = useState({
    companiesCount: 0,
    vectorsCount: 0,
    chartsCount: 0
  });

  // Upload States
  const [uploadStatus, setUploadStatus] = useState({
    pdf: { status: 'idle', message: '' },
    table: { status: 'idle', message: '' },
    image: { status: 'idle', message: '' }
  });

  // Vision Gallery States
  const companiesList = ['Amazon', 'Google', 'Infosys', 'Microsoft', 'TCS'];
  const [selectedChart, setSelectedChart] = useState('TCS');
  const [highlightGallery, setHighlightGallery] = useState(false);
  const [lightboxImg, setLightboxImg] = useState(null);

  const chatMessagesEndRef = useRef(null);

  // Poll status counters and FastAPI connectivity
  const fetchStatus = async () => {
    try {
      const response = await fetch('/api/status');
      if (response.ok) {
        const data = await response.json();
        setStats({
          companiesCount: data.structured_database.companies_count,
          vectorsCount: data.vector_database.chunks_count,
          chartsCount: data.charts_gallery.charts_count
        });
        setIsConnected(true);
      } else {
        setIsConnected(false);
      }
    } catch (err) {
      setIsConnected(false);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  // Scroll chat to bottom on new messages
  useEffect(() => {
    chatMessagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  // Handle re-indexing POST request
  const handleIndexDatabase = async () => {
    if (isIndexing) return;
    setIsIndexing(true);
    try {
      const response = await fetch('/api/index', { method: 'POST' });
      if (response.ok) {
        alert('Indexing pipeline started in the background. It will take a few moments to ingest the PDF data.');
      } else {
        alert('Failed to trigger indexing. Verify that your backend is running.');
        setIsIndexing(false);
      }
    } catch (err) {
      alert('Error triggering index: ' + err.message);
      setIsIndexing(false);
    }

    // Stop index loader after 15 seconds (assumes indexing starts successfully)
    setTimeout(() => {
      setIsIndexing(false);
      fetchStatus();
    }, 15000);
  };

  // Helper file uploader function
  const uploadFile = async (file, type) => {
    if (!file) return;
    
    // Update progress state
    setUploadStatus(prev => ({
      ...prev,
      [type]: { status: 'loading', message: `Uploading ${file.name}...` }
    }));

    const formData = new FormData();
    formData.append('file', file);

    let endpoint = '/api/upload/pdf';
    if (type === 'table') endpoint = '/api/upload/table';
    if (type === 'image') endpoint = '/api/upload/image';

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData
      });

      const data = await response.json();

      if (response.ok) {
        setUploadStatus(prev => ({
          ...prev,
          [type]: { status: 'success', message: `✓ ${file.name} uploaded successfully!` }
        }));
        fetchStatus();
        setTimeout(() => {
          setUploadStatus(prev => ({
            ...prev,
            [type]: { status: 'idle', message: '' }
          }));
        }, 5000);
      } else {
        throw new Error(data.detail || 'Upload failed');
      }
    } catch (err) {
      setUploadStatus(prev => ({
        ...prev,
        [type]: { status: 'error', message: `Error: ${err.message}` }
      }));
    }
  };

  // Handle Query Submission
  const handleSubmitQuery = async (queryText) => {
    const activeQuery = queryText || query;
    if (!activeQuery.trim()) return;

    if (!queryText) setQuery('');

    // 1. Add user message
    setMessages(prev => [...prev, { sender: 'user', text: activeQuery }]);
    setIsLoading(true);

    try {
      const response = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: activeQuery })
      });

      if (!response.ok) {
        throw new Error('Server connection error');
      }

      const data = await response.json();

      // Check if query is routed to Vision Agent or company referenced
      if (data.routed_agent === 'vision_agent') {
        setHighlightGallery(true);
        // Deduce company name from response or query
        const queryLower = activeQuery.toLowerCase();
        const matchedCompany = companiesList.find(c => queryLower.includes(c.toLowerCase()));
        if (matchedCompany) {
          setSelectedChart(matchedCompany);
        }
        setTimeout(() => setHighlightGallery(false), 3000);
      }

      // Add bot message
      setMessages(prev => [
        ...prev,
        {
          sender: 'bot',
          agent: data.routed_agent,
          reason: data.routing_reason,
          text: data.response
        }
      ]);

    } catch (err) {
      setMessages(prev => [
        ...prev,
        {
          sender: 'bot',
          agent: 'error_agent',
          reason: 'API pipeline call failed.',
          text: '❌ Failed to complete request. Please verify your FastAPI server connection.'
        }
      ]);
    } finally {
      setIsLoading(false);
      fetchStatus();
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      
      {/* Header Panel */}
      <header className="app-header">
        <div className="logo-section">
          <div className="logo-icon">🎓</div>
          <h1>SVECW Placement Intelligence Dashboard</h1>
        </div>
        <div className="header-actions">
          <div className={`status-badge ${!isConnected ? 'error' : ''}`}>
            <div className={`status-dot ${!isConnected ? 'error' : ''}`}></div>
            <span>{isConnected ? 'FastAPI Server Active' : 'Server Offline'}</span>
          </div>
        </div>
      </header>

      {/* Main Workspace Grid */}
      <div className="workspace">
        
        {/* Left Sidebar */}
        <aside className="sidebar">
          <div>
            <div className="section-title">
              <span>📊</span> RAG System Status
            </div>
            <div className="stat-grid">
              <div className="stat-card">
                <span className="stat-val">{isConnected ? stats.companiesCount : '-'}</span>
                <span className="stat-label">Structured Companies</span>
              </div>
              <div className="stat-card">
                <span className="stat-val">{isConnected ? stats.vectorsCount : '-'}</span>
                <span className="stat-label">Text Vector Chunks</span>
              </div>
              <div className="stat-card">
                <span className="stat-val">{isConnected ? stats.chartsCount : '-'}</span>
                <span className="stat-label">Extracted Visual Charts</span>
              </div>
            </div>
          </div>

          <div>
            <div className="section-title">
              <span>⚙️</span> Ingestion Actions
            </div>
            <button 
              className="index-db-btn"
              onClick={handleIndexDatabase} 
              disabled={isIndexing || !isConnected}
            >
              {isIndexing ? '⚙️ Reindexing DB...' : '🔄 Run Indexing Pipeline'}
            </button>
          </div>

          <div>
            <div className="section-title">
              <span>📤</span> Ingest Database Files
            </div>
            <div className="upload-group">
              
              {/* PDF Dataset Upload */}
              <div className="upload-card">
                <input 
                  type="file" 
                  accept=".pdf" 
                  onChange={(e) => uploadFile(e.target.files[0], 'pdf')} 
                  disabled={!isConnected}
                />
                <div className="upload-label">
                  <span className="upload-icon">📄</span>
                  <span className="upload-text">Upload Dataset PDF</span>
                  <span className="upload-sub">PDF formatting</span>
                  {uploadStatus.pdf.message && (
                    <span className={`upload-progress-text ${uploadStatus.pdf.status}`}>{uploadStatus.pdf.message}</span>
                  )}
                </div>
              </div>

              {/* JSON Eligibility Table Upload */}
              <div className="upload-card">
                <input 
                  type="file" 
                  accept=".json" 
                  onChange={(e) => uploadFile(e.target.files[0], 'table')} 
                  disabled={!isConnected}
                />
                <div className="upload-label">
                  <span className="upload-icon">📋</span>
                  <span className="upload-text">Upload Eligibility JSON</span>
                  <span className="upload-sub">JSON company table</span>
                  {uploadStatus.table.message && (
                    <span className={`upload-progress-text ${uploadStatus.table.status}`}>{uploadStatus.table.message}</span>
                  )}
                </div>
              </div>

              {/* Chart Image Upload */}
              <div className="upload-card">
                <input 
                  type="file" 
                  accept=".png,.jpg,.jpeg" 
                  onChange={(e) => uploadFile(e.target.files[0], 'image')} 
                  disabled={!isConnected}
                />
                <div className="upload-label">
                  <span className="upload-icon">🖼️</span>
                  <span className="upload-text">Upload Chart Image</span>
                  <span className="upload-sub">PNG, JPG, JPEG</span>
                  {uploadStatus.image.message && (
                    <span className={`upload-progress-text ${uploadStatus.image.status}`}>{uploadStatus.image.message}</span>
                  )}
                </div>
              </div>

            </div>
          </div>

          <div>
            <div className="section-title">
              <span>💡</span> Quick Queries
            </div>
            <div className="quick-prompts-list">
              <button className="quick-btn" onClick={() => handleSubmitQuery("What is Amazon's CGPA cutoff?")}>Amazon CGPA Conflict</button>
              <button className="quick-btn" onClick={() => handleSubmitQuery("Which company showed the highest increase from 2021 to 2024?")}>Temporal Growth</button>
              <button className="quick-btn" onClick={() => handleSubmitQuery("A student with CGPA 7.6 and 1 backlog wants the highest-paying job they qualify for")}>Multi-Hop Scenario</button>
              <button className="quick-btn" onClick={() => handleSubmitQuery("What rounds does TCS conduct?")}>TCS Interview Rounds</button>
              <button className="quick-btn" onClick={() => handleSubmitQuery("Which company hires the most Analysts?")}>Analyst Hiring Chart</button>
            </div>
          </div>
        </aside>

        {/* Central Chat Interface */}
        <main className="chat-container">
          <div className="chat-messages">
            
            {/* RAG Welcome Screen */}
            {messages.length === 1 && (
              <div className="welcome-card">
                <h2>SVECW Placement Intelligence Engine</h2>
                <p>
                  Welcome to the ultimate agentic multimodal placement information assistant. Powered by 
                  Llama 3.3 Reasoning, Llama 3.2 Vision, ChromaDB, and Pandas Analysis.
                </p>
                <div className="agent-grid">
                  <div className="agent-chip" style={{ borderLeft: '3px solid var(--accent-amber)' }}>Pandas Dataframe</div>
                  <div className="agent-chip" style={{ borderLeft: '3px solid var(--accent-cyan)' }}>Vision Analysis</div>
                  <div className="agent-chip" style={{ borderLeft: '3px solid var(--accent-rose)' }}>Conflict Checker</div>
                  <div className="agent-chip" style={{ borderLeft: '3px solid var(--accent-emerald)' }}>Semantic Vector RAG</div>
                  <div className="agent-chip" style={{ borderLeft: '3px solid var(--accent-purple)' }}>Multi-hop Reasoning</div>
                  <div className="agent-chip" style={{ borderLeft: '3px solid var(--accent-blue)' }}>Tavily Web Search</div>
                </div>
              </div>
            )}

            {/* Messages Stream */}
            {messages.map((msg, idx) => (
              <div key={idx} className={`message ${msg.sender}`}>
                {msg.sender === 'bot' && msg.agent && (
                  <span className={`agent-badge badge-${msg.agent}`}>
                    ⚡ {msg.agent.replace('_', ' ')}
                  </span>
                )}
                <div className="message-bubble">
                  <MarkdownRenderer text={msg.text} />
                  
                  {msg.sender === 'bot' && msg.reason && (
                    <div className="routing-trace">
                      <strong>Router Trace:</strong> {msg.reason}
                    </div>
                  )}
                </div>
              </div>
            ))}

            {/* Routing Loader animation */}
            {isLoading && (
              <div className="message bot">
                <span className="agent-badge badge-rag_agent">Routing Query...</span>
                <div className="message-bubble">
                  <div className="routing-loader">
                    <div className="loader-dots">
                      <span></span><span></span><span></span>
                    </div>
                  </div>
                </div>
              </div>
            )}
            
            <div ref={chatMessagesEndRef} />
          </div>

          {/* Chat Input */}
          <div className="chat-input-bar">
            <input 
              type="text" 
              className="chat-input"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask a placement intelligence question..." 
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !isLoading) {
                  handleSubmitQuery();
                }
              }}
              disabled={isLoading || !isConnected}
            />
            <button 
              className="send-btn"
              onClick={() => handleSubmitQuery()}
              disabled={isLoading || !isConnected}
            >
              <span>Send</span>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <line x1="22" y1="2" x2="11" y2="13"></line>
                <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
              </svg>
            </button>
          </div>
        </main>

        {/* Right Gallery Panel */}
        <aside className="gallery-panel">
          <div className="section-title">
            <span>🖼️</span> Visual Hiring Assets
          </div>
          
          <div className={`chart-viewer ${highlightGallery ? 'highlighted' : ''}`}>
            <img 
              src={`/static/charts/${selectedChart}.png`} 
              className="chart-img" 
              alt={`${selectedChart} Hiring Distribution`}
              onClick={() => setLightboxImg(`/static/charts/${selectedChart}.png`)}
              onError={(e) => {
                e.target.style.display = 'none';
                const sibling = e.target.nextSibling;
                if (sibling) sibling.style.display = 'flex';
              }}
              onLoad={(e) => {
                e.target.style.display = 'block';
                const sibling = e.target.nextSibling;
                if (sibling) sibling.style.display = 'none';
              }}
            />
            <div className="chart-image-placeholder" style={{ display: 'none' }}>
              <span>🖼️</span>
              <span>Chart image not loaded. Try re-indexing databases or query the agent.</span>
            </div>
            <div className="stat-label" style={{ fontWeight: '600', marginTop: '0.4rem' }}>
              {selectedChart} - Extracted Hiring Distribution
            </div>
          </div>

          <div>
            <div className="section-title" style={{ marginBottom: '0.5rem' }}>
              📁 Vision Chart Library
            </div>
            <div className="chart-thumbnail-grid">
              {companiesList.map((company) => (
                <div 
                  key={company} 
                  className={`chart-thumb ${selectedChart === company ? 'active' : ''}`}
                  onClick={() => setSelectedChart(company)}
                >
                  <img 
                    src={`/static/charts/${company}.png`} 
                    alt={company} 
                    onError={(e) => {
                      e.target.src = 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="100" height="60" viewBox="0 0 100 60"><rect width="100%" height="100%" fill="%231e293b"/><text x="50%" y="50%" fill="%2394a3b8" dominant-baseline="middle" text-anchor="middle" font-size="10">Missing</text></svg>';
                    }}
                  />
                  <span>{company}</span>
                </div>
              ))}
            </div>
          </div>
        </aside>
      </div>

      {/* Lightbox Modal */}
      {lightboxImg && (
        <div className="lightbox-modal" onClick={() => setLightboxImg(null)}>
          <div className="lightbox-content" onClick={(e) => e.stopPropagation()}>
            <button className="close-lightbox" onClick={() => setLightboxImg(null)}>✕ Close</button>
            <img src={lightboxImg} alt="Enlarged Chart" />
            <div className="lightbox-caption">
              {selectedChart} Extracted Placement Distribution (Llama Vision Asset)
            </div>
          </div>
        </div>
      )}

    </div>
  );
}

export default App;
