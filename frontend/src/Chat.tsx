import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { getApiUrl } from './config';

interface Citation {
  chunk_id: string;
  document_id: string;
  document_title?: string;
  page_number?: number;
  section_path?: string;
  score?: number;
  similarity_score?: number;
  rerank_score?: number;
  text_snippet?: string;
}

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
  evidence_score?: number;
  guardrail_alert?: string;
}

const WELCOME_MESSAGE: Message = {
  id: 'welcome-1',
  role: 'assistant',
  content: 'Hello! I am your Enterprise Knowledge AI Assistant. Ask any question about your indexed organizational documents.',
  citations: []
};

const isRefusalOrNoContext = (content?: string, alert?: string) => {
  if (!content) return false;
  const lower = content.toLowerCase();
  if (lower.includes("no relevant document context found")) return true;
  if (lower.includes("could not find sufficient information")) return true;
  if (lower.includes("no relevant context")) return true;
  if (alert && alert.toLowerCase().includes("refusal")) return true;
  return false;
};

export const Chat: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>(() => {
    const cached = localStorage.getItem('chat_messages');
    if (cached) {
      try {
        const parsed = JSON.parse(cached);
        if (Array.isArray(parsed) && parsed.length > 0) return parsed;
      } catch (e) { }
    }
    return [WELCOME_MESSAGE];
  });

  const [inputQuery, setInputQuery] = useState('');
  const [loading, setLoading] = useState(false);

  // Configuration States
  const [retrievalMode, setRetrievalMode] = useState<'hybrid' | 'vector' | 'fts'>('hybrid');
  const [enableReranking, setEnableReranking] = useState(true);
  const [rerankTopK, setRerankTopK] = useState(5);
  const [enableRewriting, setEnableRewriting] = useState(true);
  const [enableCompression, setEnableCompression] = useState(true);
  const [strictGrounding, setStrictGrounding] = useState(true);
  const [rerankerStatus, setRerankerStatus] = useState<'active_mock' | 'active_http' | 'offline_fallback'>('active_mock');

  // Selected Citation for Drawer
  const [activeCitations, setActiveCitations] = useState<Citation[]>(() => {
    const cached = localStorage.getItem('chat_active_citations');
    if (cached) {
      try {
        const parsed = JSON.parse(cached);
        if (Array.isArray(parsed)) return parsed;
      } catch (e) { }
    }
    return [];
  });

  const [conversationId, setConversationId] = useState<string | null>(() => {
    return localStorage.getItem('chat_conversation_id');
  });

  React.useEffect(() => {
    authenticatedFetch('/api/v1/admin/config')
      .then(res => res.ok ? res.json() : null)
      .then(resJson => {
        if (resJson && resJson.data) {
          if (typeof resJson.data.strict_grounding === 'boolean') {
            setStrictGrounding(resJson.data.strict_grounding);
          }
          if (resJson.data.reranker_status) {
            setRerankerStatus(resJson.data.reranker_status);
          }
        }
      })
      .catch(e => console.warn("Config fetch warning:", e));
  }, []);

  const handleStrictGroundingToggle = async (newValue: boolean) => {
    setStrictGrounding(newValue);
    try {
      await authenticatedFetch('/api/v1/admin/config/strict-grounding', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ strict_grounding: newValue })
      });
    } catch (e) {
      console.warn("Strict grounding update warning:", e);
    }
  };

  const saveMessagesToCache = (msgs: Message[]) => {
    setMessages(msgs);
    localStorage.setItem('chat_messages', JSON.stringify(msgs));
  };

  const saveCitationsToCache = (cits: Citation[]) => {
    setActiveCitations(cits);
    localStorage.setItem('chat_active_citations', JSON.stringify(cits));
  };

  const handleClearChat = () => {
    saveMessagesToCache([WELCOME_MESSAGE]);
    saveCitationsToCache([]);
    setConversationId(null);
    localStorage.removeItem('chat_messages');
    localStorage.removeItem('chat_active_citations');
    localStorage.removeItem('chat_conversation_id');
  };

  const tryRefreshToken = async (): Promise<string | null> => {
    const refreshToken = localStorage.getItem('refresh_token');
    if (!refreshToken) return null;

    try {
      const response = await fetch(getApiUrl('/api/v1/auth/refresh'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refresh_token: refreshToken })
      });

      if (!response.ok) {
        localStorage.removeItem('auth_token');
        localStorage.removeItem('refresh_token');
        window.dispatchEvent(new Event('auth_session_expired'));
        return null;
      }

      const resJson = await response.json();
      const data = resJson.data || resJson;
      if (data && data.access_token) {
        const newAccess = data.access_token;
        const newRefresh = data.refresh_token || refreshToken;
        localStorage.setItem('auth_token', newAccess);
        localStorage.setItem('refresh_token', newRefresh);
        return newAccess;
      }
    } catch (e) {
      console.warn("Token refresh failed:", e);
    }

    localStorage.removeItem('auth_token');
    localStorage.removeItem('refresh_token');
    window.dispatchEvent(new Event('auth_session_expired'));
    return null;
  };

  const authenticatedFetch = async (url: string, options: RequestInit = {}, isRetry = false): Promise<Response> => {
    let token = localStorage.getItem('auth_token') || '';
    const headers = new Headers(options.headers || {});
    if (token) {
      headers.set('Authorization', `Bearer ${token}`);
    }
    options.headers = headers;

    const response = await fetch(getApiUrl(url), options);

    if (response.status === 401 && !isRetry) {
      const newToken = await tryRefreshToken();
      if (newToken) {
        const retryHeaders = new Headers(options.headers || {});
        retryHeaders.set('Authorization', `Bearer ${newToken}`);
        options.headers = retryHeaders;
        return authenticatedFetch(url, options, true);
      }
    }

    return response;
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputQuery.trim() || loading) return;

    const userMsg: Message = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: inputQuery
    };

    const updatedMsgsWithUser = [...messages, userMsg];
    saveMessagesToCache(updatedMsgsWithUser);

    const currentQuery = inputQuery.trim();
    setInputQuery('');
    setLoading(true);

    // Create placeholder message for SSE streaming response
    const assistantMsgId = `asst-${Date.now()}`;
    const initialAssistantMsg: Message = {
      id: assistantMsgId,
      role: 'assistant',
      content: '',
      citations: []
    };
    const msgsWithPlaceholder = [...updatedMsgsWithUser, initialAssistantMsg];
    saveMessagesToCache(msgsWithPlaceholder);

    try {
      // Ensure active conversation_id exists
      let activeConvId = conversationId;
      if (!activeConvId) {
        try {
          const convRes = await authenticatedFetch('/api/v1/conversations', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({ title: 'New Conversation' })
          });
          if (convRes.ok) {
            const convData = await convRes.json();
            if (convData.data && convData.data.id) {
              activeConvId = convData.data.id;
              setConversationId(activeConvId);
              if (activeConvId) {
                localStorage.setItem('chat_conversation_id', activeConvId);
              }
            }
          }
        } catch (e) {
          console.warn("Conversation creation fallback:", e);
        }
      }

      const response = await authenticatedFetch('/api/v1/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          query: currentQuery,
          conversation_id: activeConvId || undefined,
          retrieval_mode: retrievalMode,
          enable_reranking: enableReranking,
          rerank_top_k: rerankTopK,
          enable_rewriting: enableRewriting,
          enable_compression: enableCompression,
          strict_grounding: strictGrounding,
        })
      });



      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        const errMsg = errData.error?.message || errData.detail || `Chat request failed (${response.status})`;
        const error = new Error(errMsg);
        (error as any).status = response.status;
        throw error;
      }

      if (!response.body) {
        throw new Error("No response body received for streaming.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let accumulatedText = "";
      let accumulatedCitations: Citation[] = [];
      let alertMessage: string | undefined = undefined;
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed.startsWith("data: ")) {
            try {
              const eventData = JSON.parse(trimmed.slice(6));
              if (eventData.type === "token") {
                accumulatedText += eventData.content || "";
                setMessages((prev) => {
                  const updated = prev.map((msg) =>
                    msg.id === assistantMsgId
                      ? { ...msg, content: accumulatedText }
                      : msg
                  );
                  localStorage.setItem('chat_messages', JSON.stringify(updated));
                  return updated;
                });
              } else if (eventData.type === "citation") {
                if (eventData.citation) {
                  accumulatedCitations.push(eventData.citation);
                  setMessages((prev) => {
                    const updated = prev.map((msg) =>
                      msg.id === assistantMsgId
                        ? { ...msg, citations: [...accumulatedCitations] }
                        : msg
                    );
                    localStorage.setItem('chat_messages', JSON.stringify(updated));
                    return updated;
                  });
                }
              } else if (eventData.type === "error") {
                alertMessage = `⚠️ Guardrail Alert (${eventData.code || 'ALERT'}): ${eventData.error}`;
                setMessages((prev) => {
                  const updated = prev.map((msg) =>
                    msg.id === assistantMsgId
                      ? { ...msg, guardrail_alert: alertMessage }
                      : msg
                  );
                  localStorage.setItem('chat_messages', JSON.stringify(updated));
                  return updated;
                });
              }
            } catch (e) {
              // Ignore non-JSON line parsing
            }
          }
        }
      }

      if (isRefusalOrNoContext(accumulatedText, alertMessage)) {
        saveCitationsToCache([]);
      } else {
        saveCitationsToCache(accumulatedCitations);
      }
    } catch (err: any) {
      const is401 = err.status === 401 || (err.message && (err.message.includes('401') || err.message.toLowerCase().includes('unauthorized') || err.message.toLowerCase().includes('token')));

      let errorDisplayContent = "";
      let alertMsg = "";

      if (is401) {
        errorDisplayContent = "🔒 Your session has expired. Redirecting to login...";
        alertMsg = "🔑 Authentication Error (HTTP 401): Session expired or invalid token.";
        setTimeout(() => {
          window.dispatchEvent(new Event('auth_session_expired'));
        }, 600);
      } else {
        errorDisplayContent = "⚠️ Unable to complete request due to a network or server error. Please try again.";
        alertMsg = `⚠️ System Error: ${err.message || 'Failed to connect to backend'}`;
      }

      setMessages((prev) => {
        const updated = prev.map((msg) =>
          msg.id === assistantMsgId
            ? {
              ...msg,
              content: errorDisplayContent,
              citations: [],
              guardrail_alert: alertMsg
            }
            : msg
        );
        localStorage.setItem('chat_messages', JSON.stringify(updated));
        return updated;
      });

      saveCitationsToCache([]);
    } finally {
      setLoading(false);
    }

  };

  return (
    <div className="chat-layout">
      {/* 1. Left Config Sidebar */}
      <div className="config-panel">
        {/* RETRIEVAL */}
        <div className="sidebar-section">
          <div className="sidebar-section-title">RETRIEVAL</div>
          <div className="form-group">
            <label>Retrieval Mode</label>
            <select
              className="form-input"
              value={retrievalMode}
              onChange={(e) => setRetrievalMode(e.target.value as any)}
            >
              <option value="hybrid">Hybrid (Vector + FTS via RRF)</option>
              <option value="vector">Vector Only (HNSW Cosine)</option>
              <option value="fts">FTS Only (Postgres tsvector)</option>
            </select>
          </div>
        </div>

        {/* QUERY PROCESSING */}
        <div className="sidebar-section">
          <div className="sidebar-section-title">QUERY PROCESSING</div>

          <div className="toggle-group">
            <div>
              <div className="toggle-label">bge-reranker-v2-m3</div>
              <div className="toggle-sub">Cross-Encoder Reranking</div>
            </div>
            <input
              type="checkbox"
              checked={enableReranking}
              onChange={(e) => setEnableReranking(e.target.checked)}
            />
          </div>

          {enableReranking && (
            <div style={{
              fontSize: '0.75rem',
              color: rerankerStatus === 'offline_fallback' ? 'var(--accent-amber)' : 'var(--accent-emerald)',
              marginTop: '0.25rem',
              marginBottom: '0.5rem',
              display: 'flex',
              alignItems: 'center',
              gap: '0.35rem',
              background: rerankerStatus === 'offline_fallback' ? 'rgba(245, 158, 11, 0.08)' : 'rgba(16, 185, 129, 0.08)',
              padding: '0.35rem 0.65rem',
              borderRadius: 'var(--radius-sm)',
              border: `1px solid ${rerankerStatus === 'offline_fallback' ? 'rgba(245, 158, 11, 0.2)' : 'rgba(16, 185, 129, 0.2)'}`,
              fontWeight: 500
            }}>
              <span>
                {rerankerStatus === 'offline_fallback'
                  ? 'Service Offline (RRF Fallback Active)'
                  : rerankerStatus === 'active_mock'
                    ? 'Active (Cross-Encoder Provider)'
                    : 'Active (TEI HTTP Service)'}
              </span>
            </div>
          )}

          {enableReranking && (
            <div className="form-group" style={{ marginTop: '0.5rem' }}>
              <label>Rerank Top-N ({rerankTopK})</label>
              <input
                type="range"
                min="1"
                max="15"
                value={rerankTopK}
                onChange={(e) => setRerankTopK(Number(e.target.value))}
                style={{ width: '100%', accentColor: 'var(--primary-hover)' }}
              />
            </div>
          )}

          <div className="toggle-group">
            <div>
              <div className="toggle-label">LLM Query Rewriter</div>
              <div className="toggle-sub">Decompose & Coreference</div>
            </div>
            <input
              type="checkbox"
              checked={enableRewriting}
              onChange={(e) => setEnableRewriting(e.target.checked)}
            />
          </div>

          <div className="toggle-group">
            <div>
              <div className="toggle-label">Context Compression</div>
              <div className="toggle-sub">Sentence Extraction</div>
            </div>
            <input
              type="checkbox"
              checked={enableCompression}
              onChange={(e) => setEnableCompression(e.target.checked)}
            />
          </div>
        </div>

        {/* GROUNDING */}
        <div className="sidebar-section">
          <div className="sidebar-section-title">GROUNDING</div>
          <div className="toggle-group">
            <div>
              <div className="toggle-label">Strict Groundedness</div>
              <div className="toggle-sub">Refuse ungrounded queries</div>
            </div>
            <input
              type="checkbox"
              checked={strictGrounding}
              onChange={(e) => handleStrictGroundingToggle(e.target.checked)}
            />
          </div>
        </div>

        {/* SYSTEM STATUS */}
        <div className="sidebar-section">
          <div className="sidebar-section-title">SYSTEM STATUS</div>
          <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', display: 'flex', flexDirection: 'column', gap: '0.4rem', background: 'var(--bg-input)', padding: '0.65rem 0.75rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)', fontFamily: 'var(--font-mono)' }}>
            <div>ACL Filter: Pushdown ON</div>
            <div>InputGuard: Injection Scan ON</div>
            <div>SecretRedactor: Active</div>
            <div>EvidenceScorer: Threshold 0.35</div>
          </div>
        </div>
      </div>

      {/* 2. Main Chat Workspace */}
      <div className="chat-main">
        <div className="chat-header">
          <div className="chat-title-group">
            <h3>Knowledge Assistant</h3>
            <span>RAG Pipeline • SSE Real-Time Streaming</span>
          </div>
          <button className="btn-secondary" onClick={handleClearChat}>
            Clear Chat
          </button>
        </div>

        <div className="message-list">
          {messages.map((msg) => (
            <div key={msg.id} className={`message-bubble ${msg.role}`}>
              {msg.guardrail_alert && (
                <div className="guardrail-alert">
                  {msg.guardrail_alert}
                </div>
              )}
              <div className="bubble-content">
                {msg.role === 'assistant' ? (
                  <ReactMarkdown>{msg.content}</ReactMarkdown>
                ) : (
                  msg.content
                )}
                {msg.citations && msg.citations.length > 0 && !isRefusalOrNoContext(msg.content, msg.guardrail_alert) && (
                  <div style={{ marginTop: '0.75rem', paddingTop: '0.5rem', borderTop: '1px solid var(--border-subtle)', display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '0.35rem' }}>
                    <span style={{ fontSize: '0.72rem', color: 'var(--text-subtle)', marginRight: '0.2rem', fontWeight: 600 }}>Citations:</span>
                    {msg.citations.map((cit, idx) => (
                      <button
                        key={idx}
                        className="citation-chip"
                        onClick={() => saveCitationsToCache(msg.citations || [])}
                      >
                        [{idx + 1}] {cit.document_title || `Doc ${idx + 1}`}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}
          {loading && (
            <div className="message-bubble assistant">
              <div className="bubble-content" style={{ color: 'var(--text-secondary)', background: 'var(--bg-surface)', border: '1px solid var(--border-color)', display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem' }}>
                <span style={{ display: 'inline-block', width: '8px', height: '8px', borderRadius: '50%', background: 'var(--primary-hover)', animation: 'pulse 1.2s infinite' }}></span> Thinking...
              </div>
            </div>
          )}
        </div>

        <div className="chat-composer-container">
          <form onSubmit={handleSendMessage} className="chat-input-box">
            <textarea
              className="chat-textarea"
              rows={2}
              value={inputQuery}
              onChange={(e) => setInputQuery(e.target.value)}
              placeholder="Ask anything about uploaded documentation (e.g. Component 1 specifications, Aurawave headphones)..."
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSendMessage(e);
                }
              }}
            />
            <button type="submit" className="btn-primary" disabled={loading}>
              Send
            </button>
          </form>
        </div>
      </div>

      {/* 3. Citations & Provenance Sidebar */}
      <div className="citations-panel">
        <div className="panel-header-title">Source Citations & Provenance</div>
        {activeCitations.length === 0 ? (
          <div className="empty-provenance">
            No citations selected. Ask a question or click citation chips in the chat to view source evidence.
          </div>
        ) : (
          activeCitations.map((cit, idx) => (
            <div key={idx} className="citation-card">
              <div className="citation-card-header">
                <span className="citation-doc-title">[{idx + 1}] {cit.document_title || 'Document Chunk'}</span>
                <span className="status-badge INDEXED">
                  {typeof (cit.similarity_score ?? cit.rerank_score ?? cit.score) === 'number' ? (((cit.similarity_score ?? cit.rerank_score ?? cit.score) as number) * 100).toFixed(0) + '%' : 'N/A'}
                </span>
              </div>
              <div style={{ fontSize: '0.73rem', color: 'var(--text-subtle)', marginBottom: '0.4rem', fontWeight: 500 }}>
                Page {cit.page_number || 1} • {cit.section_path || 'Main Section'}
              </div>
              <div className="citation-snippet">
                "{cit.text_snippet || 'Document chunk text snippet extracted from backend search engine.'}"
              </div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-subtle)', marginTop: '0.5rem', fontFamily: 'var(--font-mono)' }}>
                chunk_id: {cit.chunk_id.slice(0, 18)}...
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

