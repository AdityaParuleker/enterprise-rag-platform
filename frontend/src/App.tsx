import React, { useState, useEffect } from 'react';
import { Login } from './Login';
import { Chat } from './Chat';
import { Upload } from './Upload';
import { EvalDashboard } from './EvalDashboard';

type TabType = 'chat' | 'documents' | 'eval';

export const App: React.FC = () => {
  const [token, setToken] = useState<string | null>(null);
  const [tenantId, setTenantId] = useState<string | null>(null);
  const [userEmail, setUserEmail] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<TabType>(() => {
    const saved = localStorage.getItem('active_tab') as TabType;
    if (saved === 'chat' || saved === 'documents' || saved === 'eval') {
      return saved;
    }
    return 'chat';
  });

  const handleTabChange = (tab: TabType) => {
    setActiveTab(tab);
    localStorage.setItem('active_tab', tab);
  };

  const handleLogout = () => {
    localStorage.removeItem('auth_token');
    localStorage.removeItem('refresh_token');
    localStorage.removeItem('tenant_id');
    localStorage.removeItem('user_email');
    localStorage.removeItem('chat_messages');
    localStorage.removeItem('chat_active_citations');
    localStorage.removeItem('chat_conversation_id');
    localStorage.removeItem('active_tab');
    setToken(null);
    setTenantId(null);
    setUserEmail(null);
  };

  useEffect(() => {
    const savedToken = localStorage.getItem('auth_token');
    const savedTenant = localStorage.getItem('tenant_id');
    const savedEmail = localStorage.getItem('user_email');

    if (savedToken) {
      setToken(savedToken);
      setTenantId(savedTenant || 'ee45761a-f471-4f41-97ff-697d927e0745');
      setUserEmail(savedEmail || 'admin@enterprise.com');
    }

    const handleSessionExpired = () => {
      handleLogout();
    };

    window.addEventListener('auth_session_expired', handleSessionExpired);
    return () => {
      window.removeEventListener('auth_session_expired', handleSessionExpired);
    };
  }, []);

  const handleLoginSuccess = (token: string, tenantId: string, email: string) => {
    setToken(token);
    setTenantId(tenantId);
    setUserEmail(email);
  };

  if (!token) {
    return <Login onLoginSuccess={handleLoginSuccess} />;
  }

  return (
    <div className="app-container">
      {/* Top Header Navigation */}
      <header className="top-nav">
        <div className="brand">
          <div className="brand-icon">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
            </svg>
          </div>
          <div>
            <div className="brand-title">Enterprise Knowledge</div>
            <div className="brand-subtitle">Intelligence Platform</div>
          </div>
        </div>

        {/* Tab Navigation */}
        <nav className="nav-tabs">
          <button
            className={`tab-btn ${activeTab === 'chat' ? 'active' : ''}`}
            onClick={() => handleTabChange('chat')}
          >
            Chat
          </button>
          <button
            className={`tab-btn ${activeTab === 'documents' ? 'active' : ''}`}
            onClick={() => handleTabChange('documents')}
          >
            Document Ingestion
          </button>
          <button
            className={`tab-btn ${activeTab === 'eval' ? 'active' : ''}`}
            onClick={() => handleTabChange('eval')}
          >
            Evaluation & Benchmarking
          </button>
        </nav>

        {/* User Info & Logout */}
        <div className="user-profile">
          <span className="tenant-badge">Tenant: {tenantId ? tenantId.slice(0, 8) : 'ee45761a'}...</span>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{userEmail}</span>
          <button className="btn-secondary" onClick={handleLogout}>
            Sign Out
          </button>
        </div>
      </header>

      {/* Main View Area */}
      <main className="main-content">
        {activeTab === 'chat' && <Chat />}
        {activeTab === 'documents' && <Upload />}
        {activeTab === 'eval' && <EvalDashboard />}
      </main>
    </div>
  );
};

export default App;
