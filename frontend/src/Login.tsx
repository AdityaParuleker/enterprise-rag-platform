import React, { useState } from 'react';

interface LoginProps {
  onLoginSuccess: (token: string, tenantId: string, email: string) => void;
}

export const Login: React.FC<LoginProps> = ({ onLoginSuccess }) => {
  const [email, setEmail] = useState('demo_admin@enterprise.com');
  const [password, setPassword] = useState('SecurePass123!');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/v1/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });

      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.detail || 'Authentication failed. Please check your credentials.');
      }

      const resJson = await response.json();
      const authData = resJson.data || resJson;
      const token = authData.access_token;
      const tenantId = authData.tenant_id || 'ee45761a-f471-4f41-97ff-697d927e0745';

      if (!token) {
        throw new Error('Access token missing from server response.');
      }

      localStorage.setItem('auth_token', token);
      if (authData.refresh_token) {
        localStorage.setItem('refresh_token', authData.refresh_token);
      }
      localStorage.setItem('tenant_id', tenantId);
      localStorage.setItem('user_email', email);

      onLoginSuccess(token, tenantId, email);
    } catch (err: any) {
      setError(err.message || 'Authentication failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-wrapper">
      <div className="login-card">
        <div className="login-header">
          <h2>Enterprise Knowledge</h2>
          <span style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-muted)' }}>Intelligence Platform</span>
          <p style={{ marginTop: '0.5rem', fontSize: '0.8rem', color: 'var(--text-subtle)' }}>Secure multi-tenant workspace login</p>
        </div>

        {error && (
          <div className="guardrail-alert">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label>Work Email</label>
            <input
              type="email"
              className="form-input"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="admin@enterprise.com"
              required
            />
          </div>

          <div className="form-group">
            <label>Password</label>
            <input
              type="password"
              className="form-input"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••••••"
              required
            />
          </div>

          <button type="submit" className="btn-primary" style={{ width: '100%', marginTop: '1rem' }} disabled={loading}>
            {loading ? 'Authenticating...' : 'Sign In to Workspace'}
          </button>
        </form>

        <div className="demo-credentials">
          <strong>Demo Sandbox Session:</strong>
          <div>Email: <code>demo_admin@enterprise.com</code></div>
          <div>Tenant Isolation: <code>ee45761a-f471-4f41-97ff...</code></div>
        </div>
      </div>
    </div>
  );
};
