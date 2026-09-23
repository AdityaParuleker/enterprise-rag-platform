import React, { useState } from 'react';

interface LoginProps {
  onLoginSuccess: (token: string, tenantId: string, email: string) => void;
}

export const Login: React.FC<LoginProps> = ({ onLoginSuccess }) => {
  const [mode, setMode] = useState<'login' | 'register'>('login');
  const [email, setEmail] = useState('demo_admin@enterprise.com');
  const [password, setPassword] = useState('password123');
  const [tenantName, setTenantName] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);

  const handleTabSwitch = (newMode: 'login' | 'register') => {
    setMode(newMode);
    setError(null);
    setSuccessMsg(null);
    if (newMode === 'register') {
      if (email === 'demo_admin@enterprise.com') {
        setEmail('');
        setPassword('');
      }
    } else {
      if (!email) {
        setEmail('demo_admin@enterprise.com');
        setPassword('password123');
      }
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setSuccessMsg(null);

    if (mode === 'login') {
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
    } else {
      // Register Mode
      try {
        const regRes = await fetch('/api/v1/auth/register', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            email,
            password,
            tenant_name: tenantName.trim() || undefined,
          }),
        });

        if (!regRes.ok) {
          const data = await regRes.json().catch(() => ({}));
          throw new Error(data.detail || 'Registration failed. User may already exist.');
        }

        setSuccessMsg('Account created successfully! Signing you in...');

        // Automatically log in after registration
        const loginRes = await fetch('/api/v1/auth/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ email, password }),
        });

        if (!loginRes.ok) {
          setMode('login');
          throw new Error('Registration succeeded! Please sign in with your new credentials.');
        }

        const resJson = await loginRes.json();
        const authData = resJson.data || resJson;
        const token = authData.access_token;
        const tenantId = authData.tenant_id;

        localStorage.setItem('auth_token', token);
        if (authData.refresh_token) {
          localStorage.setItem('refresh_token', authData.refresh_token);
        }
        localStorage.setItem('tenant_id', tenantId);
        localStorage.setItem('user_email', email);

        onLoginSuccess(token, tenantId, email);
      } catch (err: any) {
        setError(err.message || 'Registration failed');
      } finally {
        setLoading(false);
      }
    }
  };

  return (
    <div className="login-wrapper">
      <div className="login-card">
        <div className="login-header">
          <h2>Enterprise Knowledge</h2>
          <span style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-muted)' }}>Intelligence Platform</span>
          <p style={{ marginTop: '0.5rem', fontSize: '0.8rem', color: 'var(--text-subtle)' }}>
            {mode === 'login' ? 'Secure multi-tenant workspace login' : 'Create your workspace account'}
          </p>
        </div>

        {/* Tab Toggle: Sign In vs Register */}
        <div className="auth-tabs">
          <button
            type="button"
            className={`auth-tab ${mode === 'login' ? 'active' : ''}`}
            onClick={() => handleTabSwitch('login')}
          >
            Sign In
          </button>
          <button
            type="button"
            className={`auth-tab ${mode === 'register' ? 'active' : ''}`}
            onClick={() => handleTabSwitch('register')}
          >
            Register Account
          </button>
        </div>

        {error && (
          <div className="guardrail-alert">
            {error}
          </div>
        )}

        {successMsg && (
          <div className="alert-success">
            {successMsg}
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
              placeholder="user@enterprise.com"
              required
            />
          </div>

          {mode === 'register' && (
            <div className="form-group">
              <label>Organization Name (Optional)</label>
              <input
                type="text"
                className="form-input"
                value={tenantName}
                onChange={(e) => setTenantName(e.target.value)}
                placeholder="Acme Corp"
              />
            </div>
          )}

          <div className="form-group">
            <label>Password</label>
            <input
              type="password"
              className="form-input"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••••••"
              minLength={6}
              required
            />
          </div>

          <button type="submit" className="btn-primary" style={{ width: '100%', marginTop: '1rem' }} disabled={loading}>
            {loading ? (mode === 'login' ? 'Authenticating...' : 'Creating Account...') : (mode === 'login' ? 'Sign In to Workspace' : 'Create Account & Sign In')}
          </button>
        </form>

        {mode === 'login' ? (
          <div className="demo-credentials">
            <strong>Demo Sandbox Session:</strong>
            <div>Email: <code>demo_admin@enterprise.com</code></div>
            <div>Password: <code>password123</code></div>
            <div>Tenant Isolation: <code>ee45761a-f471-4f41-97ff...</code></div>
          </div>
        ) : (
          <div style={{ marginTop: '1.25rem', textAlign: 'center', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            Already have an account?{' '}
            <button
              type="button"
              onClick={() => handleTabSwitch('login')}
              style={{ background: 'none', border: 'none', color: '#a78bfa', cursor: 'pointer', fontWeight: 600, textDecoration: 'underline' }}
            >
              Sign In here
            </button>
          </div>
        )}
      </div>
    </div>
  );
};
