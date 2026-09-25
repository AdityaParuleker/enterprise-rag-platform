// API Configuration Helper for Local Dev and Vercel Production Deployment

export const API_BASE_URL = 
  import.meta.env.VITE_API_BASE_URL || 
  (import.meta.env.PROD ? 'https://enterprise-rag-platform-zygo.onrender.com' : '');

export const getApiUrl = (path: string): string => {
  const cleanPath = path.startsWith('/') ? path : `/${path}`;
  return `${API_BASE_URL}${cleanPath}`;
};

/**
 * Decode base64 payload of a JWT token string.
 */
export const decodeJwtPayload = (token: string): any | null => {
  try {
    const parts = token.split('.');
    if (parts.length !== 3) return null;
    let base64Url = parts[1].replace(/-/g, '+').replace(/_/g, '/');
    while (base64Url.length % 4) {
      base64Url += '=';
    }
    const jsonPayload = decodeURIComponent(
      atob(base64Url)
        .split('')
        .map((c) => '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2))
        .join('')
    );
    return JSON.parse(jsonPayload);
  } catch (e) {
    return null;
  }
};

/**
 * Check if the JWT access token is expiring soon (within bufferSeconds) or already expired.
 */
export const isTokenExpiringSoon = (token: string | null, bufferSeconds = 300): boolean => {
  if (!token) return true;
  const payload = decodeJwtPayload(token);
  if (!payload || !payload.exp) return false; // If non-decodable, let backend validate
  const nowInSeconds = Math.floor(Date.now() / 1000);
  return payload.exp - nowInSeconds <= bufferSeconds;
};

/**
 * Perform a silent token refresh using the stored refresh_token.
 */
export const refreshAccessToken = async (): Promise<string | null> => {
  const refreshToken = localStorage.getItem('refresh_token');
  if (!refreshToken) return null;

  try {
    const response = await fetch(getApiUrl('/api/v1/auth/refresh'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ refresh_token: refreshToken })
    });

    if (!response.ok) {
      // Refresh token is revoked or expired
      localStorage.removeItem('auth_token');
      localStorage.removeItem('refresh_token');
      return null;
    }

    const resJson = await response.json();
    const data = resJson.data || resJson;
    const newAccessToken = data.access_token;
    const newRefreshToken = data.refresh_token;

    if (newAccessToken) {
      localStorage.setItem('auth_token', newAccessToken);
      if (newRefreshToken) {
        localStorage.setItem('refresh_token', newRefreshToken);
      }
      return newAccessToken;
    }
  } catch (e) {
    // Network error during refresh
  }
  return null;
};

/**
 * Pre-flight auth check: If token is expiring within 5 minutes or expired, silently refresh it before API calls.
 */
export const ensureValidToken = async (): Promise<string> => {
  let token = localStorage.getItem('auth_token') || '';
  if (isTokenExpiringSoon(token, 300)) {
    const refreshed = await refreshAccessToken();
    if (refreshed) {
      token = refreshed;
    }
  }
  return token;
};

