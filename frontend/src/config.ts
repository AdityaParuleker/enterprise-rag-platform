// API Configuration Helper for Local Dev and Vercel Production Deployment

export const API_BASE_URL = 
  import.meta.env.VITE_API_BASE_URL || 
  (import.meta.env.PROD ? 'https://enterprise-rag-platform-zygo.onrender.com' : '');

export const getApiUrl = (path: string): string => {
  const cleanPath = path.startsWith('/') ? path : `/${path}`;
  return `${API_BASE_URL}${cleanPath}`;
};
