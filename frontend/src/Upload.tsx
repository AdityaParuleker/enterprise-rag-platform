import React, { useState, useEffect } from 'react';
import { getApiUrl } from './config';

interface DocItem {
  id: string;
  filename: string;
  status: 'INDEXED' | 'QUEUED' | 'PARSING' | 'CHUNKING' | 'EMBEDDING' | 'FAILED' | string;
  version: number;
  created_at: string;
  file_size?: number;
}

const getIngestionProgress = (status: string) => {
  const s = (status || '').toUpperCase();
  switch (s) {
    case 'INDEXED':
    case 'COMPLETED':
      return { percentage: 100, label: '100%', color: 'var(--accent-emerald)', bg: 'var(--accent-emerald)' };
    case 'EMBEDDING':
      return { percentage: 85, label: '85%', color: 'var(--accent-cyan)', bg: 'var(--accent-cyan)' };
    case 'CHUNKING':
      return { percentage: 60, label: '60%', color: 'var(--primary-hover)', bg: 'var(--primary-hover)' };
    case 'PARSING':
      return { percentage: 35, label: '35%', color: 'var(--accent-amber)', bg: 'var(--accent-amber)' };
    case 'QUEUED':
      return { percentage: 15, label: '15%', color: 'var(--accent-amber)', bg: 'var(--accent-amber)' };
    case 'FAILED':
      return { percentage: 0, label: '0%', color: 'var(--accent-rose)', bg: 'var(--accent-rose)' };
    default:
      return { percentage: 0, label: '0%', color: 'var(--text-subtle)', bg: 'var(--text-subtle)' };
  }
};



export const Upload: React.FC = () => {
  const [documents, setDocuments] = useState<DocItem[]>(() => {
    const cached = localStorage.getItem('tenant_documents');
    if (cached) {
      try {
        const parsed = JSON.parse(cached);
        if (Array.isArray(parsed)) return parsed;
      } catch (e) {}
    }
    return [];
  });
  const [uploading, setUploading] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [toastMessage, setToastMessage] = useState<{ title: string; body: string; type: 'warning' | 'success' | 'error' } | null>(null);

  const showToast = (title: string, body: string, type: 'warning' | 'success' | 'error' = 'warning') => {
    setToastMessage({ title, body, type });
    setTimeout(() => {
      setToastMessage(null);
    }, 6000);
  };

  const saveDocumentsToCache = (newDocs: DocItem[]) => {
    setDocuments(newDocs);
    localStorage.setItem('tenant_documents', JSON.stringify(newDocs));
  };

  const fetchDocuments = async () => {
    try {
      const token = localStorage.getItem('auth_token') || '';
      const response = await fetch(getApiUrl('/api/v1/documents'), {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (response.ok) {
        const resData = await response.json();
        const docArray = resData.data || resData.documents;
        if (Array.isArray(docArray)) {
          const mapped: DocItem[] = docArray.map((d: any) => ({
            id: d.id || `doc-${Date.now()}`,
            filename: d.filename || 'uploaded_document',
            status: d.status || 'INDEXED',
            version: d.version || 1,
            created_at: typeof d.created_at === 'string' ? d.created_at.slice(0, 19).replace('T', ' ') : '',
            file_size: d.file_size || 12000
          }));

          saveDocumentsToCache(mapped);
          return;
        }
      }
    } catch (e) {
      // Keep existing cached documents
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, []);

  useEffect(() => {
    const IN_PROGRESS_STATUSES = ['QUEUED', 'PARSING', 'CHUNKING', 'EMBEDDING', 'PROCESSING'];
    const hasPending = documents.some(d => IN_PROGRESS_STATUSES.includes((d.status || '').toUpperCase()));
    if (!hasPending) return;

    const timer = setInterval(() => {
      fetchDocuments();
    }, 1000);

    return () => clearInterval(timer);
  }, [documents]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const handleUploadSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile || uploading) return;

    setUploading(true);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const token = localStorage.getItem('auth_token') || '';
      const response = await fetch(getApiUrl('/api/v1/documents'), {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` },
        body: formData
      });

      if (!response.ok) {
        if (response.status === 401) {
          throw new Error('Authentication token expired or invalid. Please click "Sign Out" at top-right and sign in again.');
        }
        const errData = await response.json().catch(() => ({}));
        const msg = typeof errData.detail === 'string'
          ? errData.detail
          : (errData.message || 'Upload failed. Please check server connection.');
        throw new Error(msg);
      }

      const resJson = await response.json();
      const payload = resJson.data || resJson;

      if (payload.deduplicated) {
        // Show duplicate file popup message
        showToast(
          'File Already Exists',
          `The document "${selectedFile.name}" already exists in the system for this tenant. Duplicate upload skipped.`,
          'warning'
        );
        await fetchDocuments();
        setSelectedFile(null);
        return;
      }

      showToast(
        'Upload Successful',
        `Document "${selectedFile.name}" uploaded and queued for processing.`,
        'success'
      );
      await fetchDocuments();
      setSelectedFile(null);
    } catch (err: any) {
      showToast('Upload Error', err.message || 'Failed to upload document', 'error');
      await fetchDocuments();
      setSelectedFile(null);
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (docId: string) => {
    try {
      const token = localStorage.getItem('auth_token') || '';
      await fetch(getApiUrl(`/api/v1/documents/${docId}`), {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${token}` }
      });
    } catch (e) {
      // Proceed
    }
    const updated = documents.filter((d) => d.id !== docId);
    saveDocumentsToCache(updated);
  };

  const handleRetry = async (docId: string) => {
    try {
      const token = localStorage.getItem('auth_token') || '';
      const response = await fetch(getApiUrl(`/api/v1/documents/${docId}/retry`), {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (response.ok) {
        showToast('Retry Dispatched', 'Document re-queued for ingestion.', 'success');
        await fetchDocuments();
      } else {
        const errJson = await response.json().catch(() => ({}));
        showToast('Retry Failed', errJson.detail || 'Could not retry ingestion.', 'error');
      }
    } catch (e: any) {
      showToast('Retry Failed', e.message || 'Error triggering retry.', 'error');
    }
  };


  return (
    <div className="doc-grid" style={{ position: 'relative' }}>
      {/* Toast Popup Notification */}
      {toastMessage && (
        <div style={{
          position: 'fixed',
          top: '24px',
          right: '24px',
          zIndex: 9999,
          backgroundColor: toastMessage.type === 'warning' ? '#1f1a00' : toastMessage.type === 'error' ? '#2d0607' : '#042f1a',
          border: `1.5px solid ${toastMessage.type === 'warning' ? '#f59e0b' : toastMessage.type === 'error' ? '#ef4444' : '#10b981'}`,
          borderRadius: 'var(--radius-md)',
          padding: '0.9rem 1.1rem',
          boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.6), 0 8px 10px -6px rgba(0, 0, 0, 0.5)',
          display: 'flex',
          alignItems: 'flex-start',
          gap: '0.75rem',
          maxWidth: '400px',
          animation: 'slideIn 0.3s cubic-bezier(0.16, 1, 0.3, 1)'
        }}>
          <div style={{ flex: 1 }}>
            <div style={{
              fontWeight: 600,
              fontSize: '0.9rem',
              color: toastMessage.type === 'warning' ? '#fde68a' : toastMessage.type === 'error' ? '#fca5a5' : '#a7f3d0',
              marginBottom: '0.2rem'
            }}>
              {toastMessage.title}
            </div>
            <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.45 }}>
              {toastMessage.body}
            </div>
          </div>
          <button
            onClick={() => setToastMessage(null)}
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--text-subtle)',
              fontSize: '1.2rem',
              cursor: 'pointer',
              padding: '0',
              lineHeight: 1
            }}
          >
            ×
          </button>
        </div>
      )}
      {/* Upload Dropzone Panel */}
      <div className="config-panel" style={{ overflow: 'visible' }}>
        <div className="panel-header-title">Document Ingestion Pipeline</div>
        <p style={{ fontSize: '0.83rem', color: 'var(--text-secondary)', marginBottom: '1.25rem', lineHeight: 1.5 }}>
          Upload PDF, DOCX, Markdown, HTML, or CSV. Ingestion executes magic-byte validation, SecretRedactor scanning, chunking, and pgvector embedding indexing.
        </p>

        <form onSubmit={handleUploadSubmit}>
          <div className="dropzone" onClick={() => document.getElementById('fileInput')?.click()}>
            <div style={{ fontSize: '0.92rem', fontWeight: 600, color: 'var(--text-main)' }}>
              {selectedFile ? selectedFile.name : 'Choose File or Drag & Drop'}
            </div>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-subtle)', marginTop: '0.35rem' }}>
              Maximum file size: 50MB (PDF, DOCX, MD, HTML, CSV, TXT)
            </div>
            <input
              id="fileInput"
              type="file"
              onChange={handleFileChange}
              style={{ display: 'none' }}
              accept=".pdf,.docx,.md,.html,.csv,.txt"
            />
          </div>

          <button
            type="submit"
            className="btn-primary"
            style={{ width: '100%', marginTop: '1rem' }}
            disabled={!selectedFile || uploading}
          >
            {uploading ? 'Processing Ingestion State Machine...' : 'Upload & Index Document'}
          </button>
        </form>

        <div style={{ marginTop: '1.25rem', fontSize: '0.78rem', color: 'var(--text-subtle)', lineHeight: 1.6, background: 'var(--bg-input)', padding: '0.75rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
          <div><strong>Security Checks:</strong> MIME Magic Bytes, Decompression Bomb Defense, SSRF Prevention</div>
          <div><strong>Secret Redaction:</strong> AWS Keys, Private Keys, JWTs scrubbed prior to embedding</div>
        </div>
      </div>

      {/* Indexed Document List Table Panel */}
      <div className="citations-panel">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
          <div className="panel-header-title" style={{ margin: 0 }}>Tenant Indexed Documents ({documents.length})</div>
          <button className="btn-secondary" onClick={fetchDocuments}>Refresh</button>
        </div>

        <div className="table-responsive">
          <table className="doc-table">
            <thead>
              <tr>
                <th>Document Name</th>
                <th>Version</th>
                <th>Status</th>
                <th>Uploaded Date</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {documents.length === 0 ? (
                <tr>
                  <td colSpan={5} style={{ textAlign: 'center', color: 'var(--text-subtle)', padding: '2rem' }}>
                    No indexed documents found for this tenant. Upload a document to get started.
                  </td>
                </tr>
              ) : (
                documents.map((doc) => (
                  <tr key={doc.id}>
                    <td style={{ fontWeight: 600, color: 'var(--text-main)' }}>{doc.filename}</td>
                    <td><span className="tenant-badge">v{doc.version}</span></td>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
                        <span className={`status-badge ${doc.status}`}>
                          {doc.status}
                        </span>
                        {(() => {
                          const prog = getIngestionProgress(doc.status);
                          return (
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                              <div style={{ width: '60px', height: '6px', background: 'var(--bg-input)', borderRadius: '3px', overflow: 'hidden', border: '1px solid var(--border-subtle)' }}>
                                <div style={{ width: `${prog.percentage}%`, height: '100%', background: prog.bg, transition: 'width 0.4s ease' }} />
                              </div>
                              <span style={{ fontSize: '0.72rem', fontWeight: 700, color: prog.color, fontFamily: 'var(--font-mono)' }}>
                                {prog.label}
                              </span>
                            </div>
                          );
                        })()}
                      </div>
                    </td>
                    <td style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>{doc.created_at}</td>
                    <td>
                      <div style={{ display: 'flex', gap: '0.4rem' }}>
                        {(doc.status === 'FAILED' || doc.status === 'QUEUED') && (
                          <button
                            className="btn-secondary"
                            style={{ padding: '0.2rem 0.5rem', fontSize: '0.73rem', color: 'var(--accent-cyan)', borderColor: 'var(--accent-cyan)' }}
                            onClick={() => handleRetry(doc.id)}
                          >
                            Retry
                          </button>
                        )}
                        <button
                          className="btn-secondary"
                          style={{ padding: '0.2rem 0.5rem', fontSize: '0.73rem', color: 'var(--accent-rose)' }}
                          onClick={() => handleDelete(doc.id)}
                        >
                          Delete
                        </button>
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
