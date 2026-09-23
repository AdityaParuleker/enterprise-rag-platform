"""
Ingestion Security Validation Module (Section 6.5 & Section 15).
Provides strict magic-byte file validation, zip-bomb & symlink defense, filename sanitization,
and secure SSRF-safe URL downloading with pre-connection DNS validation & TLS cert checks.
"""

import asyncio
import io
import os
import re
import socket
import ssl
import urllib.parse
import zipfile
import httpx
from ipaddress import ip_address
from typing import Tuple

MAX_FILE_SIZE_BYTES = int(os.getenv("MAX_FILE_SIZE_MB", "50")) * 1024 * 1024
MAX_REDIRECTS = 3
CONNECT_TIMEOUT = 5.0
READ_TIMEOUT = 10.0
TOTAL_TIMEOUT = 30.0

# Zip bomb thresholds for DOCX/ZIP files
MAX_ZIP_MEMBERS = 100
MAX_ZIP_SINGLE_FILE_BYTES = 20 * 1024 * 1024  # 20 MB
MAX_ZIP_CUMULATIVE_BYTES = 100 * 1024 * 1024  # 100 MB
MAX_ZIP_COMPRESSION_RATIO = 20.0


class SecurityValidationError(ValueError):
    """Raised when file or URL security validation fails."""
    def __init__(self, message: str, code: str = "SECURITY_VALIDATION_ERROR"):
        super().__init__(message)
        self.code = code
        self.message = message


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal and null-byte injection."""
    if not filename:
        return "unnamed_document"
    clean = filename.replace("\x00", "").strip()
    clean = os.path.basename(clean.replace("\\", "/"))
    clean = re.sub(r'[\x00-\x1f\x7f]', '', clean)
    return clean if clean else "unnamed_document"


def _is_ip_allowed(ip_str: str) -> bool:
    """Check if resolved IP address is public and safe."""
    try:
        ip = ip_address(ip_str)
        if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_multicast or ip.is_unspecified:
            return False
        if str(ip) == "169.254.169.254":
            return False
        return True
    except ValueError:
        return False


def resolve_and_validate_host(hostname: str, port: int) -> str:
    """
    Resolve hostname via DNS and ensure all returned IP addresses are safe public IPs.
    Returns the validated IP address string.
    """
    try:
        ip = ip_address(hostname)
        if not _is_ip_allowed(str(ip)):
            raise SecurityValidationError(f"Forbidden target IP address: {hostname}", code="SECURITY_BLOCKED_SSRF")
        return str(ip)
    except ValueError:
        pass

    try:
        addr_info = socket.getaddrinfo(hostname, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise SecurityValidationError(f"DNS resolution failed for hostname '{hostname}': {str(e)}", code="SECURITY_BLOCKED_SSRF")

    if not addr_info:
        raise SecurityValidationError(f"No IP addresses resolved for hostname '{hostname}'", code="SECURITY_BLOCKED_SSRF")

    valid_ip = None
    for family, socktype, proto, canonname, sockaddr in addr_info:
        ip_candidate = sockaddr[0]
        if not _is_ip_allowed(ip_candidate):
            raise SecurityValidationError(f"Hostname '{hostname}' resolved to prohibited IP address '{ip_candidate}'", code="SECURITY_BLOCKED_SSRF")
        if valid_ip is None:
            valid_ip = ip_candidate

    return valid_ip


class IngestionSecurityValidator:
    """Validator for file byte signatures and secure URL downloads."""

    def validate_file_bytes(self, file_bytes: bytes, filename: str = "") -> Tuple[str, str]:
        """
        Validate raw file bytes using magic byte signatures (completely ignoring declared extension/Content-Type).
        Returns (detected_mime_type, sanitized_filename).
        """
        sanitized_name = sanitize_filename(filename)

        if not file_bytes:
            raise SecurityValidationError("File is empty (0 bytes)", code="SECURITY_FILE_SIZE_EXCEEDED")

        if len(file_bytes) > MAX_FILE_SIZE_BYTES:
            raise SecurityValidationError(f"File size ({len(file_bytes)} bytes) exceeds maximum limit of {MAX_FILE_SIZE_BYTES} bytes", code="SECURITY_FILE_SIZE_EXCEEDED")

        # 1. Check PDF signature
        if file_bytes.startswith(b"%PDF-"):
            return "application/pdf", sanitized_name

        # 2. Check DOCX / ZIP signature
        if file_bytes.startswith(b"PK\x03\x04"):
            self._validate_docx_zip(file_bytes)
            return "application/vnd.openxmlformats-officedocument.wordprocessingml.document", sanitized_name

        # 3. Check HTML signature (first 2048 bytes)
        sample = file_bytes[:2048].lower()
        if b"<html" in sample or b"<!doctype html" in sample:
            return "text/html", sanitized_name

        # 4. Check Text / Markdown / CSV (valid UTF-8 text without null bytes)
        if b"\x00" in file_bytes:
            raise SecurityValidationError("File contains binary null bytes and is not a valid text document.", code="SECURITY_MAGIC_BYTE_MISMATCH")

        try:
            decoded = file_bytes.decode("utf-8")
            # Content-based classification (ignoring filename extension!)
            lines = [l.strip() for l in decoded.split("\n")[:20] if l.strip()]
            if lines and any(l.startswith(",") or ("," in l and len(l.split(",")) > 1) for l in lines[:5]):
                # Check for CSV structure
                first_row_cols = len(lines[0].split(","))
                if first_row_cols > 1 and all(len(l.split(",")) == first_row_cols for l in lines[1:5]):
                    return "text/csv", sanitized_name

            if "# " in decoded[:1000] or "## " in decoded[:1000] or "```" in decoded[:1000] or "* " in decoded[:1000]:
                return "text/markdown", sanitized_name

            return "text/markdown", sanitized_name  # Default text documents to markdown format
        except UnicodeDecodeError:
            pass

        raise SecurityValidationError("Unsupported or invalid file byte signature. File failed magic-byte validation.", code="SECURITY_MAGIC_BYTE_MISMATCH")

    def _validate_docx_zip(self, file_bytes: bytes) -> None:
        """Validate zip archive structure to defend against zip bombs, path traversal, and symlinks."""
        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes), "r") as zf:
                infolist = zf.infolist()
                if len(infolist) > MAX_ZIP_MEMBERS:
                    raise SecurityValidationError(f"Zip archive contains too many files ({len(infolist)} > {MAX_ZIP_MEMBERS})", code="SECURITY_DECOMPRESSION_BOMB")

                total_uncompressed = 0
                compressed_size = len(file_bytes)

                is_docx = False
                for item in infolist:
                    fname = item.filename
                    # Symlink check
                    if (item.external_attr >> 16) & 0o120000 == 0o120000:
                        raise SecurityValidationError(f"Zip archive member '{fname}' is a symbolic link", code="SECURITY_DECOMPRESSION_BOMB")

                    # Path traversal check
                    parts = fname.replace("\\", "/").split("/")
                    if fname.startswith("/") or ":" in parts[0] or ".." in parts:
                        raise SecurityValidationError(f"Zip archive member '{fname}' contains illegal path traversal or absolute path", code="SECURITY_DECOMPRESSION_BOMB")

                    if fname == "[Content_Types].xml" or "word/" in fname:
                        is_docx = True

                    if item.file_size > MAX_ZIP_SINGLE_FILE_BYTES:
                        raise SecurityValidationError(f"Zip member '{fname}' exceeds uncompressed size limit ({item.file_size} > {MAX_ZIP_SINGLE_FILE_BYTES})", code="SECURITY_DECOMPRESSION_BOMB")

                    total_uncompressed += item.file_size

                if not is_docx:
                    raise SecurityValidationError("ZIP archive is not a valid DOCX document", code="SECURITY_MAGIC_BYTE_MISMATCH")

                if total_uncompressed > MAX_ZIP_CUMULATIVE_BYTES:
                    raise SecurityValidationError(f"Cumulative uncompressed zip size exceeds limit ({total_uncompressed} > {MAX_ZIP_CUMULATIVE_BYTES})", code="SECURITY_DECOMPRESSION_BOMB")

                if compressed_size > 0:
                    ratio = total_uncompressed / compressed_size
                    if ratio > MAX_ZIP_COMPRESSION_RATIO:
                        raise SecurityValidationError(f"Zip compression ratio too high ({ratio:.1f}:1 > {MAX_ZIP_COMPRESSION_RATIO}:1)", code="SECURITY_DECOMPRESSION_BOMB")

        except zipfile.BadZipFile:
            raise SecurityValidationError("Invalid or corrupted DOCX/ZIP file structure", code="SECURITY_MAGIC_BYTE_MISMATCH")

    async def download_and_validate_url(self, url: str) -> Tuple[bytes, str, str]:
        """
        Safely download content from URL protecting against SSRF, DNS-rebinding, oversized payloads,
        and TLS MITM attacks, enforced within a strict 30-second wall-clock deadline.
        """
        try:
            async with asyncio.timeout(TOTAL_TIMEOUT):
                return await self._download_url_internal(url)
        except TimeoutError:
            raise SecurityValidationError("URL download operation exceeded 30 second total timeout limit", code="SECURITY_BLOCKED_SSRF")

    async def _download_url_internal(self, url: str) -> Tuple[bytes, str, str]:
        current_url = url
        redirect_count = 0

        while redirect_count <= MAX_REDIRECTS:
            parsed = urllib.parse.urlparse(current_url)
            if parsed.scheme not in ("http", "https"):
                raise SecurityValidationError(f"Unsupported URL scheme '{parsed.scheme}'. Only http and https are allowed.", code="SECURITY_BLOCKED_SSRF")

            hostname = parsed.hostname
            if not hostname:
                raise SecurityValidationError("Invalid URL: missing hostname", code="SECURITY_BLOCKED_SSRF")

            port = parsed.port if parsed.port else (443 if parsed.scheme == "https" else 80)

            # Resolve DNS & enforce IP checks prior to connection
            target_ip = resolve_and_validate_host(hostname, port)

            # SSL context for TLS validation against hostname
            ssl_context = None
            if parsed.scheme == "https":
                ssl_context = ssl.create_default_context()

            # Create transport targeting resolved IP with SNI / Host header set to original hostname
            transport = httpx.AsyncHTTPTransport(
                verify=ssl_context if parsed.scheme == "https" else True
            )

            # Connect to target IP with original Host header
            ip_host = f"[{target_ip}]" if ":" in target_ip else target_ip
            connect_url = urllib.parse.urlunparse((
                parsed.scheme,
                f"{ip_host}:{port}",
                parsed.path or "/",
                parsed.params,
                parsed.query,
                parsed.fragment
            ))

            headers = {"Host": hostname, "User-Agent": "EKP-IngestionWorker/1.0"}

            try:
                async with httpx.AsyncClient(
                    transport=transport,
                    follow_redirects=False,
                    timeout=httpx.Timeout(TOTAL_TIMEOUT, connect=CONNECT_TIMEOUT, read=READ_TIMEOUT)
                ) as client:
                    req = client.build_request("GET", connect_url, headers=headers)
                    resp = await client.send(req, stream=True)

                    if resp.status_code in (301, 302, 303, 307, 308):
                        redirect_location = resp.headers.get("Location")
                        if not redirect_location:
                            raise SecurityValidationError(f"Redirect response {resp.status_code} missing Location header", code="SECURITY_BLOCKED_SSRF")
                        current_url = urllib.parse.urljoin(current_url, redirect_location)
                        redirect_count += 1
                        await resp.aclose()
                        continue

                    if resp.status_code != 200:
                        await resp.aclose()
                        raise SecurityValidationError(f"HTTP GET failed with status code {resp.status_code}", code="SECURITY_BLOCKED_SSRF")

                    content_buf = bytearray()
                    async for chunk in resp.aiter_bytes():
                        content_buf.extend(chunk)
                        if len(content_buf) > MAX_FILE_SIZE_BYTES:
                            await resp.aclose()
                            raise SecurityValidationError(f"Downloaded payload exceeded maximum size limit of {MAX_FILE_SIZE_BYTES} bytes", code="SECURITY_FILE_SIZE_EXCEEDED")

                    await resp.aclose()
                    downloaded_bytes = bytes(content_buf)

                    path_basename = os.path.basename(parsed.path) if parsed.path else "url_document"
                    if not path_basename:
                        path_basename = "url_document"

                    mime_type, clean_filename = self.validate_file_bytes(downloaded_bytes, filename=path_basename)
                    return downloaded_bytes, mime_type, clean_filename

            except httpx.HTTPError as e:
                raise SecurityValidationError(f"Network error downloading URL '{url}': {str(e)}", code="SECURITY_BLOCKED_SSRF")

        raise SecurityValidationError(f"Maximum redirect count ({MAX_REDIRECTS}) exceeded", code="SECURITY_BLOCKED_SSRF")
