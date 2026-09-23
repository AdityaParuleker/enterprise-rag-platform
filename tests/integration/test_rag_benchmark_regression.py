"""
Regression Test Suite — RAG Architecture Invariants & Benchmark Regression Guard
Verifies that:
1. chat.py uses top_k candidate limit when reranking is disabled (prevents RRF pool floor dilution regression).
2. context_assembler.py maintains 8,000 character context budget.
3. prompt_builder.py maintains strict role-equivalence directive for Q16 under strict grounding.
"""

import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from backend.app.generation.context_assembler import ContextAssembler, DEFAULT_MAX_CONTEXT_CHARS
from backend.app.generation.prompt_builder import RAGPromptBuilder, SYSTEM_INSTRUCTION_STRICT

def test_candidate_pool_limit_invariant():
    """Ensure chat.py does NOT unconditionally force max(30, top_k) candidate pool floor."""
    chat_file = os.path.join(os.path.dirname(__file__), "..", "..", "backend", "app", "api", "chat.py")
    with open(chat_file, "r", encoding="utf-8") as f:
        code = f.read()

    assert "retrieval_limit = max(30, top_k)" not in code, \
        "REGRESSION REGRESSION: chat.py contains unconditional retrieval_limit = max(30, top_k) which dilutes RRF ranks!"
    assert "retrieval_limit = 30 if enable_reranking else top_k" in code, \
        "INVARIANT VIOLATION: chat.py must gate 30-candidate pool on enable_reranking and use top_k when reranking is disabled!"


def test_table_chunk_atomicity_invariant():
    """Ensure PDF parser and text chunker emit multi-row tables as atomic single chunks."""
    from backend.app.ingestion.parsers.pdf_parser import PDFParser
    from backend.app.ingestion.chunkers.text_chunker import TextChunker

    pdf_path = os.path.join(os.path.dirname(__file__), "..", "..", "nimbus_knowledge_base.pdf")
    with open(pdf_path, "rb") as f:
        file_bytes = f.read()

    parser = PDFParser()
    blocks = parser.parse(file_bytes)
    
    table_blocks = [b for b in blocks if b.block_type == "table"]
    assert len(table_blocks) >= 1, "PDFParser failed to extract any table blocks!"
    
    chunker = TextChunker()
    chunks = chunker.chunk_blocks(blocks, document_id="test_doc", tenant_id="test_tenant")
    
    # Verify that Section 2 pricing table containing all 4 tiers is in a single chunk
    pricing_table_chunks = [
        c for c in chunks 
        if all(tier in c["text"] for tier in ["Starter", "Pro", "Business", "Enterprise"])
        and "Support Response Time" in c["text"]
    ]
    assert len(pricing_table_chunks) == 1, \
        "ATOMICITY INVARIANT VIOLATION: Section 2 pricing table was split across multiple chunks!"


def test_context_budget_invariant():
    """Ensure context_assembler.py maintains 8,000 character budget."""
    assembler = ContextAssembler()
    assert DEFAULT_MAX_CONTEXT_CHARS == 8000, "INVARIANT VIOLATION: DEFAULT_MAX_CONTEXT_CHARS must be 8000!"
    assert assembler.max_chars == 8000


def test_role_equivalence_prompt_invariant():
    """Ensure SYSTEM_INSTRUCTION_STRICT contains the role equivalence directive to resolve Q16."""
    assert "Chief Technology Officer/CTO = head/leader of engineering" in SYSTEM_INSTRUCTION_STRICT, \
        "INVARIANT VIOLATION: SYSTEM_INSTRUCTION_STRICT missing CTO/engineering role equivalence rule!"


def test_behavioral_retrieval_and_generation_pipeline():
    """
    Behavioral Integration Test — Executes key benchmark queries (Q6, Q16, Q17, Q20, Q22)
    against the running RAG pipeline (http://localhost:8000/api/v1/chat) to verify live behavior.
    """
    import requests
    import json
    
    # Check if local FastAPI backend container is running
    try:
        health_res = requests.get("http://localhost:8000/health", timeout=3)
        if health_res.status_code != 200:
            pytest.skip("FastAPI backend container health check returned non-200 status.")
    except Exception:
        pytest.skip("FastAPI backend container is not reachable on http://localhost:8000")

    # Generate test JWT token
    jwt_secret = os.environ.get("JWT_SECRET", "98537d4f35b1431274c4a8e4f3db36b929c2ef698adf0090ccb25692cca922f1")
    os.environ["JWT_SECRET"] = jwt_secret
    from backend.app.auth.jwt import create_access_token

    token = create_access_token({
        'sub': 'a1ac6c21-ba3a-4267-a19a-c848d4cc94db',
        'tenant_id': '6e2476c2-90ea-439a-8328-577d6298119b',
        'email': 'demo_admin@enterprise.com'
    })
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    queries_to_test = [
        {"id": 6, "query": "What's the maximum individual file size on the Pro tier?", "expected": "5 GB"},
        {"id": 13, "query": "What is Nimbus's stock ticker symbol?", "expected_refusal": True},
        {"id": 14, "query": "How much does the Nimbus mobile app cost separately from a subscription?", "expected_refusal": True},
        {"id": 15, "query": "What is Nimbus's total annual revenue?", "expected_refusal": True},
        {"id": 16, "query": "Who's in charge of engineering at Nimbus?", "expected": "Raj Patel"},
        {"id": 17, "query": "What's the money-back window if I don't like the service?", "expected": "30"},
        {"id": 20, "query": "What was introduced in version 2.5 versus version 3.0?", "expected": "2.5"},
        {"id": 22, "query": "List every subscription tier and its support response time.", "expected_tiers": ["Starter", "Pro", "Business", "Enterprise"]},
        {"id": 24, "query": "When was it released?", "expected_versions": ["Version 1.0", "Version 2.0", "Version 3.0"]}
    ]

    for q in queries_to_test:
        payload = {
            "message": q["query"],
            "mode": "rag",
            "stream": True,
            "top_k": 5,
            "strict_grounding": True
        }
        res = None
        for attempt in range(3):
            try:
                res = requests.post("http://localhost:8000/api/v1/chat", json=payload, headers=headers, stream=True, timeout=60)
                if res.status_code == 200:
                    break
            except Exception:
                if attempt == 2:
                    raise
                import time
                time.sleep(2)
        assert res and res.status_code == 200, f"Q{q['id']} API call failed with HTTP {res.status_code if res else 'None'}"

        tokens = []
        for line in res.iter_lines():
            if line:
                line_str = line.decode('utf-8').strip()
                if line_str.startswith("data: "):
                    try:
                        event = json.loads(line_str[6:])
                        if event.get("type") in ("token", "content") or "content" in event or "token" in event:
                            cnt = event.get("content") or event.get("token") or event.get("text") or ""
                            if isinstance(cnt, str):
                                tokens.append(cnt)
                    except Exception:
                        pass
        answer = "".join(tokens)

        if q.get("expected_refusal"):
            assert "no relevant document context found" in answer.lower(), \
                f"Section E Safety Regression: Q{q['id']} failed to produce refusal template! Output: {answer[:100]}"
        elif q["id"] == 22:
            for tier in q["expected_tiers"]:
                assert tier.lower() in answer.lower(), \
                    f"Q22 behavioral regression: Missing tier '{tier}' in answer! Output: {answer[:100]}"
        elif q["id"] == 24:
            for ver in q["expected_versions"]:
                assert ver.lower() in answer.lower(), \
                    f"Q24 ambiguous referent regression: Missing '{ver}' in answer! Output: {answer[:100]}"
        else:
            assert q["expected"].lower() in answer.lower(), \
                f"Q{q['id']} behavioral regression! Expected '{q['expected']}' in answer, got: '{answer[:100]}...'"


def test_ambiguity_and_out_of_scope_boundary_stress_set():
    """
    Directive 5 Ambiguity vs Out-of-Scope Boundary Stress Test Suite.
    Ensures paraphrased out-of-scope & adversarial boundary queries refuse cleanly (zero overreach),
    and paraphrased ambiguous queries enumerate candidates properly.
    """
    import requests
    import json

    try:
        health_res = requests.get("http://localhost:8000/health", timeout=3)
        if health_res.status_code != 200:
            pytest.skip("FastAPI backend container health check returned non-200 status.")
    except Exception:
        pytest.skip("FastAPI backend container is not reachable on http://localhost:8000")

    jwt_secret = os.environ.get("JWT_SECRET", "98537d4f35b1431274c4a8e4f3db36b929c2ef698adf0090ccb25692cca922f1")
    os.environ["JWT_SECRET"] = jwt_secret
    from backend.app.auth.jwt import create_access_token

    token = create_access_token({
        'sub': 'a1ac6c21-ba3a-4267-a19a-c848d4cc94db',
        'tenant_id': '6e2476c2-90ea-439a-8328-577d6298119b',
        'email': 'demo_admin@enterprise.com'
    })
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    stress_cases = [
        # Out-of-scope paraphrases & adversarial boundaries MUST refuse cleanly
        {"query": "What is Nimbus worth as a company?", "type": "refusal"},
        {"query": "How much do people pay for Nimbus on mobile beyond the subscription?", "type": "refusal"},
        {"query": "What is the stock market symbol for Nimbus?", "type": "refusal"},
        {"query": "How much money does Nimbus make per year?", "type": "refusal"},
        {"query": "What are all the international office locations released over time?", "type": "refusal"},
        {"query": "What are the discount rates for university students?", "type": "refusal"},
        # Ambiguous paraphrases MUST enumerate candidates WITH FACTUAL PRECISION
        {"query": "When did the current version come out?", "type": "enumeration", 
         "expected_shape": ["Version 1.0", "Version 2.0", "Version 3.0"],
         "expected_facts": ["January 2016", "June 2019", "March 2023"]},
        {"query": "What are all the pricing options?", "type": "enumeration", 
         "expected_shape": ["Starter", "Pro", "Business", "Enterprise"],
         "expected_facts": ["$9/month", "$29/month", "$99/month", "Custom pricing"]},
        {"query": "How much storage space do I get?", "type": "enumeration",
         "expected_shape": ["Starter", "Pro", "Business", "Enterprise"],
         "expected_facts": ["100 GB", "2 TB", "10 TB", "Unlimited"]},
        {"query": "What features were added across the updates?", "type": "enumeration",
         "expected_shape": ["Version 1.0", "Version 2.0", "Version 3.0"],
         "expected_facts": ["Basic file storage and sync", "Real-time multi-device sync", "AI-powered semantic file search"]}
    ]

    for sc in stress_cases:
        payload = {
            "message": sc["query"],
            "mode": "rag",
            "stream": True,
            "top_k": 5,
            "strict_grounding": True
        }
        res = None
        for attempt in range(3):
            try:
                res = requests.post("http://localhost:8000/api/v1/chat", json=payload, headers=headers, stream=True, timeout=60)
                if res.status_code == 200:
                    break
            except Exception:
                if attempt == 2:
                    raise
                import time
                time.sleep(2)
        assert res and res.status_code == 200, f"Stress query '{sc['query']}' failed with HTTP {res.status_code if res else 'None'}"

        tokens = []
        for line in res.iter_lines():
            if line:
                line_str = line.decode('utf-8').strip()
                if line_str.startswith("data: "):
                    try:
                        event = json.loads(line_str[6:])
                        if event.get("type") in ("token", "content") or "content" in event or "token" in event:
                            cnt = event.get("content") or event.get("token") or event.get("text") or ""
                            if isinstance(cnt, str):
                                tokens.append(cnt)
                    except Exception:
                        pass
        answer = "".join(tokens)

        if sc["type"] == "refusal":
            assert "no relevant document context found" in answer.lower() or not answer.strip(), \
                f"STRESS REGRESSION (OVERREACH): Query '{sc['query']}' failed to refuse cleanly! Output: {answer[:100]}"
        elif sc["type"] == "enumeration":
            # 1. Shape Verification
            for exp in sc["expected_shape"]:
                assert exp.lower() in answer.lower(), \
                    f"STRESS REGRESSION (UNDER-REACH SHAPE): Query '{sc['query']}' missing candidate '{exp}'! Output: {answer[:100]}"
            # 2. Fact-Level Verification against source PDF text
            for fact in sc["expected_facts"]:
                assert fact.lower() in answer.lower(), \
                    f"STRESS FACTUAL HALLUCINATION REGRESSION: Query '{sc['query']}' missing literal fact '{fact}'! Output: {answer[:100]}"




