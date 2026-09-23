"""
Conversational Query Classifier Module
Classifies incoming chat queries into CONVERSATIONAL or DOMAIN categories
prior to database retrieval, vector search, or ACL filtering.
Supports local open-source LLM classification (Qwen3-0.6B) with strict binary YES/NO parsing.
"""

import os
import re
import time
import logging
from enum import Enum
from typing import List, Dict, Any, Optional
from backend.app.generation.llm_client import LLMClient

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    CONVERSATIONAL = "conversational"
    DOMAIN = "domain"


CLASSIFIER_SYSTEM_PROMPT = """You are a binary intent classifier for an enterprise assistant.

Determine whether the user's message is PURELY CONVERSATIONAL.

Return exactly one word:
YES
or
NO

YES means the message is purely social/conversational, such as:
greetings, small talk, chit-chat, self-introductions, pleasantries,
thank-you messages, acknowledgements, or farewells.

NO means the message is a technical, enterprise, domain, documentation,
knowledge-base, configuration, policy, or information request, or anything
that may require retrieving information from the knowledge base.

If a message contains both a conversational greeting and a substantive
domain question, return NO.

Do NOT include thinking tags (<think>...</think>), reasoning, or explanations.
Return ONLY YES or NO."""


def parse_binary_response(raw_response: str) -> Optional[QueryType]:
    """
    Strict binary output parser for classifier LLM responses.
    Strips reasoning tags (<think>...</think>) if present, then checks for exact 'YES' or 'NO'.
    Returns QueryType.CONVERSATIONAL for 'YES', QueryType.DOMAIN for 'NO',
    and None for any malformed/unexpected output (which triggers fail-closed fallback to DOMAIN).
    """
    if not raw_response or not isinstance(raw_response, str):
        return None

    # Strip out full or unclosed <think>...</think> blocks (case-insensitive, DOTALL)
    cleaned = re.sub(r"<think>.*?(?:</think>|$)", "", raw_response, flags=re.DOTALL | re.IGNORECASE).strip().upper()

    if cleaned == "YES":
        return QueryType.CONVERSATIONAL
    elif cleaned == "NO":
        return QueryType.DOMAIN
    else:
        return None


STANDALONE_CONVERSATIONAL_PHRASES = {
    "hi", "hello", "hey", "hey there", "hi there", "good morning", "good afternoon", "good evening",
    "how are you", "how are you?", "how is it going", "how's it going", "how's it going?",
    "what's up", "what's up?", "nice to meet you", "thank you", "thanks", "thanks a lot",
    "you're welcome", "you welcome", "okay", "ok", "got it", "great", "perfect", "sounds good",
    "bye", "goodbye", "see you later", "have a nice day", "can you help me", "can you help me?",
    "hello everyone", "hey everyone", "greetings", "what is your name", "what's your name",
    "what is your name?", "what's your name?", "can you tell me your name", "can you tell me your name?",
    "cool", "nice", "alright", "sure", "yep", "yeah", "no problem", "np", "fine", "good", "awesome", "sweet"
}

SUBSTANTIVE_DOMAIN_KEYWORDS = [
    "policy", "procedure", "auth", "authentication", "login", "password", "backup",
    "upload", "file", "document", "pdf", "docx", "leave", "vacation", "system",
    "requirement", "workflow", "process", "onboarding", "approval", "limit", "size",
    "architecture", "schema", "table", "api", "endpoint", "database", "pgvector",
    "config", "configuration", "setting", "rule", "permission", "role", "acl",
    "handbook", "component", "failover", "aurawave", "model", "llm", "rag", "eval"
]

DOMAIN_INTENT_PATTERNS = [
    r"\b(i\s+am|i'm|im)\s+(looking\s+for|trying\s+to|asking\s+about|searching\s+for|wondering\s+about|need\s+to\s+know)\b",
    r"\b(can\s+you|could\s+you|please)\s+(explain|find|show|provide|describe)\b\s+(the|a|an|how|our|my|enterprise|system|policy|process|workflow|table|schema|database|config)",
    r"\b(how\s+do\s+i|how\s+can\s+i|how\s+to)\b\s+(configure|setup|use|upload|access|login|reset|authenticate|create|delete|update|query)"
]

GREETING_WORDS = r"(?:hi|hello|hey|greetings|howdy|good\s+(?:morning|afternoon|evening|day))"
TARGET_WORDS = r"(?:there|everyone|all|team|friends)"
GREETING_PATTERN = re.compile(rf"^{GREETING_WORDS}(?:\s+{TARGET_WORDS})?$", re.IGNORECASE)

PLEASANTRY_PATTERN = re.compile(
    r"^(how\s+are\s+you|how\s+is\s+it\s+going|how's\s+it\s+going|hows\s+it\s+going|what's\s+up|whats\s+up|nice\s+to\s+meet\s+you|hope\s+(you're|youre|you\s+are)\s+doing\s+well|can\s+you\s+help\s+me|have\s+a\s+(nice|great)\s+day|what\s+is\s+your\s+name|what's\s+your\s+name|whats\s+your\s+name|can\s+you\s+tell\s+me\s+your\s+name)$",
    re.IGNORECASE
)

# Conservative name intro pattern: strictly 1 to 2 alphabetic name tokens (e.g., "Sarah", "Alex Smith")
# Excludes action/verb phrases like "looking for", "trying to", "asking about"
NAME_INTRO_PATTERN = re.compile(
    r"^(i\s+am|i'm|im|my\s+name\s+is|this\s+is)\s+[a-zA-Z]{1,25}(\s+[a-zA-Z]{1,25})?$",
    re.IGNORECASE
)

FAREWELL_THANKS_PATTERN = re.compile(
    r"^(thank\s+you|thanks(\s+a\s+lot)?|you're\s+welcome|youre\s+welcome|you\s+welcome|ok|okay|got\s+it|great|perfect|sounds\s+good|cool|nice|alright|sure|yep|yeah|no\s+problem|np|fine|good|awesome|sweet|bye|goodbye|see\s+you(\s+later)?)$",
    re.IGNORECASE
)

MEMORY_RECALL_PATTERNS = [
    # User identity & name recall (e.g. "what is my name", "hey what's my name again", "who am i", "do you remember my name")
    re.compile(r"\b(?:what\s+is|what's|whats|what\s+was)\s+my\s+name\b", re.IGNORECASE),
    re.compile(r"\b(?:do\s+you\s+know|do\s+you\s+remember|tell\s+me|can\s+you\s+tell\s+me)\s+my\s+name\b", re.IGNORECASE),
    re.compile(r"\bwho\s+am\s+i\b", re.IGNORECASE),
    re.compile(r"\bdo\s+you\s+(?:know|remember)\s+(?:who\s+i\s+am|me)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+did\s+i\s+(?:say|tell\s+you)\s+my\s+name\s+(?:is|was)\b", re.IGNORECASE),

    # User prior statement recall (e.g. "what did I just ask", "what did I say earlier", "what was my question")
    re.compile(r"\b(?:what\s+did\s+i|what\s+was\s+my)\s+(?:just\s+|last\s+|previous\s+|prior\s+)?(?:ask|say|tell\s+you|question|message)\b", re.IGNORECASE),

    # Assistant prior answer recall (e.g. "what did you say earlier", "what was your last answer")
    re.compile(r"\b(?:what\s+did\s+you|what\s+was\s+your)\s+(?:just\s+|last\s+|previous\s+|prior\s+)?(?:say|tell\s+me|answer|reply|response)\b", re.IGNORECASE)
]


class QueryClassifier:
    """
    Classifies incoming user queries into CONVERSATIONAL or DOMAIN intent.
    Uses a multi-stage priority approach:
    1. Input Normalization
    2. Substantive Domain Intent Safety Check (Force DOMAIN if query requests enterprise/technical info)
    3. History-Aware Contextual Check (Follow-ups vs Acknowledgements)
    4. Conversational Greeting, Pleasantry & Conservative Self-Introduction Fast-Path
    5. Local LLM-Based Binary Intent Classification (Qwen3-0.6B) with strict YES/NO parsing
    6. Fail-Safe Default to DOMAIN
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        enabled: Optional[bool] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout_seconds: Optional[float] = None,
        use_fast_path: Optional[bool] = None
    ):
        self.llm_client = llm_client or LLMClient()
        self.enabled = (
            enabled
            if enabled is not None
            else os.getenv("QUERY_CLASSIFIER_ENABLED", "true").lower() in ("true", "1", "yes")
        )
        provider = os.getenv("LLM_PROVIDER", "ollama").lower()
        default_model = os.getenv("GEMINI_MODEL", "gemini-3.6-flash") if provider in ("gemini", "google") else "qwen3:0.6b"
        default_timeout = "6.0" if provider in ("gemini", "google") else "4.0"

        self.model = model or os.getenv("QUERY_CLASSIFIER_MODEL", default_model)
        self.max_tokens = (
            max_tokens
            if max_tokens is not None
            else int(os.getenv("QUERY_CLASSIFIER_MAX_TOKENS", "40"))
        )
        self.temperature = (
            temperature
            if temperature is not None
            else float(os.getenv("QUERY_CLASSIFIER_TEMPERATURE", "0"))
        )
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else float(os.getenv("QUERY_CLASSIFIER_TIMEOUT_SECONDS", default_timeout))
        )
        self.use_fast_path = (
            use_fast_path
            if use_fast_path is not None
            else os.getenv("QUERY_CLASSIFIER_USE_FAST_PATH", "true").lower() in ("true", "1", "yes")
        )

    def _is_conversational_clause(self, clause: str) -> bool:
        c = clause.strip()
        if not c:
            return True
        c_clean = re.sub(r"[^\w\s]", "", c.lower()).strip()
        if c_clean in STANDALONE_CONVERSATIONAL_PHRASES:
            return True
        if GREETING_PATTERN.match(c_clean):
            return True
        if PLEASANTRY_PATTERN.match(c_clean):
            return True
        if NAME_INTRO_PATTERN.match(c_clean):
            return True
        if FAREWELL_THANKS_PATTERN.match(c_clean):
            return True

        for pattern in MEMORY_RECALL_PATTERNS:
            if pattern.search(clause) or pattern.search(c_clean):
                return True

        # Check compound greeting + intro / pleasantry in single clause (e.g. "hi i am sarah")
        leading_greeting_match = re.match(rf"^{GREETING_WORDS}(?:\s+{TARGET_WORDS})?\s+(.+)$", c_clean, re.IGNORECASE)
        if leading_greeting_match:
            remainder = leading_greeting_match.group(1).strip()
            if (
                remainder in STANDALONE_CONVERSATIONAL_PHRASES
                or PLEASANTRY_PATTERN.match(remainder)
                or NAME_INTRO_PATTERN.match(remainder)
                or FAREWELL_THANKS_PATTERN.match(remainder)
            ):
                return True

        return False

    def _is_purely_conversational(self, clean_query: str) -> bool:
        normalized_query = clean_query.lower().strip()
        stripped_query = re.sub(r"[^\w\s]", "", normalized_query).strip()

        if stripped_query in STANDALONE_CONVERSATIONAL_PHRASES:
            return True

        # Split query into clauses by punctuation (. , ! ? ;)
        clauses = [part.strip() for part in re.split(r"[,.!?;\n]+", clean_query) if part.strip()]
        if not clauses:
            return True

        return all(self._is_conversational_clause(clause) for clause in clauses)

    async def classify_query(
        self,
        query: str,
        history_messages: Optional[List[Dict[str, Any]]] = None
    ) -> QueryType:
        # STEP 1: Input Normalization
        clean_query = query.strip() if query else ""
        if not clean_query:
            return QueryType.CONVERSATIONAL

        normalized_query = clean_query.lower()

        # STEP 2: Substantive Domain Intent Safety Check
        # Check explicit domain keywords
        for kw in SUBSTANTIVE_DOMAIN_KEYWORDS:
            if re.search(r"\b" + re.escape(kw) + r"\b", normalized_query):
                logger.info(f"QueryClassifier: Matched domain keyword '{kw}' -> DOMAIN")
                return QueryType.DOMAIN

        # Check explicit domain intent action/verb patterns
        for pattern in DOMAIN_INTENT_PATTERNS:
            if re.search(pattern, normalized_query):
                logger.info(f"QueryClassifier: Matched domain intent pattern -> DOMAIN")
                return QueryType.DOMAIN

        # STEP 3: History-Aware Contextual Check
        if history_messages and len(history_messages) >= 2:
            last_assistant_msg = None
            for msg in reversed(history_messages):
                role = msg.get("role") or (msg.get("sender") if isinstance(msg, dict) else None)
                if role == "assistant":
                    last_assistant_msg = msg
                    break

            if last_assistant_msg:
                stripped_query = re.sub(r"[^\w\s]", "", normalized_query).strip()
                acknowledgement_phrases = {
                    "thanks", "thank you", "okay", "ok", "got it", "great", "perfect", "sounds good",
                    "cool", "nice", "alright", "sure", "yep", "yeah", "no problem", "np", "fine", "good", "awesome", "sweet"
                }
                if stripped_query in acknowledgement_phrases:
                    logger.info("QueryClassifier: Post-domain response acknowledgement -> CONVERSATIONAL")
                    return QueryType.CONVERSATIONAL

                followup_triggers = ["what about", "and what about", "can you explain", "explain that", "more details", "how about", "tell me more"]
                if any(tr in normalized_query for tr in followup_triggers):
                    logger.info("QueryClassifier: Contextual domain follow-up query -> DOMAIN")
                    return QueryType.DOMAIN

        # STEP 4: Conversational Greeting, Pleasantry & Conservative Self-Introduction Fast-Path
        if self.use_fast_path and self._is_purely_conversational(clean_query):
            logger.info("QueryClassifier: Recognized pure conversational fast-path -> CONVERSATIONAL")
            return QueryType.CONVERSATIONAL

        # STEP 5: Local LLM-Based Binary Intent Classification (Qwen3-0.6B)
        if self.enabled:
            start_time = time.time()
            try:
                prompt = f"{CLASSIFIER_SYSTEM_PROMPT}\n\nUSER MESSAGE:\n{clean_query}"
                if history_messages:
                    context_snippet = "\n".join(
                        f"{m.get('role', 'user')}: {m.get('content', '')[:200]}"
                        for m in history_messages[-2:]
                    )
                    prompt = f"{CLASSIFIER_SYSTEM_PROMPT}\n\nRECENT CONVERSATION CONTEXT:\n{context_snippet}\n\nUSER MESSAGE:\n{clean_query}"

                llm_response = self.llm_client.generate_response(
                    prompt=prompt,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout_seconds
                )
                latency = time.time() - start_time
                parsed_res = parse_binary_response(llm_response)

                if parsed_res is not None:
                    logger.info(
                        f"QueryClassifier: Local LLM model='{self.model}' output='{llm_response.strip()}' "
                        f"parsed='{parsed_res.value}' latency={latency:.3f}s"
                    )
                    return parsed_res
                else:
                    logger.warning(
                        f"QueryClassifier: Strict binary parser rejected output '{llm_response.strip()}'. "
                        f"Failing closed to DOMAIN. latency={latency:.3f}s"
                    )
                    return QueryType.DOMAIN
            except Exception as e:
                latency = time.time() - start_time
                logger.warning(
                    f"QueryClassifier: Local LLM classification failed or timed out ({e}). "
                    f"Failing closed to DOMAIN fallback. latency={latency:.3f}s"
                )
                return QueryType.DOMAIN

        # STEP 6: Fail-Safe Default to DOMAIN
        return QueryType.DOMAIN
