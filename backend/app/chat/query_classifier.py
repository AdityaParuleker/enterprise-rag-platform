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


CLASSIFIER_SYSTEM_PROMPT = """You are an intent classifier for an enterprise assistant.

Classify the LATEST user message as CONVERSATIONAL or TECHNICAL, using the conversation history for context.

CONVERSATIONAL includes greetings, personal statements/opinions, identity sharing, thanks, and casual remarks that do not request information from documents.

TECHNICAL includes any request for facts, specs, procedures, or document-grounded information.

Return exactly one word:
YES
or
NO

YES means the latest user message is CONVERSATIONAL.
NO means the latest user message is TECHNICAL.

If a message contains both a casual remark and a substantive request for enterprise document information, return NO.

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
    "who are you", "who are you?", "who are u", "who are u?",
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
    r"^(how\s+are\s+you|how\s+is\s+it\s+going|how's\s+it\s+going|hows\s+it\s+going|what's\s+up|whats\s+up|nice\s+to\s+meet\s+you|hope\s+(you're|youre|you\s+are)\s+doing\s+well|can\s+you\s+help\s+me|have\s+a\s+(nice|great)\s+day|what\s+is\s+your\s+name|what's\s+your\s+name|whats\s+your\s+name|can\s+you\s+tell\s+me\s+your\s+name|who\s+are\s+you|(and\s+)?whats?\s+your\s+fav(orite|ourite)\s+[\w\s]{1,40})$",
    re.IGNORECASE
)

# Conservative name intro pattern: strictly 1 to 2 alphabetic name tokens (e.g., "Sarah", "Alex Smith")
NAME_INTRO_PATTERN = re.compile(
    r"^(i\s+am|i'm|im|my\s+name\s+is|this\s+is)\s+[a-zA-Z]{1,25}(\s+[a-zA-Z]{1,25})?$",
    re.IGNORECASE
)

# Casual preference and personal statement pre-filter pattern (e.g., "I like soccer", "I love coffee")
CASUAL_PREFERENCE_PATTERN = re.compile(
    r"^(i\s+(like|love|enjoy|prefer|dislike|hate|play|watch|am\s+a\s+fan\s+of)\s+[\w\s]{1,40}|my\s+(favorite|favourite)\s+[\w\s]{1,40}\s+is\s+[\w\s]{1,40}|i\s+(live|work)\s+(in|at)\s+[\w\s]{1,40}|and\s+what's\s+your\s+favorite\s+[\w\s]{1,40})$",
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
    4. Cheap Rule-Based Pre-Filter (Fast-Path for greetings, intros, casual preferences, thanks)
    5. Context-Aware LLM-Based Intent Classification (passing last 3-4 turns) with strict YES/NO parsing
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
        provider_name = self.llm_client.provider.__class__.__name__.lower()
        is_gemini = "gemini" in provider_name or bool(os.getenv("GEMINI_API_KEY"))
        default_model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite") if is_gemini else "qwen3:0.6b"
        default_timeout = "6.0" if is_gemini else "4.0"

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
        if CASUAL_PREFERENCE_PATTERN.match(c_clean):
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
                or CASUAL_PREFERENCE_PATTERN.match(remainder)
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
        context_turns_count = len(history_messages) if history_messages else 0

        if not clean_query:
            logger.info(
                f"QueryClassifier Decision | query='' | context_turns={context_turns_count} | "
                f"result=conversational | path=pre-filter:empty-input | detail=Empty query string"
            )
            return QueryType.CONVERSATIONAL

        normalized_query = clean_query.lower()

        # STEP 2: Substantive Domain Intent Safety Check
        for kw in SUBSTANTIVE_DOMAIN_KEYWORDS:
            if re.search(r"\b" + re.escape(kw) + r"\b", normalized_query):
                logger.info(
                    f"QueryClassifier Decision | query={repr(clean_query)} | context_turns={context_turns_count} | "
                    f"result=domain | path=pre-filter:domain-keyword | detail=Matched substantive keyword '{kw}'"
                )
                return QueryType.DOMAIN

        for pattern in DOMAIN_INTENT_PATTERNS:
            if re.search(pattern, normalized_query):
                logger.info(
                    f"QueryClassifier Decision | query={repr(clean_query)} | context_turns={context_turns_count} | "
                    f"result=domain | path=pre-filter:domain-intent-pattern | detail=Matched domain intent pattern"
                )
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
                    logger.info(
                        f"QueryClassifier Decision | query={repr(clean_query)} | context_turns={context_turns_count} | "
                        f"result=conversational | path=history:acknowledgement | detail=Post-domain response acknowledgement"
                    )
                    return QueryType.CONVERSATIONAL

                followup_triggers = ["what about", "and what about", "can you explain", "explain that", "more details", "how about", "tell me more"]
                if any(tr in normalized_query for tr in followup_triggers):
                    logger.info(
                        f"QueryClassifier Decision | query={repr(clean_query)} | context_turns={context_turns_count} | "
                        f"result=domain | path=history:followup | detail=Contextual domain follow-up query"
                    )
                    return QueryType.DOMAIN

        # STEP 4: Cheap Rule-Based Conversational Pre-Filter (Fast-Path)
        if self.use_fast_path and self._is_purely_conversational(clean_query):
            logger.info(
                f"QueryClassifier Decision | query={repr(clean_query)} | context_turns={context_turns_count} | "
                f"result=conversational | path=pre-filter:rule-based | detail=Matched conversational rule-based pattern"
            )
            return QueryType.CONVERSATIONAL

        # STEP 5: Context-Aware LLM-Based Intent Classification (passing last 3-4 turns / up to 6 messages)
        if self.enabled:
            start_time = time.time()
            try:
                history_snippet = ""
                if history_messages:
                    # Pass last 3-4 turns (up to 6 messages: user & assistant turns)
                    recent_messages = history_messages[-6:]
                    formatted_turns = []
                    for m in recent_messages:
                        r = "User" if (m.get("role") == "user" or m.get("sender") == "user") else "Assistant"
                        formatted_turns.append(f"{r}: {m.get('content', '')[:200]}")
                    history_snippet = "\n".join(formatted_turns)

                if history_snippet:
                    prompt = f"{CLASSIFIER_SYSTEM_PROMPT}\n\nRECENT CONVERSATION HISTORY:\n{history_snippet}\n\nLATEST USER MESSAGE:\n{clean_query}"
                else:
                    prompt = f"{CLASSIFIER_SYSTEM_PROMPT}\n\nLATEST USER MESSAGE:\n{clean_query}"

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
                        f"QueryClassifier Decision | query={repr(clean_query)} | context_turns={context_turns_count} | "
                        f"result={parsed_res.value} | path=llm-classifier | "
                        f"detail=LLM output='{llm_response.strip()}' model='{self.model}' latency={latency:.3f}s"
                    )
                    return parsed_res
                else:
                    logger.warning(
                        f"QueryClassifier Decision | query={repr(clean_query)} | context_turns={context_turns_count} | "
                        f"result=domain | path=fallback:domain-default | "
                        f"detail=Strict binary parser rejected output '{llm_response.strip()}'. latency={latency:.3f}s"
                    )
                    return QueryType.DOMAIN
            except Exception as e:
                latency = time.time() - start_time
                logger.warning(
                    f"QueryClassifier Decision | query={repr(clean_query)} | context_turns={context_turns_count} | "
                    f"result=domain | path=fallback:domain-default | "
                    f"detail=LLM classification failed or timed out ({e}). latency={latency:.3f}s"
                )
                return QueryType.DOMAIN

        # STEP 6: Fail-Safe Default to DOMAIN
        logger.info(
            f"QueryClassifier Decision | query={repr(clean_query)} | context_turns={context_turns_count} | "
            f"result=domain | path=fallback:disabled-default | detail=Classifier disabled or unhandled"
        )
        return QueryType.DOMAIN

