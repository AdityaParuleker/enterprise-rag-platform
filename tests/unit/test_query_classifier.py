"""
Unit Tests for Conversational Query Classifier (Query Routing Before Retrieval)
"""

import pytest
from unittest.mock import MagicMock
from backend.app.chat.query_classifier import QueryClassifier, QueryType


@pytest.fixture
def mock_llm_client():
    client = MagicMock()
    # Default mock LLM output to DOMAIN for safety
    client.generate_response.return_value = "DOMAIN"
    return client


@pytest.fixture
def classifier(mock_llm_client):
    return QueryClassifier(llm_client=mock_llm_client)


@pytest.mark.asyncio
async def test_pure_conversational_greetings(classifier):
    conversational_queries = [
        "Hi",
        "Hello",
        "Hey",
        "Hi there",
        "Hi I am Sarah",
        "Hi, I am Sarah",
        "Hello, my name is Sarah",
        "Hey, I'm Sarah",
        "Hi, my name is John",
        "Nice to meet you",
        "Hello everyone",
        "Good morning, I'm Sarah",
        "Hey, how are you?",
        "Hi, hope you're doing well",
        "My name is Sarah",
        "Good morning",
        "Good afternoon",
        "Good evening",
        "How's it going?",
        "What's up?",
        "Thank you",
        "Thanks",
        "Thanks a lot",
        "You're welcome",
        "Okay",
        "Ok",
        "Got it",
        "Great",
        "Perfect",
        "Sounds good",
        "Bye",
        "Goodbye",
        "See you later",
        "Have a nice day",
        "Can you help me?",
        "hi i am john",
        "hello my name is alex",
        "What is your name?",
        "Can you tell me your name?"
    ]
    for q in conversational_queries:
        res = await classifier.classify_query(q)
        assert res == QueryType.CONVERSATIONAL, f"Expected CONVERSATIONAL for query: '{q}', got {res}"


@pytest.mark.asyncio
async def test_substantive_domain_queries(classifier):
    domain_queries = [
        "What is the leave policy?",
        "Explain the authentication process.",
        "What is the maximum upload size?",
        "How does the backup process work?",
        "What documents are required?",
        "What are the system requirements?",
        "Explain the approval workflow.",
        "How long does the onboarding process take?",
        "I am looking for the leave policy",
        "I am trying to understand authentication",
        "I am asking about the backup process"
    ]
    for q in domain_queries:
        res = await classifier.classify_query(q)
        assert res == QueryType.DOMAIN, f"Expected DOMAIN for query: '{q}', got {res}"


@pytest.mark.asyncio
async def test_mixed_greeting_and_domain_queries(classifier):
    mixed_queries = [
        "Hi, what is the leave policy?",
        "Hello, can you explain the authentication process?",
        "Good morning, what is the maximum upload size?",
        "Hey, how does the backup process work?",
        "Can you help me configure authentication?",
        "Thanks, but can you explain the approval workflow?",
        "Hi, I am Sarah. What is the leave policy?",
        "Hello, I'm Sarah. Can you explain authentication?",
        "Hey, my name is Sarah. What is the maximum upload size?",
        "Hi, can you tell me how the backup process works?"
    ]
    for q in mixed_queries:
        res = await classifier.classify_query(q)
        assert res == QueryType.DOMAIN, f"Expected DOMAIN for mixed query: '{q}', got {res}"


@pytest.mark.asyncio
async def test_paired_intent_conversational_vs_domain(classifier):
    """
    Verifies paired intent examples to ensure semantic understanding rather than string matching.
    """
    # Pair 1: Greeting + Intro vs Greeting + Intro + Substantive Question
    assert await classifier.classify_query("Hi I am Sarah") == QueryType.CONVERSATIONAL
    assert await classifier.classify_query("Hi I am Sarah. Can you tell me about the leave policy?") == QueryType.DOMAIN

    # Pair 2: Direct Intro vs Direct Intro + Substantive Question
    assert await classifier.classify_query("My name is Sarah") == QueryType.CONVERSATIONAL
    assert await classifier.classify_query("My name is Sarah. How does authentication work?") == QueryType.DOMAIN


@pytest.mark.asyncio
async def test_context_dependent_followup_domain_queries(classifier):
    history = [
        {"role": "user", "content": "What is the maximum upload size?"},
        {"role": "assistant", "content": "The maximum document upload size is 20 MB per file."}
    ]

    # Contextual follow-up must be classified as DOMAIN
    followup_query = "Okay, and what about images?"
    res = await classifier.classify_query(followup_query, history_messages=history)
    assert res == QueryType.DOMAIN, f"Expected DOMAIN for follow-up query, got {res}"

    explain_simpler = "Can you explain that in simpler terms?"
    res_explain = await classifier.classify_query(explain_simpler, history_messages=history)
    assert res_explain == QueryType.DOMAIN, f"Expected DOMAIN for explanation request, got {res_explain}"


@pytest.mark.asyncio
async def test_acknowledgment_after_domain_answer(classifier):
    history = [
        {"role": "user", "content": "What is the maximum upload size?"},
        {"role": "assistant", "content": "The maximum document upload size is 20 MB per file."}
    ]

    ack_query = "Thanks!"
    res = await classifier.classify_query(ack_query, history_messages=history)
    assert res == QueryType.CONVERSATIONAL, f"Expected CONVERSATIONAL for acknowledgement, got {res}"


from backend.app.chat.query_classifier import QueryClassifier, QueryType, parse_binary_response


def test_parse_binary_response():
    # Valid binary inputs
    assert parse_binary_response("YES") == QueryType.CONVERSATIONAL
    assert parse_binary_response("yes") == QueryType.CONVERSATIONAL
    assert parse_binary_response("  YES \n") == QueryType.CONVERSATIONAL

    assert parse_binary_response("NO") == QueryType.DOMAIN
    assert parse_binary_response("no") == QueryType.DOMAIN
    assert parse_binary_response(" NO ") == QueryType.DOMAIN

    # Thinking tags handling (Qwen3) -> strip <think>...</think> blocks and parse remainder
    assert parse_binary_response("<think>the user is greeting me</think>\nYES") == QueryType.CONVERSATIONAL
    assert parse_binary_response("<think>this needs docs</think>\nNO") == QueryType.DOMAIN
    assert parse_binary_response("<think>unclear...\n") is None

    # Strict parser fail-closed invalid/unexpected outputs -> None (triggers DOMAIN fallback)
    assert parse_binary_response("YES.") is None
    assert parse_binary_response("NO.") is None
    assert parse_binary_response("Yes, this is conversational.") is None
    assert parse_binary_response("I think YES") is None
    assert parse_binary_response("CONVERSATIONAL") is None
    assert parse_binary_response("DOMAIN") is None
    assert parse_binary_response("maybe") is None
    assert parse_binary_response("") is None
    assert parse_binary_response(None) is None


@pytest.mark.asyncio
async def test_memory_recall_casual_variations_and_overlapping_domain(classifier):
    """
    Verifies memory recall queries (exact & casual variations) route to CONVERSATIONAL,
    while domain queries with overlapping words route to DOMAIN.
    """
    # Exact memory recall queries
    assert await classifier.classify_query("what is my name") == QueryType.CONVERSATIONAL
    assert await classifier.classify_query("who am i") == QueryType.CONVERSATIONAL
    assert await classifier.classify_query("do you remember my name") == QueryType.CONVERSATIONAL

    # Casual variations with leading/trailing words
    assert await classifier.classify_query("hey what's my name") == QueryType.CONVERSATIONAL
    assert await classifier.classify_query("what is my name again?") == QueryType.CONVERSATIONAL
    assert await classifier.classify_query("so who am i") == QueryType.CONVERSATIONAL
    assert await classifier.classify_query("hey what's my name again?") == QueryType.CONVERSATIONAL

    # Domain queries with overlapping question words -> MUST route to DOMAIN
    assert await classifier.classify_query("what battery do I need") == QueryType.DOMAIN
    assert await classifier.classify_query("what is the leave policy") == QueryType.DOMAIN
    assert await classifier.classify_query("what document format is supported") == QueryType.DOMAIN


@pytest.mark.asyncio
async def test_local_llm_binary_classification_yes(mock_llm_client):
    mock_llm_client.generate_response.return_value = "YES"
    classifier = QueryClassifier(llm_client=mock_llm_client, use_fast_path=False, enabled=True)

    res = await classifier.classify_query("Is this a test message")
    assert res == QueryType.CONVERSATIONAL
    mock_llm_client.generate_response.assert_called_once()


@pytest.mark.asyncio
async def test_local_llm_binary_classification_no(mock_llm_client):
    mock_llm_client.generate_response.return_value = "NO"
    classifier = QueryClassifier(llm_client=mock_llm_client, use_fast_path=False, enabled=True)

    res = await classifier.classify_query("Is this a test message")
    assert res == QueryType.DOMAIN
    mock_llm_client.generate_response.assert_called_once()


@pytest.mark.asyncio
async def test_local_llm_fail_closed_on_invalid_output(mock_llm_client):
    invalid_outputs = ["YES.", "NO.", "Yes, conversational", "I think YES", "maybe", "", "CONVERSATIONAL"]
    for out in invalid_outputs:
        mock_llm_client.generate_response.return_value = out
        classifier = QueryClassifier(llm_client=mock_llm_client, use_fast_path=False, enabled=True)
        res = await classifier.classify_query("Some ambiguous string")
        assert res == QueryType.DOMAIN, f"Expected DOMAIN for invalid output '{out}', got {res}"


@pytest.mark.asyncio
async def test_local_llm_fail_closed_on_exception_or_timeout(mock_llm_client):
    mock_llm_client.generate_response.side_effect = RuntimeError("Ollama connection timeout (2.0s)")
    classifier = QueryClassifier(llm_client=mock_llm_client, use_fast_path=False, enabled=True)

    res = await classifier.classify_query("Some ambiguous string")
    assert res == QueryType.DOMAIN, "Should fail closed to DOMAIN on exception/timeout"


@pytest.mark.asyncio
async def test_query_classifier_disabled_flag(mock_llm_client):
    classifier = QueryClassifier(llm_client=mock_llm_client, use_fast_path=False, enabled=False)

    res = await classifier.classify_query("Some ambiguous string")
    assert res == QueryType.DOMAIN
    assert not mock_llm_client.generate_response.called


@pytest.mark.asyncio
async def test_ambiguous_query_llm_classification(mock_llm_client):
    mock_llm_client.generate_response.return_value = "YES"
    classifier = QueryClassifier(llm_client=mock_llm_client)

    ambiguous_query = "I'm just testing your capability"
    res = await classifier.classify_query(ambiguous_query)
    assert res == QueryType.CONVERSATIONAL
    mock_llm_client.generate_response.assert_called_once()


@pytest.mark.asyncio
async def test_user_approval_table_matrix(classifier):
    domain_history = [
        {"role": "user", "content": "What is the maximum upload size?"},
        {"role": "assistant", "content": "The maximum document upload size is 20 MB per file."}
    ]

    assert await classifier.classify_query("Hello") == QueryType.CONVERSATIONAL
    assert await classifier.classify_query("How are you?") == QueryType.CONVERSATIONAL
    assert await classifier.classify_query("Thanks!") == QueryType.CONVERSATIONAL
    assert await classifier.classify_query("Good morning") == QueryType.CONVERSATIONAL
    assert await classifier.classify_query("Can you help me?") == QueryType.CONVERSATIONAL

    assert await classifier.classify_query("What is the leave policy?") == QueryType.DOMAIN
    assert await classifier.classify_query("Hi, what is the leave policy?") == QueryType.DOMAIN
    assert await classifier.classify_query("Can you help me configure authentication?") == QueryType.DOMAIN
    assert await classifier.classify_query("What is the maximum upload size?") == QueryType.DOMAIN

    assert await classifier.classify_query("Explain that in simpler terms", history_messages=domain_history) == QueryType.DOMAIN
    assert await classifier.classify_query("And what about images?", history_messages=domain_history) == QueryType.DOMAIN
    assert await classifier.classify_query("Thanks!", history_messages=domain_history) == QueryType.CONVERSATIONAL


@pytest.mark.asyncio
async def test_expanded_conversational_phrases_and_acknowledgements(classifier):
    """Verifies all expanded short conversational phrases work standalone and post-domain response."""
    expanded_phrases = [
        "cool", "nice", "alright", "sure", "yep", "yeah",
        "no problem", "np", "fine", "good", "great", "awesome", "sweet"
    ]

    domain_history = [
        {"role": "user", "content": "What is the maximum upload size?"},
        {"role": "assistant", "content": "The maximum document upload size is 20 MB per file."}
    ]

    # Test standalone fast-path classification
    for phrase in expanded_phrases:
        res_standalone = await classifier.classify_query(phrase)
        assert res_standalone == QueryType.CONVERSATIONAL, f"Expected CONVERSATIONAL for standalone '{phrase}', got {res_standalone}"

        res_history = await classifier.classify_query(phrase, history_messages=domain_history)
        assert res_history == QueryType.CONVERSATIONAL, f"Expected CONVERSATIONAL for post-domain '{phrase}', got {res_history}"


@pytest.mark.asyncio
async def test_memory_recall_queries(classifier):
    """Verifies memory recall queries route to CONVERSATIONAL and domain queries with overlapping words route to DOMAIN."""
    memory_recall_queries = [
        "what is my name",
        "what's my name",
        "whats my name",
        "who am i",
        "do you remember my name",
        "do you remember me",
        "hey what's my name",
        "what is my name again?",
        "hey what's my name again?",
        "so what is my name",
        "can you tell me my name",
        "what did I just ask",
        "what did I say earlier",
        "what was my question",
        "what did you say earlier",
        "what was your last answer"
    ]

    for q in memory_recall_queries:
        res = await classifier.classify_query(q)
        assert res == QueryType.CONVERSATIONAL, f"Expected CONVERSATIONAL for memory recall query: '{q}', got {res}"

    domain_overlapping_queries = [
        "what battery do I need",
        "what is the leave policy",
        "what is the battery life of aurawave pro",
        "what policy applies to me",
        "what is the config setting for my account"
    ]

    for q in domain_overlapping_queries:
        res = await classifier.classify_query(q)
        assert res == QueryType.DOMAIN, f"Expected DOMAIN for domain query with overlapping words: '{q}', got {res}"


@pytest.mark.asyncio
async def test_user_requested_benchmark_cases(classifier, mock_llm_client):
    """
    Verifies all 8 user-requested classification benchmark test cases:
    - 'I am raj' -> CONVERSATIONAL
    - 'I like soccer' -> CONVERSATIONAL
    - 'thanks' / 'ok' / 'who are you' -> CONVERSATIONAL
    - 'What are the Component 1 specifications?' -> TECHNICAL (DOMAIN)
    - 'What about the second one?' (follow-up after technical answer) -> TECHNICAL (DOMAIN)
    - Follow-up 'and what's your favorite team?' after 'I like soccer' -> CONVERSATIONAL
    """
    # 1. Identity statement
    assert await classifier.classify_query("I am raj") == QueryType.CONVERSATIONAL

    # 2. Personal statement / preference
    assert await classifier.classify_query("I like soccer") == QueryType.CONVERSATIONAL

    # 3. Thanks / ok / identity question
    assert await classifier.classify_query("thanks") == QueryType.CONVERSATIONAL
    assert await classifier.classify_query("ok") == QueryType.CONVERSATIONAL
    assert await classifier.classify_query("who are you") == QueryType.CONVERSATIONAL

    # 4. Technical / document specification question
    assert await classifier.classify_query("What are the Component 1 specifications?") == QueryType.DOMAIN

    # 5. Technical follow-up query after a technical answer
    tech_history = [
        {"role": "user", "content": "What are the Component 1 specifications?"},
        {"role": "assistant", "content": "Component 1 features dual noise-canceling microphones and Bluetooth 5.3."}
    ]
    assert await classifier.classify_query("What about the second one?", history_messages=tech_history) == QueryType.DOMAIN

    # 6. Casual follow-up question after casual intro/preference
    casual_history = [
        {"role": "user", "content": "I am raj"},
        {"role": "assistant", "content": "Hello Raj! How can I help you today?"},
        {"role": "user", "content": "I like soccer"},
        {"role": "assistant", "content": "Soccer is a great sport!"}
    ]
    assert await classifier.classify_query("and what's your favorite team?", history_messages=casual_history) == QueryType.CONVERSATIONAL






