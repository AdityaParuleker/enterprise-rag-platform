"""
Phase 1 Unit Test: Route Registration & Uniqueness Audit
"""

from backend.app.main import app


def test_documents_route_uniqueness():
    """Verify /api/v1/documents route is registered exactly once in FastAPI routes."""
    all_routes = []
    for route in app.routes:
        if type(route).__name__ == "_IncludedRouter":
            all_routes.extend(route.original_router.routes)
        elif hasattr(route, "path"):
            all_routes.append(route)

    doc_routes = [
        r for r in all_routes
        if hasattr(r, "path") and r.path.rstrip("/") == "/api/v1/documents"
    ]

    # Check that POST /api/v1/documents exists exactly once
    post_doc_routes = [
        r for r in doc_routes
        if "POST" in getattr(r, "methods", set())
    ]
    assert len(post_doc_routes) == 1, f"Expected 1 POST /api/v1/documents route, found {len(post_doc_routes)}"
