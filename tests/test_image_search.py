from services import image_search
from services.image_search import ImageSearchResult, select_top_k


def test_search_images_normalizes_tavily_candidates_and_dedupes(monkeypatch):
    payload = {
        "results": [
            {
                "title": "Source page",
                "url": "https://example.com/page",
                "images": [
                    "https://cdn.example.com/a.jpg?utm_source=news",
                    {
                        "url": "https://cdn.example.com/b.jpg?ref=feed",
                        "description": "Detailed visual description",
                    },
                ],
            },
            {
                "title": "Duplicate page",
                "url": "https://example.com/duplicate",
                "images": [
                    {
                        "url": "https://cdn.example.com/a.jpg",
                        "description": "Duplicate image",
                    }
                ],
            },
        ],
        "images": [
            {
                "url": "https://cdn.example.com/c.jpg?gclid=123",
                "description": "Top-level image",
            }
        ],
    }
    calls = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return payload

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def post(self, *args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})
            return FakeResponse()

    monkeypatch.setattr(image_search.httpx, "Client", FakeClient)

    results = image_search.search_images(
        "  metallic flats  ",
        limit=10,
        api_key="test-key",
        include_image_descriptions=True,
    )

    assert [result.image_url for result in results] == [
        "https://cdn.example.com/a.jpg",
        "https://cdn.example.com/b.jpg",
        "https://cdn.example.com/c.jpg",
    ]
    assert [result.title for result in results] == [
        "Source page",
        "Detailed visual description",
        "Top-level image",
    ]
    assert len(calls) == 1
    request_json = calls[0]["kwargs"]["json"]
    assert request_json["query"] == "metallic flats"
    assert request_json["include_image_descriptions"] is True
    assert request_json["include_images"] is True


def test_select_top_k_preserves_backend_order():
    candidates = [
        ImageSearchResult(
            title=f"Result {idx}",
            page_url=f"https://example.com/{idx}",
            image_url=f"https://cdn.example.com/{idx}.jpg",
            thumbnail_url=f"https://cdn.example.com/{idx}.jpg",
            source="tavily",
            engine="tavily",
        )
        for idx in range(3)
    ]

    selected = select_top_k(candidates, k=2)

    assert selected == candidates[:2]
