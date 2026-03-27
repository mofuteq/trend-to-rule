from core.models import ExampleQuerySpec


_DROP_AUDIENCE_TERMS = {"unisex"}

_DROP_CONTEXT_TERMS = {"outfit", "street style", "fashion"}
_DROP_VIBE_TERMS = {
    "quality",
    "high quality",
    "sophisticated",
    "minimalism",
    "clothing",
    "fashion",
    "and",
}

_ABSTRACT_ITEM_TERMS = {"fashion", "clothing",
                        "style", "wear", "apparel", "outfit"}

_AUDIENCE_REWRITES = {
    "men": "menswear",
    "male": "menswear",
    "women": "womenswear",
    "female": "womenswear",
}

_ITEM_REWRITES = {
    "quarter-zip": "quarter zip sweater",
    "quarter zip": "quarter zip sweater",
    "half-zip": "quarter zip sweater",
    "half zip": "quarter zip sweater",
}

_LOOK_HINT_ITEMS = {
    "sweater",
    "shirt",
    "t-shirt",
    "tee",
    "hoodie",
    "jacket",
    "coat",
    "vest",
    "jeans",
    "trousers",
    "pants",
    "sneakers",
    "quarter zip sweater",
}

_CONTEXT_REWRITES = {
    "silicon valley": "tech casual",
    "smart casual": "office wear",
    "tech style": "tech casual",
    "tech-friendly": "tech casual",
}

_VIBE_REWRITES = {
    "minimalist": "minimal",
    "minimalism": "minimal",
    "tech-infused": "tech casual",
    "tech-friendly": "tech casual",
    "tech style": "tech casual",
    "quiet luxury": "minimal",
    "understated": "minimal",
    "sustainable": "organic cotton",
}


def _clean(term: str | None) -> str | None:
    """Normalize one query term and drop empty values.

    Args:
        term: One optional query term.

    Returns:
        Stripped term or `None` when empty.
    """
    if term is None:
        return None
    value = " ".join(term.strip().split())
    return value or None


def _normalize_audience(term: str | None) -> str | None:
    """Normalize audience hints and drop low-signal values.

    Args:
        term: One optional audience hint.

    Returns:
        Normalized audience hint or `None` when it should be ignored.
    """
    value = _clean(term)
    if value is None:
        return None

    lowered = value.lower()
    if lowered in _DROP_AUDIENCE_TERMS:
        return None
    return _AUDIENCE_REWRITES.get(lowered, value)


def _normalize_context(term: str | None) -> str | None:
    """Normalize context hints and drop overly broad moodboard terms.

    Args:
        term: One optional context hint.

    Returns:
        Normalized context hint or `None` when it should be ignored.
    """
    value = _clean(term)
    if value is None:
        return None

    lowered = value.lower()
    rewritten = _CONTEXT_REWRITES.get(lowered, value)
    if rewritten.lower() in _DROP_CONTEXT_TERMS:
        return None
    return rewritten


def _normalize_material(term: str | None) -> str | None:
    """Normalize material hints and rewrite broad concept words.

    Args:
        term: One optional material hint.

    Returns:
        Normalized material hint or `None` when empty.
    """
    value = _clean(term)
    if value is None:
        return None

    lowered = value.lower()
    if lowered == "sustainable":
        return "organic cotton"
    return value


def _normalize_item(term: str | None) -> str | None:
    """Normalize item hints and reject abstract non-searchable values.

    Args:
        term: One optional item hint.

    Returns:
        Normalized item hint or `None` when it is too abstract for retrieval.
    """
    value = _clean(term)
    if value is None:
        return None

    lowered = value.lower()
    rewritten = _ITEM_REWRITES.get(lowered, value)
    if rewritten.lower() in _ABSTRACT_ITEM_TERMS:
        return None
    return rewritten


def _derive_look_hint(
    *,
    item: str,
    context: str | None,
    material: str | None,
    silhouette: str | None,
    color: str | None,
) -> str | None:
    """Add a lightweight look-oriented hint when the query is too product-like.

    Args:
        item: Normalized concrete item.
        context: Normalized context term.
        material: Normalized material term.
        silhouette: Normalized silhouette term.
        color: Normalized color term.

    Returns:
        A small derived hint such as `outfit`, or `None` when not needed.
    """
    if context is not None:
        return None

    lowered_item = item.lower()
    if lowered_item not in _LOOK_HINT_ITEMS:
        return None

    modifier_count = sum(
        value is not None for value in (material, silhouette, color)
    )
    if modifier_count < 2:
        return None
    return "outfit"


def _normalize_vibe(term: str | None) -> str | None:
    """Normalize vibe terms for search rendering.

    Args:
        term: One optional vibe term.

    Returns:
        Cleaned vibe term or `None` when empty or too abstract.
    """
    value = _clean(term)
    if value is None:
        return None

    lowered = value.lower()
    rewritten = _VIBE_REWRITES.get(lowered, value)
    if rewritten.lower() in _DROP_VIBE_TERMS:
        return None
    return rewritten


def _unique_preserve_order(parts: list[str | None]) -> list[str]:
    """Drop duplicate query terms while preserving order.

    Args:
        parts: Ordered optional query parts.

    Returns:
        Ordered unique query parts.
    """
    seen: set[str] = set()
    result: list[str] = []
    for part in parts:
        if part is None:
            continue
        key = part.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(part)
    return result


def render_example_query(spec: ExampleQuerySpec) -> str:
    """Render one example-query spec as a plain search string.

    Search prioritizes concrete wardrobe, material, and context terms over
    abstract descriptive language. Broad moodboard terms are either rewritten
    into more retrieval-friendly phrases or dropped to reduce drift, while a
    lightweight derived look hint can reduce product-only retrieval.

    Args:
        spec: Structured query spec.

    Returns:
        Space-joined query string using prioritized non-empty fields.

    Raises:
        ValueError: If the required `item` field is missing or too abstract after normalization.
    """
    item = _normalize_item(spec.item)
    if item is None:
        raise ValueError("ExampleQuerySpec.item must not be empty or abstract")

    material = _normalize_material(spec.material)
    silhouette = _clean(spec.silhouette)
    color = _clean(spec.color)
    context = _normalize_context(spec.context)
    audience = _normalize_audience(spec.audience)
    vibe = _normalize_vibe(spec.vibe)
    look_hint = _derive_look_hint(
        item=item,
        context=context,
        material=material,
        silhouette=silhouette,
        color=color,
    )

    parts = _unique_preserve_order(
        [
            item,
            material,
            silhouette,
            color,
            context,
            audience,
            vibe,
            look_hint,
        ]
    )
    if len(parts) == 1:
        parts.extend(["tech casual", "minimal"])
    return " ".join(parts)
