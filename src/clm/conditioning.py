"""Helpers for selecting conditioning modes."""

VALID_CONDITIONING_TYPES = ("none", "descriptors", "graph")


def resolve_conditioning_type(
    conditional: bool = False,
    conditioning_type: str | None = None,
) -> str:
    """
    Resolve conditioning mode with backward compatibility for the legacy
    boolean `conditional` flag.
    """
    if conditioning_type is None:
        return "descriptors" if conditional else "none"

    conditioning_type = conditioning_type.lower()
    if conditioning_type not in VALID_CONDITIONING_TYPES:
        raise ValueError(
            f"Unknown conditioning type '{conditioning_type}'. "
            f"Expected one of: {', '.join(VALID_CONDITIONING_TYPES)}"
        )

    if conditioning_type == "none" and conditional:
        return "descriptors"

    return conditioning_type
