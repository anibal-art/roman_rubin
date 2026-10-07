"""Reproducible CROIN-origin realizations."""

from hashlib import blake2b

CAUSTIC_ORIGINS = (
    "central_caustic",
    "second_caustic",
    "third_caustic",
)

ORIGIN_SCHEME = "caustic_origin_v1"


def choose_catalog_caustic_origin(event_seed):
    payload = (
        f"roman_rubin:{ORIGIN_SCHEME}:{int(event_seed)}"
    ).encode("utf-8")

    digest = blake2b(
        payload,
        digest_size=8,
    ).digest()

    value = int.from_bytes(digest, "big")

    return CAUSTIC_ORIGINS[
        value % len(CAUSTIC_ORIGINS)
    ]
