from __future__ import annotations

import base64
import hashlib
import hmac
import os
from typing import Tuple

ALGORITHM = "pbkdf2_sha256"
ITERATIONS = 390000
SALT_BYTES = 16


def hash_password(password: str) -> str:
    salt = os.urandom(SALT_BYTES)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITERATIONS, dklen=32)
    return "$".join(
        [
            ALGORITHM,
            str(ITERATIONS),
            base64.b64encode(salt).decode("utf-8"),
            base64.b64encode(digest).decode("utf-8"),
        ]
    )


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, iter_str, salt_b64, hash_b64 = encoded.split("$")
        if algorithm != ALGORITHM:
            return False
        iterations = int(iter_str)
        salt = base64.b64decode(salt_b64.encode("utf-8"))
        stored = base64.b64decode(hash_b64.encode("utf-8"))
    except ValueError:
        return False
    new_digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations, dklen=len(stored))
    return hmac.compare_digest(new_digest, stored)
