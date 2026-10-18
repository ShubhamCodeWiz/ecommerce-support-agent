from datetime import datetime, timezone, timedelta
import jwt
import pytest

from pydantic_ai_agent.config import settings
from pydantic_ai_agent.security import (
    ALGORITHM,
    TokenError,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


# 1. Valid Token Roundtrip
def test_valid_token_roundtrip():
    user_id = 1
    token = create_access_token(user_id)
    payload = decode_access_token(token)

    assert payload["sub"] == "1"
    assert "exp" in payload
    assert "iat" in payload


# 2. Expired Token Test (Using monkeypatch)
def test_expired_token(monkeypatch):
    # एक्सपायरी को -5 मिनट कर दिया (भूतकाल का टोकन)
    monkeypatch.setattr(settings, "access_token_expire_minutes", -5)

    expired_token = create_access_token(user_id=1)

    # टोकन एक्सपायर हो चुका है, इसलिए decode_access_token को TokenError फेंकना चाहिए
    with pytest.raises(TokenError, match="Invalid or expired token"):
        decode_access_token(expired_token)


# 3. Tampered Token Test (टोकन के साथ छेड़छाड़)
def test_tampered_token():
    token = create_access_token(user_id=1)

    # टोकन के आखिरी कैरेक्टर को बदल दिया (सिग्नेचर करप्ट हो गया)
    tampered_char = "a" if token[-1] != "a" else "b"
    tampered_token = token[:-1] + tampered_char

    with pytest.raises(TokenError):
        decode_access_token(tampered_token)


# 4. Token without EXP claim (सबसे ज़रूरी टेस्ट)
def test_token_missing_exp_rejected():
    # बिना 'exp' वाला कच्चा टोकन मैन्युअली बनाया
    raw_payload = {
        "sub": "1",
        "iat": datetime.now(timezone.utc),
        # 'exp' जानबूझकर नहीं डाला
    }
    token_without_exp = jwt.encode(
        raw_payload,
        settings.secret_key.get_secret_value(),
        algorithm=ALGORITHM,
    )

    # decode_access_token में options={"require": ["exp"]} लगा है,
    # इसलिए यह बिना exp वाले टोकन को तुरंत रिजेक्ट करेगा
    with pytest.raises(TokenError):
        decode_access_token(token_without_exp)


# 5. Wrong Secret Key Test
def test_token_with_wrong_secret():
    payload = {
        "sub": "1",
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(minutes=10),
    }
    # किसी फर्जी की (Fake Key) से टोकन साइन किया
    fake_secret = "a" * 32
    token_with_wrong_key = jwt.encode(payload, fake_secret, algorithm=ALGORITHM)

    with pytest.raises(TokenError):
        decode_access_token(token_with_wrong_key)

def test_password_hash_and_verify_success():
    raw_password = "supersecretpassword123"
    hashed = hash_password(raw_password)

    # 1. हैश खाली नहीं होना चाहिए और प्लेन पासवर्ड उसमें दिखना नहीं चाहिए
    assert hashed != raw_password
    assert raw_password not in hashed

    # 2. सही पासवर्ड से verification True होना चाहिए
    assert verify_password(raw_password, hashed) is True


def test_password_verify_failure():
    raw_password = "correct_password"
    hashed = hash_password(raw_password)

    # गलत पासवर्ड से verification False आना चाहिए
    assert verify_password("wrong_password", hashed) is False


def test_password_hashing_uses_unique_salts():
    raw_password = "same_password"

    # एक ही पासवर्ड को दो बार हैश किया
    hash_1 = hash_password(raw_password)
    hash_2 = hash_password(raw_password)

    # Argon2 के ऑटोमैटिक सॉल्टिंग की वजह से दोनों हैश अलग होने चाहिए
    assert hash_1 != hash_2

    # लेकिन दोनों ही उसी पासवर्ड से सही वेरीफाई होने चाहिए
    assert verify_password(raw_password, hash_1) is True
    assert verify_password(raw_password, hash_2) is True