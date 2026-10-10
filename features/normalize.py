"""Text preprocessing shared by every feature.

Same steps as features/translit/learning/Preprocessing.ipynb, which produced data/preprocessed_data.jsonl.
Models were trained on that output, so anything a user types must go through the same functions first.
"""
import re
import unicodedata

# ---------- Arabic script ----------

TASHKEEL = re.compile(r"[ً-ْٰ]")

# Allowed: Arabic letters ء–ي + Tunisian ڤ ڨ, digits 0-9, whitespace, punctuation
AR_NOT_ALLOWED = re.compile(
    r"[^ء-غف-يڤڨ"   # Arabic letters (ـ excluded) + ڤ ڨ
    r"0-9\s"                                       # digits + spaces
    r"!\"%'()*,\-./:;?\[\]_…،؛؟]"                 # punctuation (Latin + Arabic)
)
PUNCT_TO_DOT = re.compile(r"[!?;,،؛؟]+")
AR_ELONG = re.compile(r"([^\d\s]|[235789])\1+")


def normalize_arabic(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)                          # presentation forms → standard
    text = text.replace("ی", "ي").replace("ک", "ك")                     # Persian → Arabic
    text = TASHKEEL.sub("", text)                                       # remove diacritics
    text = text.replace("ـ", "")                                        # remove tatweel
    text = text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))    # Arabic-Indic → 0-9
    text = PUNCT_TO_DOT.sub(".", text)                                  # ! ? ; , → .
    text = AR_NOT_ALLOWED.sub(" ", text)                                # drop everything else (emojis…)
    text = re.sub(r"\s+", " ", text).strip()                            # collapse spaces
    return AR_ELONG.sub(r"\1", text)                                    # برشاااا → برشا


# ---------- Arabizi (Latin script) ----------

# Allowed: a-z, digits (3/5/7/9/8 are letters in Arabizi!), whitespace, punctuation
AZ_NOT_ALLOWED = re.compile(r"[^a-z0-9\s!\"%'()*,\-./:;?\[\]_]")
AZ_ELONG = re.compile(r"([^\d\s]|[235789])\1{2,}")


def normalize_arabizi(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("…", "...")
    text = unicodedata.normalize("NFKD", text)                          # é → e + accent mark
    text = "".join(c for c in text if not unicodedata.combining(c))     # drop accent marks
    text = text.lower()
    text = PUNCT_TO_DOT.sub(".", text)                                  # ! ? ; , → .
    text = AZ_NOT_ALLOWED.sub(" ", text)                                # drop emojis, ⊙○●🏿, <>=^ …
    text = re.sub(r"\s+", " ", text).strip()                            # collapse spaces
    return AZ_ELONG.sub(r"\1\1", text)                                  # barchaaaa → barchaa
