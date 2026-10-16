"""Offline-first statement and receipt ingestion (no AI, no network).

Statement import pipeline (no AI, no network):

    PDF bytes -> text (PyPDF2) -> normalised lines -> detect_statement_format()
      -> one parser per layout:
           phonepe / gpay / paytm / bhim  transaction "blocks" (Paid to X, DEBIT, Rs 450, UTR...)
           bank_table                     dated rows, wrapped narration, debit/credit via balance
           sms                            one forwarded bank SMS per line (or per paragraph)
           generic                        tries all of the above and keeps the best result
      -> rows (dicts) with date, merchant, type, amount, category, description, format.

Debug a real file with:  uv run python scripts/inspect_statement.py statement.pdf --text
"""

from __future__ import annotations

import io
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import clock
from app.models import Category, Merchant, Transaction

# ---------------------------------------------------------------------------
# Shared patterns
# ---------------------------------------------------------------------------

_AMOUNT = r"(?:(?:₹|rs\.?|inr)\s*([0-9][0-9,]*(?:\.\d{1,2})?)|(?:amount|spent|paid|debited|credited)\D{0,15}?([0-9][0-9,]*(?:\.\d{1,2})?))"
_NUM = r"\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?"
_MONEY2 = r"\d{1,3}(?:,\d{2,3})+\.\d{2}|\d+\.\d{2}"  # bank tables always print paise

_MON = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
    r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
_DATE_RE = re.compile(
    r"(?<![\w:])(?:"
    r"(?P<d1>\d{1,2})(?P<s1>[/\-.])(?P<m1>\d{1,2})(?P=s1)(?P<y1>\d{4}|\d{2})"
    r"|(?P<y2>\d{4})-(?P<m2>\d{1,2})-(?P<d2>\d{1,2})"
    rf"|(?P<d3>\d{{1,2}})(?:st|nd|rd|th)?[\s\-/]*(?P<mon3>{_MON})\.?"
    r"(?:[\s\-/,']*(?P<y3>\d{4}|\d{2})(?![\d:]))?"
    rf"|(?P<mon4>{_MON})\.?\s*(?P<d4>\d{{1,2}})(?:st|nd|rd|th)?(?:,\s*|\s+)(?P<y4>\d{{4}})"
    r")(?!\w)",
    re.I,
)
_TIME_RE = re.compile(r"(?<![\d:])\d{1,2}:\d{2}(?::\d{2})?\s*(?:[ap]\.?m\.?)?(?![\d\w])", re.I)
_SERIAL_RE = re.compile(r"\d{1,4}[.)]?\s+")
_VPA_RE = re.compile(r"(?<![\w.+-])([\w.+-]{2,}@[a-zA-Z][a-zA-Z0-9.]*[a-zA-Z])")
_MOBILE_RE = re.compile(r"^(?:\+?91[-\s]?)?[6-9]\d{9}(?:[-_.]\d{1,3})?$")
_BANK_MONEY_RE = re.compile(
    rf"(?<![\w.,/:@-])(?:(?P<sign>-)\s?)?₹?(?P<num>{_MONEY2})(?![\d]|[.,]\d)"
    r"(?:\s*\(?(?P<drcr>dr|cr)\)?\.?(?![a-z]))?",
    re.I,
)
_CUR_AMOUNT_RE = re.compile(rf"(?:(?P<sign>[+-])\s?)?₹\s?(?P<num>{_NUM})(?![\d])")

FORMATS = ("phonepe", "gpay", "paytm", "bhim", "bank_table", "sms", "generic")
FORMAT_LABELS: dict[str, str | None] = {
    "phonepe": "PhonePe",
    "gpay": "Google Pay",
    "paytm": "Paytm",
    "bhim": "BHIM UPI",
    "bank_table": "bank statement",
    "sms": "bank SMS",
    "generic": None,
}


def _to_float(value: str) -> float:
    return float(value.replace(",", ""))


def _month(value: str) -> int:
    return [
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    ].index(value[:3].lower()) + 1


def _date_from_match(match: re.Match[str], default_year: int | None = None) -> str | None:
    g = match.groupdict()
    try:
        if g["d1"]:
            day, month, year = int(g["d1"]), int(g["m1"]), g["y1"]
        elif g["y2"]:
            day, month, year = int(g["d2"]), int(g["m2"]), g["y2"]
        elif g["d3"]:
            day, month, year = int(g["d3"]), _month(g["mon3"]), g["y3"]
        else:
            day, month, year = int(g["d4"]), _month(g["mon4"]), g["y4"]
        if year is None:
            full_year = default_year or clock.today_local().year
        else:
            full_year = int(year) + (2000 if len(year) == 2 else 0)
        if not 1990 <= full_year <= 2100:
            return None
        return date(full_year, month, day).isoformat()
    except ValueError, TypeError:
        return None


def _parse_date(value: str, default_year: int | None = None) -> str | None:
    match = _DATE_RE.search(value or "")
    return _date_from_match(match, default_year) if match else None


def _leading_date(line: str, default_year: int | None = None) -> tuple[str, int] | None:
    """(iso date, end offset) when the line begins with a date (optionally after a serial no.)."""
    match = _DATE_RE.match(line)
    if not match:
        serial = _SERIAL_RE.match(line)
        if serial:
            match = _DATE_RE.match(line, serial.end())
    if not match:
        return None
    iso = _date_from_match(match, default_year)
    return (iso, match.end()) if iso else None


def _blank_dates(text: str) -> str:
    return _DATE_RE.sub(lambda m: " " * len(m.group(0)), text)


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

_GLUE_FIXES = [
    # PyPDF2 sometimes drops the space between text runs: "2025Paid to", "SWIGGYDEBIT".
    (re.compile(r"(?<=\d{4})(?=[A-Za-z])"), " "),
    (re.compile(r"(?<=\S)(?=(?-i:DEBIT|CREDIT)(?:\s|₹|$))"), " "),
    (
        re.compile(
            r"(?<=[^\s(])(?=(?:Paid to|Received from|Transaction ID|UTR No|Paid by|"
            r"UPI Transaction ID|UPI Ref)\b)"
        ),
        " ",
    ),
    (re.compile(r"(?<=\w)(?=₹)"), " "),
]


def _normalize_line(line: str) -> str:
    line = line.replace("\u00a0", " ").replace("\u2009", " ").replace("\u202f", " ")
    line = line.replace("\t", " ").replace("\u20a8", "₹")  # "₨" sign
    # Rupee glyphs that PDF fonts map to private-use or placeholder characters.
    line = re.sub(r"[\ue000-\uf8ff](?=\s?\d)|(?<![\w?])[`?](?=\d)", "₹", line)
    line = re.sub(r"(?i)\b(?:rs|inr)\.?\s*(?=\d)", "₹", line)
    line = re.sub(r"₹\s+(?=\d)", "₹", line)
    for pattern, replacement in _GLUE_FIXES:
        line = pattern.sub(replacement, line)
    return " ".join(line.split())


def _numbered_lines(text: str) -> list[tuple[int, str]]:
    out = []
    for number, raw in enumerate((text or "").splitlines(), start=1):
        line = _normalize_line(raw)
        if line:
            out.append((number, line))
    return out


def _guess_year(text: str) -> int | None:
    years = [int(y) for y in re.findall(r"(?<!\d)(20[0-9]{2})(?!\d)", text or "")]
    return Counter(years).most_common(1)[0][0] if years else None


# ---------------------------------------------------------------------------
# Merchant extraction
# ---------------------------------------------------------------------------


def _wordset(*chunks: str) -> frozenset[str]:
    return frozenset(word for chunk in chunks for word in chunk.split())


# Words in bank narrations that are never the merchant: transfer kinds, bank codes, remarks.
_NARRATION_NOISE = _wordset(
    "UPI DR CR P2M P2A P2P MB IB IMPS NEFT RTGS ACH NACH ECS POS PCD VPS VIN BIL BPAY ONL INB",
    "MMT TPT TRF TRANSFER TO BY FROM PAYMENT PAYMENTS PAY PAYME PAYM PAYMEN PAYMENTFROMPH NA",
    "UPIINTENT COLLECT REQUEST SENT USING VIA REF NO TXN ID D C INWARD OUTWARD REV REVERSAL",
    "DEBIT CREDIT CARD PURCHASE ECOM CHQ CLG INF INFT BILLPAY BBPS SI MANDATE AUTOPAY OTHERS",
    "OTHER PH TP PAID CLEARING MICR CTS NFS ATM WDL CASH UPIAR UPIRET IN OUT FT IFT",
    "YESB HDFC HDFCBANK ICIC ICICI SBIN SBI UTIB AXIS KKBK KOTAK PUNB BARB CNRB UBIN IDIB IOBA",
    "INDB FDRL IDFB AIRP PYTM PPIW KARB CIUB YBL IBL AXL OKAXIS OKSBI OKICICI OKHDFCBANK",
)
_BANK_SEGMENT_RE = re.compile(
    r"^(?:[a-z&.]+\s+){0,4}bank(?:\s+(?:ltd|limited|of\s+\w+))?\.?$|^state bank of india$", re.I
)
_IFSC_RE = re.compile(r"^[A-Z]{4}0[A-Z0-9]{6}$", re.I)
_NAME_STOP_RE = re.compile(
    r"\s(?-i:DEBIT|CREDIT|Debit|Credit)\b(?!\s+[Cc]ard)|[+-]?\s?₹"
    r"|\b(?:transaction\s+id|txn\s+id|utr|upi\s+(?:ref|transaction|id)|ref(?:erence)?\s*(?:no|id|#)"
    r"|order\s+id|paid\s+(?:by|with|from)|debited\s+from|credited\s+to|received\s+in|status"
    r"|notes?|tags?|on\s+\d)\b"
    r"|(?<![\d:])\d{1,2}:\d{2}",
    re.I,
)


def _display_name(value: str) -> str:
    value = " ".join(value.split()).strip(" .,:;|-_*#()[]")
    if not value:
        return ""
    if value.isupper() or value.islower():
        value = " ".join(word[:1].upper() + word[1:].lower() for word in value.split())
    return value[:40].strip()


def _clean_merchant(value: str) -> str:
    value = re.sub(r"\bUPI[-/:]?", "", value, flags=re.I)
    value = re.sub(r"\b\d{6,}\b", "", value)
    value = re.sub(r"[@#].*$", "", value).replace("-", " ")
    value = re.sub(
        r"\b(?:payment|pay|to|from|ref(?:erence)?|txn|transfer|via)\b", " ", value, flags=re.I
    )
    value = re.sub(r"\s+", " ", value).strip(" .,:;|-")
    return value.title() if value else "Unknown"


def _merchant_from_vpa(vpa: str | None) -> str | None:
    if not vpa:
        return None
    local = vpa.split("@", 1)[0]
    if _MOBILE_RE.match(local) or sum(ch.isdigit() for ch in local) >= 6:
        return vpa.lower()
    words = [w for w in re.split(r"[._\-+]+", local) if w and not w.isdigit()]
    return _display_name(" ".join(words)) or vpa.lower()


def _is_id_word(word: str) -> bool:
    """Reference numbers, masked cards/accounts, IFSC codes and VPAs: never part of a name."""
    bare = word.strip(".,:;()[]'\"").upper()
    return (
        not bare
        or "@" in bare
        or sum(ch.isdigit() for ch in bare) >= 4
        or bool(re.fullmatch(r"[X*]{3,}\w*", bare))
        or bool(_IFSC_RE.match(bare))
    )


def _is_noise_word(word: str) -> bool:
    bare = word.strip(".,:;()[]'\"").upper()
    return _is_id_word(word) or len(bare) == 1 or bare in _NARRATION_NOISE


def _segment_name(segment: str) -> str:
    """Drop ids anywhere, and transfer-kind words at the edges ("POS ... ", "ACH D")."""
    words = [w for w in segment.split() if not _is_id_word(w)]
    while words and _is_noise_word(words[0]):
        words.pop(0)
    while words and _is_noise_word(words[-1]):
        words.pop()
    return " ".join(words)


def _narration_vpa(text: str) -> str | None:
    """The VPA inside a narration. HDFC joins fields with "-" ("UPI-SWIGGY-SWIGGY@YBL-..."),
    so prefer the "/" or "-" separated field that holds the "@"."""
    for field_text in re.split(r"[/|\s]+|-", text):
        if re.fullmatch(r"[\w.+]{2,}@[A-Za-z][A-Za-z0-9.]*", field_text):
            return field_text.lower()
    found = _VPA_RE.search(text)
    return found.group(1).lower() if found else None


def merchant_from_narration(narration: str) -> tuple[str, str | None]:
    """Clean merchant name (and VPA, if any) from a bank narration or UPI string.

    ``UPI/DR/527812345678/SWIGGY/YESB/swiggy@ybl/Payment`` -> ("Swiggy", "swiggy@ybl")
    ``IMPS-527812345678-RAHUL SHARMA-HDFC-XXXX1234`` -> ("Rahul Sharma", None)
    """
    text = " ".join((narration or "").split())
    vpa = _narration_vpa(text)
    upper = text.upper()
    if re.search(
        r"\bATM\b|\bNWD\b|\bATW\b|\bEAW\b|CASH\s*(?:WDL|WITHDRAWAL|WD)\b|\bWDL\b", upper
    ) and not re.search(r"\bUPI\b", upper):
        return "ATM Withdrawal", vpa
    if re.search(r"\bINT(?:EREST)?[.\s]*(?:PD|PAID|CREDIT|CR)\b|\bCREDIT\s+INTEREST\b", upper):
        return "Interest", vpa
    if re.search(r"\bCASH\s*(?:DEP|DEPOSIT)\b|\bBY\s+CASH\b", upper):
        return "Cash Deposit", vpa
    if re.match(r"^\s*(?:EMI|LOAN)\b", upper):
        return "Loan EMI", vpa
    text = re.sub(r"^(?:BY|TO)\s+(?:TRANSFER|TRF|CLG)\b[\s:-]*", "", text, flags=re.I)
    text = re.sub(r"^(?:REV|REVERSAL|RTN|RETURN)\b[\s:/-]*", "", text, flags=re.I)
    for segment in re.split(r"[/|*:\\]+|-(?=\S)|(?<=\S)-|\s-\s", text):
        segment = segment.strip()
        if not segment or _BANK_SEGMENT_RE.match(segment):
            continue
        name = _display_name(_segment_name(segment))
        if name and not name.isdigit():
            return name, vpa
    from_vpa = _merchant_from_vpa(vpa)
    if from_vpa:
        return from_vpa, vpa
    if re.search(r"\b(?:CHQ|CHEQUE|CLG)\b", upper):
        return "Cheque", vpa
    return "Unknown", vpa


def _name_from_phrase(rest: str) -> tuple[str, str | None]:
    """Payee/payer text after "Paid to", "Received from", "To:" (cut at ids, amounts...)."""
    stop = _NAME_STOP_RE.search(rest)
    name = rest[: stop.start()] if stop else rest
    name = _blank_dates(name)
    vpa_match = _VPA_RE.search(name)
    vpa = vpa_match.group(1) if vpa_match else None
    if vpa:
        name = name.replace(vpa, " ")
    name = re.sub(r"\(\s*\)|\bupi\s*id\b|\bvpa\b", " ", name, flags=re.I)
    if re.search(r"(?:^|\s)UPI[/-]|[/-]UPI\b|IMPS|NEFT", name, re.I):
        merchant, nvpa = merchant_from_narration(name)
        return merchant, vpa or nvpa
    cleaned = _display_name(name)
    if not cleaned:
        return _merchant_from_vpa(vpa) or "Unknown", vpa
    return cleaned, vpa


# ---------------------------------------------------------------------------
# Categories (India-centric keyword table; the user's own confirmed mapping wins)
# ---------------------------------------------------------------------------

# Brands: distinctive names, safe to look for anywhere (merchant, VPA, narration).
_BRANDS: dict[str, str] = {
    "Food & Dining": "swiggy zomato dominos domino mcdonalds mcdonald kfc starbucks dunzo eatsure "
    "faasos behrouz box8 burgerking subway haldiram haldirams chaayos ccd wowmomo eatfit freshmenu "
    "rebelfoods pizzahut|biryani by kilo|burger king|pizza hut|barbeque nation|third wave coffee|"
    "blue tokai|cafe coffee day|wow momo|rebel foods|zomato online|swiggy food",
    "Groceries": "blinkit grofers zepto bigbasket bbnow instamart swiggyinstamart dmart jiomart "
    "spencers ratnadeep licious freshtohome milkbasket|swiggy instamart|avenue supermarts|"
    "more retail|more supermarket|reliance fresh|reliance smart|star bazaar|big bazaar|"
    "natures basket|country delight|jio mart|amazon fresh|flipkart minutes",
    "Transport": "uber ola olacabs rapido roppen irctc redbus abhibus dmrc bmrcl hpcl bpcl iocl "
    "indianoil nayara fastag netc yulu bluesmart parkplus|ani technologies|namma yatri|indian oil|"
    "bharat petroleum|hindustan petroleum|shell india|jio bp|ola cabs|uber india|delhi metro|"
    "mumbai metro|namma metro",
    "Shopping": "amazon flipkart myntra ajio meesho nykaa decathlon tatacliq croma pantaloons "
    "westside uniqlo ikea pepperfry lenskart firstcry snapdeal shopsy zara|tata cliq|"
    "reliance digital|reliance trends|max fashion|shoppers stop|urban ladder|amazon pay",
    "Entertainment": "netflix hotstar jiohotstar jiocinema spotify youtube bookmyshow pvr inox "
    "cinepolis primevideo sonyliv zee5 gaana wynk audible steam playstation|prime video|"
    "amazon prime|disney hotstar|youtube premium|google play|apple music|paytm insider",
    "Bills & Utilities": "bescom tneb tangedco msedcl mahadiscom bses cesc tsspdcl apspdcl kseb "
    "uppcl pspcl wbsedcl dhbvn jvvnl bwssb jio airtel vodafone bsnl mtnl hathway excitel tatasky "
    "dishtv d2h indane bharatgas|tata power|adani electricity|torrent power|mahanagar gas|"
    "indraprastha gas|adani gas|hp gas|reliance jio|vodafone idea|act fibernet|act broadband|"
    "tata play|tata sky|dish tv|sun direct|airtel dth",
    "Rent": "nobroker nestaway rentomojo|house rent",
    "Health": "apollo medplus 1mg pharmeasy netmeds practo thyrocare metropolis healthians cultfit "
    "curefit fortis manipal narayana cloudnine|tata 1mg|apollo pharmacy|lal pathlabs|dr lal|"
    "max healthcare|aster dm|cult fit",
    "Education": "byjus unacademy vedantu upgrad coursera udemy simplilearn physicswallah whitehat",
    "Insurance": "lic policybazaar acko|life insurance corporation|hdfc life|sbi life|"
    "icici prudential life|icici pru life|icici lombard|bajaj allianz|tata aig|star health|"
    "niva bupa|max bupa|care health|max life|digit insurance|insurance premium|lic premium",
    "Investments": "zerodha groww upstox angelone kuvera indmoney smallcase etmoney kfintech "
    "camsonline iccl nsdl cdsl 5paisa fyers icicidirect|angel one|angel broking|paytm money|"
    "et money|coin by zerodha|mutual fund|bse star|indian clearing corporation|hdfc securities|"
    "icici direct|kotak securities|sovereign gold",
    "Loan/EMI": "kreditbee moneyview|bajaj finance|bajaj finserv|tata capital|home credit|"
    "home loan|personal loan|car loan|loan emi",
    "Travel": "makemytrip goibibo cleartrip yatra ixigo easemytrip agoda airbnb oyo treebo "
    "fabhotels indigo vistara spicejet akasa airasia emirates|booking com|oyo rooms|air india|"
    "interglobe aviation|air asia|go first|taj hotels",
    "Cash": "|atm withdrawal|atm wdl|cash withdrawal|cash wdl",
}
# Generic shop words: only trusted when they are part of the payee's own name.
_GENERIC: dict[str, str] = {
    "Food & Dining": "restaurant restaurants cafe dhaba bakery bakers sweets canteen food foods "
    "kitchen pizza burger eatery juice chai tiffin caterers rasoi bhojanalaya|tea stall",
    "Groceries": "supermarket grocery groceries kirana provisions vegetables fruits dairy "
    "|general store|departmental store|provision store",
    "Transport": "petrol fuel fuels toll parking taxi cab cabs metro railway|filling station|"
    "service station",
    "Shopping": "fashion garments textiles electronics footwear mall boutique",
    "Entertainment": "cinema cinemas movies multiplex",
    "Bills & Utilities": "electricity recharge broadband postpaid prepaid dth utility utilities "
    "|water bill|gas bill|bill payment|property tax",
    "Health": "pharmacy pharma medical medicals chemist chemists hospital hospitals clinic "
    "diagnostics diagnostic pathlab dental doctor healthcare gym fitness",
    "Education": "school college university academy tuition coaching institute vidyalaya",
    "Insurance": "insurance",
    "Investments": "sip mf ppf nps",
    "Loan/EMI": "emi loan",
    "Travel": "hotels resort resorts airlines airline airways travels tours",
    "Cash": "atm",
}
# Reliable even in a narration remark of a person-to-person payment ("RAHUL/rent").
_STRONG_CATEGORIES = {"Cash", "Rent", "Loan/EMI", "Investments", "Insurance"}
_STRONG_WORDS: dict[str, str] = {
    "Cash": "atm nwd atw eaw|atm wdl|cash wdl|cash withdrawal",
    "Rent": "rent|house rent",
    "Loan/EMI": "emi|loan emi|nach emi",
    "Investments": "sip|mutual fund",
    "Insurance": "lic|insurance premium",
}
_BUSINESS_WORDS = _wordset(
    "store stores mart enterprises enterprise traders trading services service pvt private ltd "
    "limited llp inc co company corporation corp agency agencies industries solutions "
    "technologies tech foods hotel restaurant cafe bakery medical medicals pharmacy hospital "
    "clinic sweets center centre bank finance motors petroleum fuels station shop mall market "
    "india bazaar bazar retail supermarket collection collections fashion emporium online "
    "digital labs kitchen house world point hub express"
)
_P2P_MARKERS = re.compile(
    r"\b(?:imps|neft|rtgs|p2a|p2p|mmt|trf|transfer|transferred|money\s+sent|sent\s+to)\b", re.I
)


def _build_index(table: dict[str, str]) -> list[tuple[str, str]]:
    index = []
    for category, spec in table.items():
        single, *multi = spec.split("|")
        for keyword in single.split() + [m.strip() for m in multi]:
            if keyword:
                index.append((keyword, category))
    return sorted(index, key=lambda item: len(item[0]), reverse=True)


_BRAND_INDEX = _build_index(_BRANDS)
_GENERIC_INDEX = _build_index(_GENERIC)
_STRONG_INDEX = _build_index(_STRONG_WORDS)


def _word_text(value: str) -> str:
    """Lower-case words separated by single spaces, padded: "Domino's-Pizza" -> " dominos pizza "."""
    value = (value or "").lower().replace("'", "").replace("\u2019", "")
    return " " + " ".join(re.split(r"[^a-z0-9]+", value)).strip() + " "


def _keyword_category(text: str, index: list[tuple[str, str]], *, prefix: bool) -> str | None:
    """Longest whole-word keyword match ("ola" never matches "hola" or "cola")."""
    words = _word_text(text)
    if not words.strip():
        return None
    tokens = words.split()
    for keyword, category in index:
        if f" {keyword} " in words:
            return category
        # Concatenated brand forms ("swiggystores", "netflixcom") for long, distinctive names.
        if (
            prefix
            and " " not in keyword
            and len(keyword) >= 6
            and any(token.startswith(keyword) for token in tokens)
        ):
            return category
    return None


def _vpa_words(vpa: str | None) -> str:
    return re.sub(r"[._\-+]+", " ", vpa.split("@", 1)[0]) if vpa else ""


def _looks_like_person(merchant: str, vpa: str | None, description: str) -> bool:
    local = vpa.split("@", 1)[0] if vpa else ""
    if local and _MOBILE_RE.match(local):
        return True
    if merchant and _MOBILE_RE.match(merchant.split("@", 1)[0].replace(" ", "")):
        return True
    words = [w.lower().strip(".") for w in (merchant or "").split()]
    if (
        not words
        or len(words) > 4
        or merchant == "Unknown"
        or any(not w.isalpha() for w in words)
        or any(w in _BUSINESS_WORDS for w in words)
    ):
        return False
    vpa_tokens = set(_vpa_words(vpa).lower().split())
    if vpa_tokens & {w for w in words if len(w) >= 3}:
        return True
    return bool(_P2P_MARKERS.search(description or ""))


def _learned_category(merchant: str, user_id: str, db: Session) -> str | None:
    normalized = merchant.casefold().strip()
    learned = (
        db.execute(
            select(Category.name)
            .join(Merchant, Merchant.default_category_id == Category.id)
            .where(Merchant.user_id == user_id, func.lower(Merchant.name) == normalized)
        )
        .scalars()
        .first()
    )
    if learned:
        return learned
    # Supports older ledger rows where the merchant did not yet have a default.
    return (
        db.execute(
            select(Category.name)
            .join(Transaction, Transaction.category_id == Category.id)
            .join(Merchant, Transaction.merchant_id == Merchant.id)
            .where(Merchant.user_id == user_id, func.lower(Merchant.name) == normalized)
            .order_by(Transaction.created_at.desc())
        )
        .scalars()
        .first()
    )


def categorize_transaction(
    merchant: str,
    description: str = "",
    tx_type: str = "expense",
    *,
    user_id: str | None = None,
    db: Session | None = None,
    vpa: str | None = None,
) -> str:
    """Pick a category, conservatively.

    Order: income -> "Income"; the user's own confirmed merchant/category pair; a known Indian
    brand or shop word in the payee name/VPA; strong narration codes (ATM, EMI, rent, SIP);
    person-to-person payments -> "Transfers"; known brands in the narration. Anything else stays
    ``Uncategorized`` so a wrong guess never becomes financial data.
    """
    if tx_type == "income":
        return "Income"
    merchant = (merchant or "").strip()
    if user_id and db is not None and merchant and merchant != "Unknown":
        learned = _learned_category(merchant, user_id, db)
        if learned and learned != "Uncategorized":
            return learned
    if vpa is None:
        found = _VPA_RE.search(description or "")
        vpa = found.group(1) if found else None
    payee = f"{merchant if merchant != 'Unknown' else ''} {_vpa_words(vpa)}"
    hit = _keyword_category(payee, _BRAND_INDEX, prefix=True) or _keyword_category(
        payee, _GENERIC_INDEX, prefix=False
    )
    if hit:
        return hit
    hit = _keyword_category(description, _STRONG_INDEX, prefix=False)
    if hit:
        return hit
    if _looks_like_person(merchant, vpa, description):
        return "Transfers"
    return _keyword_category(description, _BRAND_INDEX, prefix=False) or "Uncategorized"


# ---------------------------------------------------------------------------
# Lines that are never transactions
# ---------------------------------------------------------------------------

_HEADER_WORDS = {
    "date": r"\bdate\b|\bdt\b",
    "narr": r"\b(?:narration|particulars|description|details|remarks)\b",
    "ref": r"\b(?:chq|cheque|ref)\b",
    "debit": r"\b(?:withdrawals?|debits?|dr)\b",
    "credit": r"\b(?:deposits?|credits?|cr)\b",
    "balance": r"\bbalance\b|\bbal\b",
    "amount": r"\bamount\b|\bamt\b",
    "type": r"\btype\b|\btime\b",
}
_NOISE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "page number",
        re.compile(
            r"^(?:page\s*(?:no\.?)?\s*:?\s*)?\d+\s*(?:of|/)\s*\d+$|\bpage\s*\d+\s*(?:of|/)\s*\d+\b"
            r"|^page\s*\d+$|^-\s*\d+\s*-$",
            re.I,
        ),
    ),
    (
        "totals/summary",
        re.compile(
            r"^(?:grand\s+|page\s+|sub\s*)?totals?\b|^(?:dr|cr)\s+count\b|statement\s+summary"
            r"|^summary\b|^total\s+(?:debits?|credits?|withdrawals?|deposits?)",
            re.I,
        ),
    ),
    (
        "disclaimer/footer",
        re.compile(
            r"computer[\s-]generated|system[\s-]generated|does not require (?:any )?signature"
            r"|customer\s*care|call\s+us|toll[\s-]free|www\.|https?://|registered office"
            r"|regd\.?\s*office|\bcin\b\s*[:\-]|\bgstin\b|disclaimer|terms\s*(?:and|&)\s*conditions"
            r"|end of statement|for any (?:queries|query|discrepanc)|contact us|report any"
            r"|^legends?\b|abbreviations|unless the constituent|\bdicgc\b|deposit insurance"
            r"|^important\b|^note\s*:|^\*+|please (?:do not|note|contact|review|check)"
            r"|grievance|this is an? (?:electronic|computer|system|auto)",
            re.I,
        ),
    ),
    (
        "account details",
        re.compile(
            r"^(?:account|a/c|acct|customer|cust|ifsc|micr|branch|address|name|e-?mail|phone"
            r"|mobile|nominee|nomination|joint|currency|product|scheme|statement|period|from date"
            r"|to date|ckyc|pan|cif|od limit|drawing power|interest rate|open(?:ing)? date)\b"
            r"[^:]{0,40}:"
            r"|^(?:statement of (?:account|transactions)|account statement|transaction statement"
            r"|transaction history|upi statement|passbook)\b",
            re.I,
        ),
    ),
    (
        "bank name",
        re.compile(
            r"^(?:hdfc|icici|axis|kotak(?:\s+mahindra)?|yes|idfc(?:\s+first)?|indusind|federal"
            r"|canara|union|punjab national|bank of baroda|bank of india|paytm payments)\s+bank"
            r"(?:\s+(?:ltd|limited))?\.?$|^state bank of india\.?$",
            re.I,
        ),
    ),
]
_OPENING_RE = re.compile(
    r"\bopening\s+bal(?:ance)?\b|\bbalance\s+(?:b/?f|brought\s+forward|as\s+on)\b|\bb/f\b"
    r"|\bbrought\s+forward\b|\bprevious\s+balance\b",
    re.I,
)
_CLOSING_RE = re.compile(
    r"\bclosing\s+bal(?:ance)?\b|\bbalance\s+c/?f\b|\bc/f\b|\bcarried\s+forward\b", re.I
)
_PERIOD_GAP_RE = re.compile(
    r"\s*(?:statement\s+)?(?:period|for\s+the\s+period|from|between)?\s*[:\-]?\s*"
    r"(?:-|to|–|till|and)\s*",
    re.I,
)


def _is_period(line: str) -> bool:
    """ "Oct 01, 2025 - Oct 31, 2025" / "Period: 01/10/2025 to 31/10/2025"."""
    if len(_DATE_RE.findall(line)) != 2:
        return False
    return bool(_PERIOD_GAP_RE.fullmatch(_DATE_RE.sub(" ", line)))


_DIRECTION_RE = re.compile(
    r"(?<!\w)(?P<phrase>"
    r"(?P<inc>money\s+received\s+from|received\s+from|payment\s+from|refund\s+(?:from|of|for)"
    r"|cashback\s+(?:from|received|of)|reversal\s+(?:from|of)|from\s*:|payer\s*:)"
    r"|(?P<exp>paid\s+to|payment\s+to|money\s+sent\s+to|sent\s+to|transferred\s+to|transfer\s+to"
    r"|paid\s+for|bill\s+paid(?:\s+(?:for|to))?|mobile\s+recharged?|recharged?\s+for"
    r"|to\s*:|payee\s*:)"
    r")\s*[:\-]?\s*(?P<rest>.*)",
    re.I,
)
_STATUS_SKIP_RE = re.compile(
    r"\b(?:failed|failure|declined|pending|cancell?ed|unsuccessful|expired|rejected|in\s+progress)\b",
    re.I,
)
_REFUND_RE = re.compile(r"\b(?:refund(?:ed)?|reversal|reversed|cashback|rev)\b", re.I)


def _header_hits(line: str) -> set[str]:
    return {name for name, pattern in _HEADER_WORDS.items() if re.search(pattern, line, re.I)}


def _is_table_header(line: str) -> bool:
    if re.search(r"\d+\.\d{2}|@", line) or _DIRECTION_RE.search(line):
        return False
    hits = _header_hits(line)
    words = len(line.split())
    if len(hits) >= 3 and words <= 20:
        return True
    # Header split over two lines ("Withdrawal Amt. Deposit Amt."), never a "UPI/DR/..." wrap.
    return len(hits) >= 2 and words <= 8 and not re.search(r"[\d/]", line)


def _is_bank_header(line: str) -> bool:
    hits = _header_hits(line)
    return (
        _is_table_header(line)
        and len(hits - {"amount", "type"}) >= 3
        and ("balance" in hits or {"debit", "credit"} <= hits)
    )


def _noise_reason(line: str) -> str | None:
    """Why a (non-dated) line is not a transaction, or None if it might be one."""
    if _is_period(line):
        return "statement period"
    # Page numbers and "Account No :" style lines first: "Statement From : .. To : .." is not
    # a "From: payer" line.
    for reason, pattern in _NOISE_PATTERNS:
        if reason in ("page number", "account details") and pattern.search(line):
            return reason
    if _DIRECTION_RE.search(line):
        return None
    if _is_table_header(line):
        return "table header"
    for reason, pattern in _NOISE_PATTERNS:
        if pattern.search(line):
            return reason
    return None


# ---------------------------------------------------------------------------
# Parse results
# ---------------------------------------------------------------------------


@dataclass
class StatementParse:
    """Everything one parse produced: rows plus what was skipped and why (for debugging)."""

    format: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[tuple[int, str, str]] = field(default_factory=list)  # (line no, reason, text)
    text: str = ""


def _row(
    *,
    fmt: str,
    tx_date: str | None,
    merchant: str,
    tx_type: str,
    amount: float,
    description: str,
    vpa: str | None = None,
    bank_account: str | None = None,
    reference: str | None = None,
    balance: float | None = None,
    time: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    return {
        "date": tx_date,
        "time": time,
        "merchant": merchant or "Unknown",
        "vpa": vpa,
        "type": tx_type,
        "amount": round(amount, 2),
        "bank_account": bank_account,
        "reference": reference,
        "balance": balance,
        "category": None,  # filled in by _finalise_rows
        "description": description[:300],
        "format": fmt,
        "note": note,
        "source": "pdf",
    }


# ---------------------------------------------------------------------------
# UPI app statements (PhonePe, Google Pay, Paytm, BHIM): one block per transaction
# ---------------------------------------------------------------------------

_REF_RE = re.compile(
    r"\b(?:utr(?:\s*no)?|upi\s+(?:ref(?:erence)?|transaction)\s*(?:no|id)?|ref(?:erence)?\s*(?:no|id)"
    r"|transaction\s+id|txn\s+id|order\s+id)\b\.?\s*[:#.\-]*\s*(?P<ref>[A-Za-z0-9]{6,})",
    re.I,
)
_ACCOUNT_RE = re.compile(
    r"\b(?:paid\s+(?:by|with|from)|debited\s+from|credited\s+to|received\s+in)\s*[:\-]?\s*", re.I
)


def _block_account(text: str) -> str | None:
    """ "Paid by XXXXXXXX1234" / "Paid with HDFC Bank 1234" -> the account text."""
    match = _ACCOUNT_RE.search(text)
    if not match:
        return None
    rest = text[match.end() :]
    stop = _NAME_STOP_RE.search(rest)
    rest = _DATE_RE.split(rest[: stop.start()] if stop else rest)[0]
    words = rest.split()[:5]
    return " ".join(words).strip(" -:,") or None


_TYPE_TOKEN_RE = re.compile(
    r"(?<!\w)(?P<t>DEBIT|CREDIT|Debit|Credit|DEBITED|CREDITED|Debited|Credited)(?!\w)(?!\s+[Cc]ard)"
)


def _block_amount(text: str) -> tuple[float | None, str | None]:
    """(amount, sign) from a block: currency-marked first, then a number next to DEBIT/CREDIT."""
    cleaned = _blank_dates(_TIME_RE.sub(" ", text))
    match = _CUR_AMOUNT_RE.search(cleaned)
    if match:
        return _to_float(match.group("num")), match.group("sign")

    def plausible(num: str) -> bool:  # not a UTR / phone / account number
        return len(num.split(".")[0].replace(",", "")) <= 8

    for match in re.finditer(
        rf"(?<!\w)(?:debit|credit|dr|cr)(?:ed)?\s*[:\-]?\s*(?P<num>{_NUM})(?![\d\w])", cleaned, re.I
    ):
        if plausible(match.group("num")):
            return _to_float(match.group("num")), None
    for match in re.finditer(rf"(?:^|\s)(?P<sign>[+-])\s?(?P<num>{_NUM})(?![\d\w])", cleaned):
        if plausible(match.group("num")):
            return _to_float(match.group("num")), match.group("sign")
    decimals = re.findall(rf"(?<![\w.,/:@])({_MONEY2})(?![\d\w])", cleaned)
    return (_to_float(decimals[-1]), None) if decimals else (None, None)


def _parse_block(lines: list[tuple[int, str]], fmt: str, ctx: dict[str, Any]):
    text = " ".join(line for _, line in lines)
    direction = _DIRECTION_RE.search(text)
    type_token = _TYPE_TOKEN_RE.search(text)
    amount, sign = _block_amount(text)
    if amount is None or amount <= 0:
        return None, "no amount found"
    if not direction and not type_token and not sign:
        return None, "no Paid to / Received from / DEBIT / CREDIT marker"

    if type_token:
        tx_type = "expense" if type_token.group("t").lower().startswith("debit") else "income"
    elif sign:
        tx_type = "expense" if sign == "-" else "income"
    elif direction and direction.group("inc"):
        tx_type = "income"
    else:
        tx_type = "expense"

    if _STATUS_SKIP_RE.search(text) and not (tx_type == "income" and _REFUND_RE.search(text)):
        return None, "failed/pending transaction"
    if re.search(r"\breversed\b", text, re.I) and tx_type == "expense":
        return None, "reversed payment"

    vpa = None
    if direction:
        phrase = direction.group("phrase").lower()
        merchant, vpa = _name_from_phrase(direction.group("rest"))
        if "recharge" in phrase:
            merchant = "Mobile Recharge"
        elif phrase.startswith("bill paid") and merchant == "Unknown":
            merchant = "Bill Payment"
    else:
        # "Oct 05, 2025 Mobile recharged DEBIT ₹199": the words between the date and DEBIT.
        head = _blank_dates(_TIME_RE.sub(" ", lines[0][1]))
        head = head[: type_token.start()] if type_token and type_token.start() < len(head) else head
        merchant, vpa = _name_from_phrase(head)
    if vpa is None:
        found = _VPA_RE.search(text)
        vpa = found.group(1) if found else None
        if merchant == "Unknown" and vpa:
            merchant = _merchant_from_vpa(vpa) or merchant

    date_match = _DATE_RE.search(text)
    tx_date = _date_from_match(date_match, ctx.get("default_year")) if date_match else None
    time_match = _TIME_RE.search(text)
    ref_match = _REF_RE.search(text)
    refs = {m.group(0).split()[0].lower(): m.group("ref") for m in _REF_RE.finditer(text)}
    reference = (
        refs.get("utr") or refs.get("upi") or (ref_match.group("ref") if ref_match else None)
    )
    detail = direction.group(0) if direction else lines[0][1]
    stop = _NAME_STOP_RE.search(detail, 1)
    detail = (detail[: stop.start()] if stop else detail).strip()
    description = " ".join(
        part for part in (tx_date, detail, f"ref {reference}" if reference else "") if part
    )
    return (
        _row(
            fmt=fmt,
            tx_date=tx_date,
            merchant=merchant,
            tx_type=tx_type,
            amount=amount,
            description=description,
            vpa=vpa,
            bank_account=_block_account(text),
            reference=reference,
            time=time_match.group(0) if time_match else None,
        ),
        None,
    )


def _parse_app_blocks(entries: list[tuple[int, str]], fmt: str, ctx: dict[str, Any]):
    result = StatementParse(format=fmt)
    blocks: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    state = {"date": False, "dir": False, "amt": False}

    def start(entry: tuple[int, str]) -> None:
        nonlocal current
        if current:
            blocks.append(current)
        current = [entry]
        state.update(date=False, dir=False, amt=False)
        track(entry[1])

    def track(line: str) -> None:
        state["date"] = state["date"] or bool(_DATE_RE.search(line))
        state["dir"] = state["dir"] or bool(_DIRECTION_RE.search(line))
        state["amt"] = state["amt"] or "₹" in line

    for number, line in entries:
        dated = _leading_date(line, ctx.get("default_year")) is not None
        reason = None if dated and not _is_period(line) else _noise_reason(line)
        if reason:
            result.skipped.append((number, reason, line))
            continue
        has_dir = bool(_DIRECTION_RE.search(line))
        # New block: a dated line (unless it completes a "details first, date after" block),
        # or a second "Paid to / Received from" line.
        new_by_date = dated and (state["date"] or state["amt"] or not state["dir"])
        if not current or new_by_date or (has_dir and state["dir"]):
            start((number, line))
        else:
            current.append((number, line))
            track(line)
    if current:
        blocks.append(current)

    for block in blocks:
        row, reason = _parse_block(block, fmt, ctx)
        if row:
            result.rows.append(row)
        else:
            for number, line in block:
                result.skipped.append((number, reason or "not a transaction", line))
    return result


def _parse_phonepe(entries, ctx):
    """PhonePe: "Oct 05, 2025 Paid to X DEBIT ₹450" + time, Transaction ID, UTR No, Paid by."""
    return _parse_app_blocks(entries, "phonepe", ctx)


def _parse_gpay(entries, ctx):
    """Google Pay: "05 Oct, 2025 Paid to X ₹450.00" + time, UPI Transaction ID, Paid by/with."""
    return _parse_app_blocks(entries, "gpay", ctx)


def _parse_paytm(entries, ctx):
    """Paytm: "Paid to/Money sent to/Received from", "- Rs.450" signs, UPI Ref No, yearless dates."""
    return _parse_app_blocks(entries, "paytm", ctx)


def _parse_bhim(entries, ctx):
    """BHIM: "Debit ₹450 To: payee@vpa UPI Ref No ..."; the payee is often only a VPA."""
    return _parse_app_blocks(entries, "bhim", ctx)


# ---------------------------------------------------------------------------
# Bank statement tables (HDFC / SBI / ICICI / Axis / Kotak style)
# ---------------------------------------------------------------------------

_EXPENSE_CODES = re.compile(
    r"\bUPI/DR\b|/DR/|\bDR\b|\bWDL\b|\bPOS\b|\bATM\b|\bNWD\b|\bDEBIT\b|\bTO\s+TRANSFER\b|\bPURCHASE\b"
    r"|\bCHRGS?\b|\bCHARGES\b|\bEMI\b|\bACH\s*D\b|\bNACH\s*DR\b|\bPAID\b|\bSENT\b",
    re.I,
)
_INCOME_CODES = re.compile(
    r"\bUPI/CR\b|/CR/|\bCR\b|\bBY\s+TRANSFER\b|\bNEFT\s*CR\b|\bCREDIT\b|\bINT[.\s]*PD\b|\bSALARY\b"
    r"|\bREFUND\b|\bREVERSAL\b|\bREV\b|\bCASHBACK\b|\bDEPOSIT\b|\bCASH\s+DEP\b|\bRECEIVED\b"
    r"|\bDIVIDEND\b",
    re.I,
)


def _money_tokens(text: str) -> list[tuple[float, str | None, bool, re.Match[str]]]:
    tokens = []
    for match in _BANK_MONEY_RE.finditer(_blank_dates(text)):
        drcr = (match.group("drcr") or "").lower() or None
        tokens.append((_to_float(match.group("num")), drcr, bool(match.group("sign")), match))
    return tokens


def _delta_type(prev: float | None, amount: float, balance: float | None) -> str | None:
    if prev is None or balance is None:
        return None
    if abs(prev - amount - balance) < 0.015:
        return "expense"
    if abs(prev + amount - balance) < 0.015:
        return "income"
    return None


def _parse_bank_table(entries: list[tuple[int, str]], ctx: dict[str, Any]) -> StatementParse:
    result = StatementParse(format="bank_table")
    raw_rows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    opening: float | None = None
    credit_first = False
    has_balance_col: bool | None = None
    after_summary = False

    def close() -> None:
        nonlocal current
        if current:
            raw_rows.append(current)
        current = None

    for number, line in entries:
        if after_summary:
            tokens = _money_tokens(line)
            if tokens and opening is None:
                # HDFC: "Opening Balance  Dr Count  Cr Count ..." then the numbers line.
                opening = tokens[0][0]
                result.skipped.append((number, "statement summary (opening balance)", line))
            else:
                result.skipped.append((number, "after statement summary", line))
            continue
        if re.search(r"statement\s+summary", line, re.I):
            # A summary after the table ends it; one before the table is just skipped.
            after_summary = bool(raw_rows or current)
            close()
            result.skipped.append((number, "statement summary", line))
            continue
        if _is_table_header(line):
            low = line.lower()
            if "balance" in _header_hits(line):
                has_balance_col = True
            deb = re.search(r"withdrawal|debit|\bdr\b", low)
            cre = re.search(r"deposit|credit|\bcr\b", low)
            if deb and cre:
                credit_first = cre.start() < deb.start()
            close()
            result.skipped.append((number, "table header", line))
            continue
        if _OPENING_RE.search(line):
            tokens = _money_tokens(line)
            if tokens and opening is None and not raw_rows and current is None:
                value, drcr, _, _ = tokens[-1]
                opening = -value if drcr == "dr" else value
            close()
            result.skipped.append((number, "opening balance", line))
            continue
        if _CLOSING_RE.search(line) and not _leading_date(line, ctx.get("default_year")):
            close()
            result.skipped.append((number, "closing balance", line))
            continue
        lead = _leading_date(line, ctx.get("default_year"))
        if lead and not _is_period(line):
            close()
            rest = line[lead[1] :].strip()
            current = {"line": number, "date": lead[0], "parts": [rest] if rest else []}
            continue
        reason = _noise_reason(line)
        if reason:
            close()
            result.skipped.append((number, reason, line))
            continue
        if current is None:
            result.skipped.append((number, "not a transaction (no date)", line))
            continue
        if _money_tokens(line) and len(_money_tokens(" ".join(current["parts"]))) >= 2:
            result.skipped.append((number, "extra numbers after a complete row", line))
            continue
        if len(line) > 160:
            result.skipped.append((number, "long text (footer?)", line))
            continue
        parts = current["parts"]
        # Wrapped codes like "UPI/DR/1234/SWIG" + "GY/YESB/..." continue without a space.
        if parts and re.search(r"[/\-]\S*$", parts[-1]) and not re.search(r"\s", line[:1]):
            last = parts[-1].split()[-1]
            if re.search(r"[A-Za-z0-9]$", last) and re.match(r"[A-Za-z0-9]", line):
                parts[-1] = parts[-1] + line
                continue
        parts.append(line)
    close()

    # Pass 1: amount, balance and any explicit debit/credit marker per row.
    prepared = []
    for raw in raw_rows:
        text = " ".join(raw["parts"])
        tokens = _money_tokens(text)
        if not tokens:
            result.skipped.append((raw["line"], "no amount found", text))
            continue
        explicit = None
        balance = None
        if has_balance_col is False or len(tokens) == 1:
            amount, drcr, negative, _ = next((t for t in tokens if t[1]), tokens[-1])
            amount_token = (amount, drcr, negative)
        else:
            bal_value, bal_drcr, bal_neg, _ = tokens[-1]
            balance = -bal_value if bal_drcr == "dr" or bal_neg else bal_value
            amount_token = tokens[-2][:3]
            if len(tokens) >= 3:
                first, second = tokens[-3][0], tokens[-2][0]
                if first == 0 and second > 0:
                    amount_token = tokens[-2][:3]
                    explicit = "expense" if credit_first else "income"
                elif second == 0 and first > 0:
                    amount_token = tokens[-3][:3]
                    explicit = "income" if credit_first else "expense"
        amount, drcr, negative = amount_token
        if drcr:
            explicit = "expense" if drcr == "dr" else "income"
        elif negative:
            explicit = "expense"
        chars = list(_blank_dates(text))
        for token in tokens:
            chars[token[3].start() : token[3].end()] = " " * (token[3].end() - token[3].start())
        narration = " ".join("".join(chars).split())
        prepared.append(
            {
                "line": raw["line"],
                "date": raw["date"],
                "amount": amount,
                "balance": balance,
                "explicit": explicit if drcr or negative else None,
                "column": explicit if not (drcr or negative) else None,
                "narration": narration,
            }
        )

    # Pass 2: debit vs credit from the running balance (works for newest-first lists too).
    balances = [p["balance"] for p in prepared]
    forward = sum(
        1
        for i in range(1, len(prepared))
        if _delta_type(balances[i - 1], prepared[i]["amount"], balances[i])
    )
    backward = sum(
        1
        for i in range(len(prepared) - 1)
        if _delta_type(balances[i + 1], prepared[i]["amount"], balances[i])
    )
    newest_first = backward > forward
    for i, item in enumerate(prepared):
        neighbour = i + 1 if newest_first else i - 1
        prev = balances[neighbour] if 0 <= neighbour < len(prepared) else opening
        note = None
        tx_type = item["explicit"] or _delta_type(prev, item["amount"], item["balance"])
        if tx_type is None:
            tx_type = item["column"]
        if tx_type is None:
            is_exp = bool(_EXPENSE_CODES.search(item["narration"]))
            is_inc = bool(_INCOME_CODES.search(item["narration"]))
            if is_inc and not is_exp:
                tx_type = "income"
            elif is_exp and not is_inc:
                tx_type = "expense"
            else:
                tx_type, note = "expense", "debit/credit guessed"
            note = note or "debit/credit from narration"
        merchant, vpa = merchant_from_narration(item["narration"])
        ref_match = re.search(r"(?<!\d)(\d{12})(?!\d)", item["narration"])
        result.rows.append(
            _row(
                fmt="bank_table",
                tx_date=item["date"],
                merchant=merchant,
                tx_type=tx_type,
                amount=item["amount"],
                description=f"{item['date']} {item['narration']}",
                vpa=vpa,
                reference=ref_match.group(1) if ref_match else None,
                balance=item["balance"],
                note=note,
            )
        )
    return result


# ---------------------------------------------------------------------------
# Forwarded bank SMS / receipt text: one transaction per line (or per paragraph)
# ---------------------------------------------------------------------------

_SMS_VERB_RE = re.compile(
    r"\b(?:debited|credited|debit|credit|spent|paid|sent|received|withdrawn|withdrawal|deposited"
    r"|transferred|purchase|refund(?:ed)?|txn|transaction|dr|cr)\b",
    re.I,
)
_SMS_TO_RE = re.compile(
    r"\b(?:paid\s+to|sent\s+to|transfer(?:red)?\s+to|trf\s+to|to|at|towards|merchant|info)"
    r"\s*[:\-]?\s+(?:vpa\s+)?(?P<m>.+?)"
    r"(?=\s+(?:on|ref|txn|avl|a/c|acct|account|via|for|not\s+you|is|has|dated|with|using|in|from"
    r"|upi\s+ref|bal)\b|[;(]|\.\s|\.$|$)",
    re.I,
)
_SMS_FROM_RE = re.compile(
    r"\b(?:received\s+from|from|by)\s+(?:a/c\s+linked\s+to\s+)?(?:vpa\s+)?(?P<m>.+?)"
    r"(?=\s+(?:on|ref|txn|avl|a/c|acct|account|via|for|not\s+you|is|has|dated|with|using|in"
    r"|upi\s+ref|bal)\b|[;(]|\.\s|\.$|$)",
    re.I,
)
_SMS_SKIP_CANDIDATE = re.compile(
    r"^(?:₹|\d|a/c|acct|account|card|your|ur|you|the|bank|self|xx|\*)", re.I
)


def _sms_merchant(line: str, tx_type: str) -> tuple[str, str | None]:
    icici = re.search(r";\s*(?P<m>[A-Za-z][\w .&'-]{1,40}?)\s+credited\b", line)
    if icici and tx_type == "expense":
        merchant, vpa = _name_from_phrase(icici.group("m"))
        if merchant != "Unknown":
            return merchant, vpa
    patterns = (_SMS_FROM_RE, _SMS_TO_RE) if tx_type == "income" else (_SMS_TO_RE, _SMS_FROM_RE)
    for pattern in patterns:
        for match in pattern.finditer(line):
            candidate = match.group("m").strip()
            if not candidate or _SMS_SKIP_CANDIDATE.match(candidate):
                continue
            if re.search(r"[/@]|UPI-|-\d{6,}", candidate, re.I):
                merchant, vpa = merchant_from_narration(candidate)
            else:
                merchant, vpa = _clean_merchant(candidate), None
            if merchant != "Unknown":
                return merchant, vpa
    return "Unknown", None


def _parse_line_entry(
    line: str, *, strict: bool, default_year: int | None = None
) -> tuple[dict[str, Any] | None, str | None]:
    amount_match = re.search(_AMOUNT, line, re.I)
    if not amount_match:
        return None, "no amount"
    if re.search(r"\bOTP\b|one[\s-]time\s+password", line, re.I):
        return None, "OTP message"
    if strict and not _SMS_VERB_RE.search(line):
        return None, "no debit/credit word"
    debit = re.search(r"\b(?:debited|debit|paid|spent|withdrawn|dr|sent)\b", line, re.I)
    credit = re.search(
        r"\b(?:credited|credit|received|deposit(?:ed)?|salary|cr|refund(?:ed)?)\b", line, re.I
    )
    tx_type = "income" if credit and not debit else "expense"
    if _STATUS_SKIP_RE.search(line) and not (tx_type == "income" and _REFUND_RE.search(line)):
        return None, "failed/pending transaction"
    merchant, vpa = _sms_merchant(line, tx_type)
    if merchant == "Unknown":
        merchant = _clean_merchant(_blank_dates(line[: amount_match.start()]))
    if vpa is None:
        found = _VPA_RE.search(line)
        vpa = found.group(1) if found else None
    account_match = re.search(r"\b(?:a/c|account)\s*(?:no\.?|ending)?\s*([xX*\d-]{4,})", line, re.I)
    amount = _to_float(next(group for group in amount_match.groups() if group))
    ref = re.search(r"\b(?:ref|utr|upi)[\w\s.:#]{0,8}?(\d{9,})\b", line, re.I)
    return (
        {
            **_row(
                fmt="sms",
                tx_date=_parse_date(line, default_year),
                merchant=merchant,
                tx_type=tx_type,
                amount=amount,
                description=line,
                vpa=vpa,
                bank_account=account_match.group(1) if account_match else None,
                reference=ref.group(1) if ref else None,
            ),
            "source": "sms",
        },
        None,
    )


def _sms_entries(text: str) -> list[tuple[int, str]]:
    """One entry per SMS: paragraphs when the text has blank lines, else one per line."""
    raw_lines = (text or "").splitlines()
    if any(not line.strip() for line in raw_lines) and len(raw_lines) > 1:
        entries, buffer, first = [], [], 0
        for number, raw in enumerate(raw_lines + [""], start=1):
            if raw.strip():
                if not buffer:
                    first = number
                buffer.append(_normalize_line(raw))
            elif buffer:
                entries.append((first, " ".join(buffer)))
                buffer = []
        return entries
    return _numbered_lines(text)


def _parse_sms(entries: list[tuple[int, str]], ctx: dict[str, Any], *, strict: bool = True):
    result = StatementParse(format="sms")
    for number, line in entries:
        if re.match(r"^(?:grand\s+|page\s+)?totals?\b", line, re.I):
            result.skipped.append((number, "totals/summary", line))
            continue
        row, reason = _parse_line_entry(line, strict=strict, default_year=ctx.get("default_year"))
        if row:
            result.rows.append(row)
        else:
            result.skipped.append((number, reason or "not a transaction", line))
    return result


# ---------------------------------------------------------------------------
# Format detection and the public entry points
# ---------------------------------------------------------------------------

_BRAND_MARKERS = {
    "phonepe": re.compile(r"phone\s?pe", re.I),
    "gpay": re.compile(r"google\s?pay|\bgpay\b", re.I),
    "paytm": re.compile(r"(?<![@\w.])paytm\b", re.I),
    "bhim": re.compile(r"\bbhim\b", re.I),
}


def _bank_row_count(lines: list[str]) -> int:
    count = 0
    for line in lines:
        if _leading_date(line) and len(_money_tokens(line)) >= 2:
            count += 1
    return count


def detect_statement_format(text: str) -> str:
    """Return "phonepe" | "gpay" | "paytm" | "bhim" | "bank_table" | "sms" | "generic"."""
    lines = [line for _, line in _numbered_lines(text)]
    if not lines:
        return "generic"
    # Bank tables first: their narrations often mention PhonePe/Paytm, app statements never
    # have a Debit/Credit/Balance header or two paise amounts (amount + balance) per row.
    if any(_is_bank_header(line) for line in lines) or _bank_row_count(lines) >= 3:
        return "bank_table"
    edges = "\n".join(lines[:30] + lines[-15:])
    counts = {name: len(p.findall(edges)) for name, p in _BRAND_MARKERS.items()}
    counts["paytm"] -= len(re.findall(r"@paytm", edges, re.I))
    brand, hits = max(counts.items(), key=lambda item: item[1])
    if hits > 0:
        return brand
    low = "\n".join(lines).lower()
    if re.search(r"\butr\s*no\b", low) and "paid by" in low and "transaction id" in low:
        return "phonepe"
    if "upi transaction id" in low:
        return "gpay"
    if re.search(r"money\s+(?:sent\s+to|received\s+from)", low):
        return "paytm"
    if _bank_row_count(lines) >= 2 and (
        "balance" in low or re.search(r"\b(?:upi/|neft|imps|atm|pos)\b", low)
    ):
        return "bank_table"
    sms_like = sum(
        1
        for line in lines
        if re.search(r"\b(?:debited|credited|withdrawn|spent)\b", line, re.I)
        and re.search(r"\b(?:a/c|acct|account|card|vpa|upi|avl\.?\s*bal|ref)\b", line, re.I)
    )
    if sms_like and sms_like >= max(1, len(lines) // 3):
        return "sms"
    return "generic"


_PARSERS = {
    "phonepe": _parse_phonepe,
    "gpay": _parse_gpay,
    "paytm": _parse_paytm,
    "bhim": _parse_bhim,
    "bank_table": _parse_bank_table,
}


def _finalise_rows(
    rows: list[dict[str, Any]], *, user_id: str | None, db: Session | None
) -> list[dict[str, Any]]:
    for row in rows:
        row["category"] = categorize_transaction(
            row["merchant"],
            row.get("description") or "",
            row["type"],
            user_id=user_id,
            db=db,
            vpa=row.get("vpa"),
        )
    return rows


def parse_statement_text(
    text: str,
    *,
    source: str = "pdf",
    fmt: str | None = None,
    user_id: str | None = None,
    db: Session | None = None,
) -> StatementParse:
    """Parse extracted statement text. ``fmt`` forces a format; otherwise it is detected."""
    fmt = fmt or detect_statement_format(text)
    if fmt not in FORMATS:
        raise ValueError(f"Unknown statement format {fmt!r}; use one of {', '.join(FORMATS)}")
    ctx = {"default_year": _guess_year(text)}
    if fmt == "sms":
        result = _parse_sms(_sms_entries(text), ctx)
    elif fmt == "generic":
        # Unknown layout: try every parser and keep whichever finds the most transactions.
        lines = _numbered_lines(text)
        candidates = [
            _parse_sms(lines, ctx),
            _parse_app_blocks(lines, "generic", ctx),
            _parse_bank_table(lines, ctx),
        ]
        result = max(candidates, key=lambda r: len(r.rows))  # first wins ties: line parser
        for row in result.rows:
            row["format"] = "generic"
        result.format = "generic"
    else:
        result = _PARSERS[fmt](_numbered_lines(text), ctx)
    result.text = text
    for row in result.rows:
        row["source"] = source
    _finalise_rows(result.rows, user_id=user_id, db=db)
    return result


def summarize_statement(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts, totals, date range, detected format and spend per category for a preview."""
    expenses = [r for r in rows if r.get("type") != "income"]
    incomes = [r for r in rows if r.get("type") == "income"]
    dates = sorted(r["date"] for r in rows if r.get("date"))
    by_category: dict[str, float] = defaultdict(float)
    for row in expenses:
        by_category[row.get("category") or "Uncategorized"] += float(row.get("amount") or 0)
    formats = Counter(r.get("format") for r in rows if r.get("format"))
    fmt = formats.most_common(1)[0][0] if formats else None
    period = None
    if dates:
        first, last = (datetime.fromisoformat(d) for d in (dates[0], dates[-1]))
        start = f"{first:%d %b %Y}" if first.year != last.year else f"{first:%d %b}"
        period = f"{last:%d %b %Y}" if first == last else f"{start} – {last:%d %b %Y}"
    return {
        "count": len(rows),
        "expense_count": len(expenses),
        "income_count": len(incomes),
        "total_debit": round(sum(float(r.get("amount") or 0) for r in expenses), 2),
        "total_credit": round(sum(float(r.get("amount") or 0) for r in incomes), 2),
        "date_from": dates[0] if dates else None,
        "date_to": dates[-1] if dates else None,
        "period": period,
        "format": fmt,
        "format_label": FORMAT_LABELS.get(fmt) if fmt else None,
        "categories": sorted(
            ((name, round(total, 2)) for name, total in by_category.items()),
            key=lambda item: item[1],
            reverse=True,
        ),
        "uncategorized": sum(
            1 for r in expenses if (r.get("category") or "Uncategorized") == "Uncategorized"
        ),
    }


def parse_receipt_text(
    text: str, *, user_id: str | None = None, db: Session | None = None
) -> list[dict[str, Any]]:
    """Parse forwarded bank SMS text, receipt text, or one statement row (one per line)."""
    lines = _numbered_lines(text) or [(1, _normalize_line(text or ""))]
    result = _parse_sms(lines, {"default_year": _guess_year(text)}, strict=False)
    return _finalise_rows(result.rows, user_id=user_id, db=db)


def _extract_pdf_text(file_bytes: bytes) -> str:
    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(io.BytesIO(file_bytes))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("password")
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:
        if "password" in str(exc).lower() or "decrypt" in str(exc).lower():
            raise ValueError(
                "This PDF is password-protected. Open it, save/print a copy without a password, "
                "and upload that."
            ) from exc
        raise ValueError(
            "Could not read PDF text. Upload a text-based PDF or receipt text."
        ) from exc


def statement_text_from_file(file_bytes: bytes, file_type: str = "pdf") -> tuple[str, str]:
    """(text, source) for an upload; raises ValueError with a user-friendly message."""
    if not file_bytes:
        raise ValueError("The uploaded file is empty.")
    file_type = file_type.lower().lstrip(".")
    if file_type == "pdf":
        text = _extract_pdf_text(file_bytes)
        if not text.strip():
            raise ValueError(
                "This PDF has no selectable text (it may be a scanned image). "
                "Download the statement as a text PDF from your bank or app."
            )
        return text, "pdf"
    if file_type in {"text", "txt", "sms", "receipt"}:
        return file_bytes.decode("utf-8", errors="replace"), "sms"
    raise ValueError("Unsupported file type. Use pdf or text/SMS receipt content.")


MAX_STATEMENT_BYTES = 10_000_000
MAX_LINE_CHARS = 500
MAX_STATEMENT_LINES = 20_000


async def parse_statement_file(
    file_bytes: bytes,
    file_type: str = "pdf",
    *,
    user_id: str | None = None,
    db: Session | None = None,
) -> list[dict[str, Any]]:
    """Parse a PDF statement or text/SMS receipt into normalized transaction rows.

    Each row: date (ISO or None), time, merchant, vpa, type ("expense"/"income"), amount,
    bank_account, reference, balance, category, description, format, note, source.
    """
    if len(file_bytes) > MAX_STATEMENT_BYTES:
        raise ValueError(
            f"that file is too large (more than {MAX_STATEMENT_BYTES // 1_000_000} MB)"
        )
    text, source = statement_text_from_file(file_bytes, file_type)
    # A line longer than a statement line can be is junk (or an attempt to make the parser crawl): skip it
    lines = [ln for ln in text.splitlines() if len(ln) <= MAX_LINE_CHARS][:MAX_STATEMENT_LINES]
    return parse_statement_text("\n".join(lines), source=source, user_id=user_id, db=db).rows
