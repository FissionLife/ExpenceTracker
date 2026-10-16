"""Expense category names and the words that map to them (keeps budgets and expenses on one category)."""

from __future__ import annotations

CATEGORY_WORDS: dict[str, tuple[str, ...]] = {
    "Food & Dining": (
        "chai", "tea", "coffee", "lunch", "dinner", "breakfast", "brunch", "snack", "snacks", "food",
        "meal", "pizza", "burger", "biryani", "dosa", "idli", "idly", "thali", "ice cream", "juice",
        "restaurant", "cafe", "canteen", "mess", "takeaway", "bakery", "sweets", "dessert", "drinks",
        "beer", "bar", "paratha", "samosa", "maggi", "momos", "sandwich", "roll", "dining",
    ),
    "Groceries": (
        "grocery", "groceries", "vegetables", "veggies", "fruits", "milk", "eggs", "rice", "atta",
        "dal", "kirana", "supermarket", "provisions", "bread", "curd",
    ),
    "Transport": (
        "auto", "cab", "taxi", "uber", "ola", "rapido", "metro", "bus", "train", "ticket", "petrol",
        "diesel", "fuel", "toll", "parking", "fastag", "flight", "ride", "commute",
    ),
    "Shopping": (
        "clothes", "shirt", "shoes", "shopping", "dress", "jeans", "bag", "watch", "gift", "laptop",
        "phone", "mobile", "headphones", "gadget", "furniture", "amazon", "flipkart", "myntra",
    ),
    "Entertainment": ("movie", "movies", "cinema", "netflix", "spotify", "hotstar", "prime", "game", "games", "concert", "show", "subscription"),
    "Bills & Utilities": ("electricity", "current bill", "water bill", "gas", "wifi", "internet", "broadband", "recharge", "bill", "postpaid", "dth", "maintenance"),
    "Rent": ("rent",),
    "Health": ("medicine", "medicines", "pharmacy", "doctor", "hospital", "clinic", "gym", "lab test", "dentist", "checkup", "health"),
    "Personal Care": ("haircut", "salon", "spa", "grooming", "shampoo", "parlour", "barber", "facial"),
    "Education": ("course", "tuition", "books", "fees", "exam", "class", "school", "college"),
    "Investments": ("sip", "mutual fund", "stocks", "investment", "invested", "fd", "ppf"),
    "Travel": ("hotel", "trip", "vacation", "holiday", "resort", "booking"),
}  # fmt: skip

EXPENSE_CATEGORIES = list(CATEGORY_WORDS)


def canonical_category(name: str) -> str:
    """Map a bare category word to the name the parser files spends under.

    "food", "meals", "dining" -> "Food & Dining"; "cab" -> "Transport". This keeps a budget on
    the same category the user's expenses land in, even before any such expense exists yet.
    Unknown names are title-cased and returned unchanged.
    """
    key = (name or "").strip().lower()
    if not key:
        return name
    for cat, words in CATEGORY_WORDS.items():
        if key == cat.lower():
            return cat
        for w in words:
            if key == w or key == w + "s" or key + "s" == w:
                return cat
    return name.strip().title()
