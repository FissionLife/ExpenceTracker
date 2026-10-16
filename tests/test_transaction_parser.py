import os

os.environ["WALLETLEDGER_TESTING"] = "1"  # never read the real .env in tests
os.environ["LITELLM_MODE"] = "PRODUCTION"  # litellm would otherwise load .env (real keys!)
import unittest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import Base, Category, Merchant, User
from app.services_ai.transaction_parser import (
    categorize_transaction,
    parse_receipt_text,
)


class TransactionParserTests(unittest.TestCase):
    def test_parses_debit_sms_with_clean_merchant_and_known_brand_category(self):
        rows = parse_receipt_text(
            "02/10/2026 Your A/c XX1234 debited Rs. 450.50 to UPI-SWIGGY-182390@upi ref 123456"
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["amount"], 450.50)
        self.assertEqual(rows[0]["type"], "expense")
        self.assertEqual(rows[0]["merchant"], "Swiggy")
        self.assertEqual(rows[0]["category"], "Food & Dining")  # known Indian brand
        self.assertEqual(rows[0]["date"], "2026-10-02")
        # A merchant nobody knows is still never guessed.
        self.assertEqual(
            parse_receipt_text("Your A/c XX1234 debited Rs. 99 to Zqx Traders")[0]["category"],
            "Uncategorized",
        )

    def test_income_and_unknown_category_are_safe(self):
        rows = parse_receipt_text("Salary credited INR 100000 from Acme Corp")
        self.assertEqual(rows[0]["type"], "income")
        self.assertEqual(rows[0]["category"], "Income")
        self.assertEqual(categorize_transaction("Something New"), "Uncategorized")

    def test_learns_category_from_users_confirmed_merchant_mapping(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session = sessionmaker(bind=engine)()
        user = User(telegram_chat_id="parser-test")
        session.add(user)
        session.flush()
        category = Category(user_id=user.id, name="Entertainment", type="expense")
        session.add(category)
        session.flush()
        merchant = Merchant(user_id=user.id, name="Netflix", default_category_id=category.id)
        session.add(merchant)
        session.flush()
        self.assertEqual(
            categorize_transaction("netflix", user_id=user.id, db=session), "Entertainment"
        )
        session.close()
