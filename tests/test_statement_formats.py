"""Statement import: format detection, per-format parsers, merchants, categories, duplicates.

All fixtures are SYNTHETIC (written from the public layouts of PhonePe, Google Pay, Paytm, BHIM
and HDFC/SBI/Axis/Kotak statements); names, numbers and ids are made up. Offline only.
Run with:  uv run python -m unittest tests.test_statement_formats -v
"""

import os

os.environ["WALLETLEDGER_TESTING"] = "1"  # never read the real .env in tests
os.environ["LITELLM_MODE"] = "PRODUCTION"  # litellm would otherwise load .env (real keys!)

import tempfile

_TMP_DIR = tempfile.mkdtemp(prefix="walletledger-statement-test-")
os.environ["DYNAMIC_MODELS"] = "false"
os.environ.setdefault(
    "DATABASE_URL", "sqlite:///" + os.path.join(_TMP_DIR, "test.db").replace("\\", "/")
)
os.environ["TELEGRAM_BOT_TOKEN"] = ""
os.environ["USE_WEBHOOK"] = "false"

import asyncio  # noqa: E402
import io  # noqa: E402
import unittest  # noqa: E402
import uuid  # noqa: E402

from app.bot import main as bot  # noqa: E402
from app.db import init_db  # noqa: E402
from app.services_ai.mcp_server import execute_tool  # noqa: E402

try:
    from tests.fake_llm import FakeLLM  # noqa: E402
except ImportError:  # discover -s tests (no -t .) puts tests/ on sys.path directly
    from fake_llm import FakeLLM  # noqa: E402
from app.services_ai.transaction_parser import (  # noqa: E402
    categorize_transaction,
    detect_statement_format,
    merchant_from_narration,
    parse_receipt_text,
    parse_statement_file,
    parse_statement_text,
    summarize_statement,
)


def make_pdf(pages: list[list[str]]) -> bytes:
    """Minimal multi-page text PDF (one text line per row), no extra dependencies."""
    count = len(pages)
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids ["
        + " ".join(f"{4 + 2 * i} 0 R" for i in range(count))
        + f"] /Count {count} >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, lines in enumerate(pages):
        content = "BT /F1 9 Tf 30 800 Td 12 TL " + " ".join(
            "(" + line.replace("(", "[").replace(")", "]") + ") '" for line in lines
        )
        content += " ET"
        objects.append(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            f"/Contents {5 + 2 * i} 0 R /Resources << /Font << /F1 3 0 R >> >> >>"
        )
        objects.append(f"<< /Length {len(content)} >>\nstream\n{content}\nendstream")
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1"))
    xref = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets:
        out.write(f"{offset:010d} 00000 n \n".encode())
    out.write(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode())
    out.write(f"startxref\n{xref}\n%%EOF\n".encode())
    return out.getvalue()


PHONEPE = """Transaction Statement for 98XXXXXX10
Oct 01, 2025 - Oct 31, 2025
Date Transaction Details Type Amount
Oct 05, 2025 Paid to SWIGGY DEBIT ₹450
10:32 am Transaction ID T2510051032123456789
UTR No. 527812345678
Paid by XXXXXXXX1234
Oct 06, 2025 Received from Rahul Kumar CREDIT ₹1,500
08:15 pm Transaction ID T2510062015123456789
UTR No. 527912345678
Credited to XXXXXXXX1234
Oct 07, 2025 Paid to Ola DEBIT ?236.50
09:01 am Transaction ID T2510070901123456789
UTR No. 528012345678
Paid by XXXXXXXX1234
Page 1 of 2
This is a system generated statement. For any queries, contact us at https://support.phonepe.com/statement.
Transaction Statement for 98XXXXXX10
Date Transaction Details Type Amount
Oct 08, 2025 Paid to Ramesh Kumar DEBIT ₹200
11:45 am Transaction ID T2510081145123456789
UTR No. 528112345678
Paid by XXXXXXXX1234
Oct 09, 2025 Paid to Zomato DEBIT ₹300
11:00 am Transaction ID T2510091100123456789
Payment Failed
Oct 10, 2025 Mobile recharged 98XXXXXX10 DEBIT ₹299
07:30 am Transaction ID T2510100730123456789
Paid by XXXXXXXX1234
Oct 11, 2025 Refund from Amazon CREDIT ₹999
04:00 pm Transaction ID T2510111600123456789
Oct 12, 2025 Paid to 9876543210@ybl DEBIT ₹100
06:00 pm Transaction ID T2510121800123456789
UTR No. 528312345678
Paid by XXXXXXXX1234
Page 2 of 2
"""

GPAY = """Google Pay
Transaction statement
Statement period: 01 Oct 2025 - 31 Oct 2025
Date & time Transaction details Amount
01 Oct, 2025
09:41 AM
Paid to Zepto
UPI Transaction ID: 527412345678
Paid by HDFC Bank 1234
₹612.00
02 Oct, 2025
06:10 PM
Received from Priya Sharma
UPI Transaction ID: 527512345678
Received in HDFC Bank 1234
₹2,000.00
03 Oct, 2025
08:00 PM
Paid to Netflix
UPI Transaction ID: 527612345678
Paid by HDFC Bank 1234
₹649.00
04 Oct, 2025
12:15 PM
Paid to HOLA STORE
UPI Transaction ID: 527712345678
Paid by HDFC Bank 1234
₹120.00
05 Oct, 2025
03:20 PM
Paid to Coca Cola Vending
UPI Transaction ID: 527812345678
Paid by HDFC Bank 1234
₹40.00
"""

PAYTM = """Paytm UPI Statement
01 Oct 2025 - 31 Oct 2025
Date & Time Transaction Details Notes & Tags Your Account Amount
05 Oct 10:32 AM Paid to Swiggy Instamart UPI ID: swiggyinstamart@ybl UPI Ref No: 527812345678 HDFC Bank - 12 - Rs.845
06 Oct 07:00 PM Money Received from Anil Verma UPI Ref No: 527912345678 HDFC Bank - 12 + Rs.5,000
07 Oct 01:15 PM Money Sent to 9812345678 UPI Ref No: 528012345678 HDFC Bank - 12 - Rs.300
08 Oct 09:00 AM Paid to BESCOM UPI Ref No: 528112345678 Paytm Wallet - Rs.1,240.50
09 Oct 09:30 AM Paid to Uber India Failed UPI Ref No: 528212345678 HDFC Bank - 12 - Rs.180
"""

BHIM = """BHIM UPI
Transaction History
05/10/2025 10:32:11
Debit ₹450.00
To: SWIGGY (swiggy@ybl)
UPI Ref No: 527812345678
Status: SUCCESS
06/10/2025 18:20:45
Credit ₹1,200.00
From: rohan.mehta@okaxis
UPI Ref No: 527912345678
Status: SUCCESS
07/10/2025 09:00:00
Debit ₹500.00
To: dmart@ybl
UPI Ref No: 528012345678
Status: FAILED
"""

# HDFC style: Withdrawal/Deposit columns collapse to "amount balance" in extracted text, the
# narration wraps onto the next line, the header repeats on page 2 and the opening balance is
# only in the summary at the end.
HDFC = """HDFC BANK Ltd.
Statement of account
Account No : 50100123456789
Statement From : 01/10/2025 To : 31/10/2025
Date Narration Chq./Ref.No. Value Dt Withdrawal Amt. Deposit Amt. Closing Balance
01/10/25 UPI-SWIGGY-SWIGGY@YBL-YESB0YB 0000527412345678 01/10/25 450.00 9,550.00
LUPI-527412345678-PAYMENT
02/10/25 NEFT CR-ICIC0000104-ACME TECHNOLOGIES PVT LTD-SALARY OCT 0000ICICN52025 02/10/25 75,000.00 84,550.00
03/10/25 ATW-512345XXXXXX1234-S1ANBG12-BANGALORE 0000003456 03/10/25 2,000.00 82,550.00
Page 1 of 2
HDFC BANK Ltd.
Date Narration Chq./Ref.No. Value Dt Withdrawal Amt. Deposit Amt. Closing Balance
04/10/25 POS 416021XXXXXX1234 DECATHLON SPORTS 0000123456 04/10/25 3,499.00 79,051.00
05/10/25 IMPS-527812345678-RAHUL SHARMA-HDFC-XXXXXXX1234-DINNER 0000527812345678 05/10/25 800.00 78,251.00
06/10/25 UPI-ZOMATO-zomato@hdfcbank-HDFC0000001-527912345678-ZOMATO ORDER 0000527912345678 06/10/25 350.00 77,901.00
07/10/25 REV-UPI-ZOMATO-zomato@hdfcbank-527912345678 0000527912345678 07/10/25 350.00 78,251.00
08/10/25 ACH D- TP ACH BAJAJ FINANCE LTD-123456789 0000123456789 08/10/25 5,500.00 72,751.00
09/10/25 INT.PD:01-07-2025 TO 30-09-2025 09/10/25 123.00 72,874.00
STATEMENT SUMMARY :-
Opening Balance Dr Count Cr Count Debits Credits Closing Bal
10,000.00 6 3 12,599.00 75,473.00 72,874.00
Page 2 of 2
"""

# SBI style: "Balance as on" opening line, dates like "3 Oct 2025", narration over 2-3 lines.
SBI = """State Bank of India
Account Name : Mr. Ravi Kumar
Account Number : 00000012345678901
Txn Date Value Date Description Ref No./Cheque No. Debit Credit Balance
Balance as on 1 Oct 2025 5,000.00
3 Oct 2025 3 Oct 2025 TO TRANSFER- TRANSFER TO 450.00 4,550.00
UPI/DR/527812345678/BLINKIT/YESB/blinkit@ybl/
Payment
5 Oct 2025 5 Oct 2025 BY TRANSFER- 1,000.00 5,550.00
UPI/CR/527912345678/PRIYA SH/SBIN/priya.sh@oksbi/UPI
7 Oct 2025 7 Oct 2025 ATM WDL ATM CASH 2,000.00 3,550.00
8 Oct 2025 8 Oct 2025 TO TRANSFER- 1,100.00 2,450.00
UPI/DR/528012345678/BESCOM/UTIB/bescom@axisbank/Electricity
**This is a computer generated statement and does not require a signature
"""

# Axis style, newest first: only the running balance tells debit from credit.
AXIS = """Axis Bank Ltd
Tran Date Chq No Particulars Debit Credit Balance Init. Br
10-10-2025 UPI/P2M/528112345678/AMAZON PAY INDIA/YES BANK/Payme 1,299.00 18,701.00 248
08-10-2025 UPI/P2A/528012345678/SURESH/HDFC BANK/rent 15,000.00 20,000.00 248
05-10-2025 NEFT/ACME LABS PVT LTD/SALARY 30,000.00 35,000.00 248
"""

# Kotak style: explicit (Dr)/(Cr) suffixes.
KOTAK = """Kotak Mahindra Bank
Date Narration Chq/Ref No Withdrawal (Dr) Deposit (Cr) Balance
01-Oct-2025 UPI/BIGBASKET/527412345678/Payment UPI-527412345678 1,250.00(Dr) 8,750.00(Cr)
02-Oct-2025 UPI/RAPIDO/527512345678/Ride UPI-527512345678 89.00(Dr) 8,661.00(Cr)
03-Oct-2025 IMPS/P2A/527612345678/ANITA DESAI/Gift 2,500.00(Cr) 11,161.00(Cr)
"""

SMS = """Rs.450.00 debited from A/c XX1234 on 05-10-25 to VPA swiggy@ybl (UPI Ref No 527812345678). Not you? Call 18002586161
Sent Rs.220.00 From HDFC Bank A/C *1234 To Uber On 06/10/25 Ref 527912345678 Not You? Call 18002586161/SMS BLOCK UPI to 7308080808
ICICI Bank Acct XX123 debited for Rs 1,999.00 on 07-Oct-25; MYNTRA credited. UPI:528012345678. Call 18002662 for dispute.
Your a/c no. XXXXXXXX1234 is credited by Rs.50,000.00 on 08-10-2025 by a/c linked to VPA acme.payroll@icici (UPI Ref no 528112345678).
Your OTP for txn of Rs 999 at Amazon is 123456. Do not share it with anyone.
Txn of Rs 300.00 at Zomato on 09-10-25 failed due to insufficient balance.
"""


class FormatDetectionTests(unittest.TestCase):
    def test_detects_each_layout(self):
        cases = {
            "phonepe": PHONEPE,
            "gpay": GPAY,
            "paytm": PAYTM,
            "bhim": BHIM,
            "bank_table": HDFC,
            "sms": SMS,
        }
        for expected, text in cases.items():
            with self.subTest(expected):
                self.assertEqual(detect_statement_format(text), expected)
        for text in (SBI, AXIS, KOTAK):
            self.assertEqual(detect_statement_format(text), "bank_table")
        self.assertEqual(detect_statement_format(""), "generic")
        self.assertEqual(detect_statement_format("hello world"), "generic")

    def test_bank_statement_mentioning_phonepe_is_still_a_bank_table(self):
        text = HDFC.replace("SWIGGY@YBL", "PHONEPE@YBL")
        self.assertEqual(detect_statement_format(text), "bank_table")


def by_merchant(rows):
    return {row["merchant"]: row for row in rows}


class UpiAppStatementTests(unittest.TestCase):
    def test_phonepe(self):
        result = parse_statement_text(PHONEPE)
        self.assertEqual(result.format, "phonepe")
        rows = by_merchant(result.rows)
        self.assertEqual(
            sorted(rows),
            sorted(
                [
                    "Swiggy",
                    "Rahul Kumar",
                    "Ola",
                    "Ramesh Kumar",
                    "Mobile Recharge",
                    "Amazon",
                    "9876543210@ybl",
                ]
            ),
        )
        swiggy = rows["Swiggy"]
        self.assertEqual(
            (swiggy["date"], swiggy["amount"], swiggy["type"], swiggy["category"]),
            ("2025-10-05", 450.0, "expense", "Food & Dining"),
        )
        self.assertEqual(swiggy["reference"], "527812345678")
        self.assertEqual(swiggy["bank_account"], "XXXXXXXX1234")
        self.assertEqual(rows["Rahul Kumar"]["type"], "income")
        self.assertEqual(rows["Ola"]["amount"], 236.5)  # rupee sign extracted as "?"
        self.assertEqual(rows["Ola"]["category"], "Transport")
        self.assertEqual(rows["Ramesh Kumar"]["category"], "Uncategorized")
        self.assertEqual(rows["Mobile Recharge"]["category"], "Bills & Utilities")
        self.assertEqual(rows["Amazon"]["type"], "income")  # refund
        self.assertEqual(rows["9876543210@ybl"]["category"], "Transfers")
        self.assertNotIn("Zomato", rows)  # failed payment skipped
        reasons = {reason for _, reason, _ in result.skipped}
        self.assertIn("failed/pending transaction", reasons)
        self.assertIn("page number", reasons)
        self.assertTrue(all(row["format"] == "phonepe" for row in result.rows))

    def test_gpay_one_field_per_line(self):
        result = parse_statement_text(GPAY)
        rows = by_merchant(result.rows)
        self.assertEqual(len(result.rows), 5)
        self.assertEqual(rows["Zepto"]["amount"], 612.0)
        self.assertEqual(rows["Zepto"]["category"], "Groceries")
        self.assertEqual(rows["Zepto"]["date"], "2025-10-01")
        self.assertEqual(rows["Priya Sharma"]["type"], "income")
        self.assertEqual(rows["Netflix"]["category"], "Entertainment")
        # "ola" must not match inside "Hola" or "Cola".
        self.assertEqual(rows["Hola Store"]["category"], "Uncategorized")
        self.assertEqual(rows["Coca Cola Vending"]["category"], "Uncategorized")

    def test_paytm_signs_and_yearless_dates(self):
        result = parse_statement_text(PAYTM)
        rows = by_merchant(result.rows)
        self.assertEqual(sorted(rows), ["9812345678", "Anil Verma", "Bescom", "Swiggy Instamart"])
        self.assertEqual(rows["Swiggy Instamart"]["category"], "Groceries")
        self.assertEqual(rows["Swiggy Instamart"]["date"], "2025-10-05")
        self.assertEqual(rows["Anil Verma"]["type"], "income")
        self.assertEqual(rows["Anil Verma"]["amount"], 5000.0)
        self.assertEqual(rows["9812345678"]["category"], "Transfers")
        self.assertEqual(rows["Bescom"]["amount"], 1240.5)
        self.assertEqual(rows["Bescom"]["category"], "Bills & Utilities")

    def test_bhim(self):
        result = parse_statement_text(BHIM)
        rows = by_merchant(result.rows)
        self.assertEqual(sorted(rows), ["Rohan Mehta", "Swiggy"])
        self.assertEqual(rows["Swiggy"]["vpa"], "swiggy@ybl")
        self.assertEqual(rows["Swiggy"]["reference"], "527812345678")
        self.assertEqual(rows["Rohan Mehta"]["type"], "income")


class BankTableTests(unittest.TestCase):
    def test_hdfc_balance_delta_wrapped_narration_and_page_headers(self):
        result = parse_statement_text(HDFC)
        self.assertEqual(result.format, "bank_table")
        got = [
            (r["date"], r["merchant"], r["type"], r["amount"], r["category"]) for r in result.rows
        ]
        self.assertEqual(
            got,
            [
                ("2025-10-01", "Swiggy", "expense", 450.0, "Food & Dining"),
                ("2025-10-02", "Acme Technologies Pvt Ltd", "income", 75000.0, "Income"),
                ("2025-10-03", "ATM Withdrawal", "expense", 2000.0, "Cash"),
                ("2025-10-04", "Decathlon Sports", "expense", 3499.0, "Shopping"),
                ("2025-10-05", "Rahul Sharma", "expense", 800.0, "Transfers"),
                ("2025-10-06", "Zomato", "expense", 350.0, "Food & Dining"),
                ("2025-10-07", "Zomato", "income", 350.0, "Income"),
                ("2025-10-08", "Bajaj Finance Ltd", "expense", 5500.0, "Loan/EMI"),
                ("2025-10-09", "Interest", "income", 123.0, "Income"),
            ],
        )
        self.assertTrue(all(r["note"] is None for r in result.rows), "all from balance delta")
        self.assertEqual(result.rows[0]["vpa"], "swiggy@ybl")
        self.assertEqual(result.rows[0]["balance"], 9550.0)
        summary = summarize_statement(result.rows)
        self.assertEqual(summary["total_debit"], 12599.0)
        self.assertEqual(summary["total_credit"], 75473.0)
        reasons = {reason for _, reason, _ in result.skipped}
        self.assertTrue(
            {"table header", "page number", "statement summary", "account details"} <= reasons
        )
        self.assertNotIn("not a transaction (no date)", reasons)

    def test_sbi_opening_balance_and_multiline_narration(self):
        rows = parse_statement_text(SBI).rows
        got = [(r["date"], r["merchant"], r["type"], r["amount"], r["category"]) for r in rows]
        self.assertEqual(
            got,
            [
                ("2025-10-03", "Blinkit", "expense", 450.0, "Groceries"),
                ("2025-10-05", "Priya Sh", "income", 1000.0, "Income"),
                ("2025-10-07", "ATM Withdrawal", "expense", 2000.0, "Cash"),
                ("2025-10-08", "Bescom", "expense", 1100.0, "Bills & Utilities"),
            ],
        )
        self.assertEqual(rows[0]["vpa"], "blinkit@ybl")

    def test_axis_newest_first(self):
        rows = parse_statement_text(AXIS).rows
        got = [(r["merchant"], r["type"], r["amount"], r["category"]) for r in rows]
        self.assertEqual(
            got,
            [
                ("Amazon Pay India", "expense", 1299.0, "Shopping"),
                ("Suresh", "expense", 15000.0, "Rent"),
                ("Acme Labs Pvt Ltd", "income", 30000.0, "Income"),
            ],
        )

    def test_kotak_dr_cr_suffix(self):
        rows = parse_statement_text(KOTAK).rows
        got = [(r["merchant"], r["type"], r["amount"], r["category"]) for r in rows]
        self.assertEqual(
            got,
            [
                ("Bigbasket", "expense", 1250.0, "Groceries"),
                ("Rapido", "expense", 89.0, "Transport"),
                ("Anita Desai", "income", 2500.0, "Income"),
            ],
        )


class SmsAndGenericTests(unittest.TestCase):
    def test_forwarded_bank_sms(self):
        result = parse_statement_text(SMS, source="sms")
        self.assertEqual(result.format, "sms")
        got = [(r["merchant"], r["type"], r["amount"], r["category"]) for r in result.rows]
        self.assertEqual(
            got,
            [
                ("Swiggy", "expense", 450.0, "Food & Dining"),
                ("Uber", "expense", 220.0, "Transport"),
                ("Myntra", "expense", 1999.0, "Shopping"),
                ("Acme Payroll", "income", 50000.0, "Income"),
            ],
        )
        self.assertEqual(result.rows[0]["date"], "2025-10-05")
        self.assertEqual(result.rows[2]["date"], "2025-10-07")
        reasons = [reason for _, reason, _ in result.skipped]
        self.assertIn("OTP message", reasons)
        self.assertIn("failed/pending transaction", reasons)

    def test_generic_lines(self):
        text = (
            "02/10/2026 Paid to Swiggy Rs. 450.50 debited\n05/10/2026 Salary credited Rs. 50,000.00"
        )
        result = parse_statement_text(text)
        self.assertEqual(result.format, "generic")
        self.assertEqual([r["amount"] for r in result.rows], [450.5, 50000.0])

    def test_receipt_text_still_one_row_per_line(self):
        rows = parse_receipt_text("Paid 120 to Chai Point via UPI\nnothing here")
        self.assertEqual(len(rows), 1)


class MerchantAndCategoryTests(unittest.TestCase):
    def test_narration_merchants(self):
        cases = {
            "UPI/DR/527812345678/SWIGGY/YESB/swiggy@ybl/Payment": "Swiggy",
            "UPI-ZOMATO-zomato@hdfcbank-HDFC0000001-527912345678-ZOMATO ORDER": "Zomato",
            "UPI/527812345678/Payment from Ph/swiggy@ybl/YES BANK LIMITED/x": "Swiggy",
            "UPI/P2M/527812345678/SWIGGY /YES BANK/Payme": "Swiggy",
            "NEFT-HDFCN52025100512345678-ACME CORP PVT LTD-SALARY": "Acme Corp Pvt Ltd",
            "IMPS-527812345678-RAHUL SHARMA-HDFC-XXXXXXXX1234": "Rahul Sharma",
            "ATM WDL SBI ATM MG ROAD": "ATM Withdrawal",
            "NWD-512345XXXXXX1234-S1ANBG12-BANGALORE": "ATM Withdrawal",
            "POS 4XXXXXXXXXXX1234 SWIGGY BANGALORE": "Swiggy Bangalore",
            "EMI 123456789 CHQ S123456": "Loan EMI",
            "INT.PD:01-07-2025 TO 30-09-2025": "Interest",
            "UPI/DR/528012345678/9876543210/YESB/9876543210@ybl/UPI": "9876543210@ybl",
        }
        for narration, expected in cases.items():
            with self.subTest(narration):
                self.assertEqual(merchant_from_narration(narration)[0], expected)

    def test_categories(self):
        cases = [
            ("Swiggy", "", "Food & Dining"),
            ("Domino's Pizza", "", "Food & Dining"),
            ("Swiggy Instamart", "", "Groceries"),
            ("OLA CABS", "", "Transport"),
            ("Hola", "", "Uncategorized"),
            ("Coca Cola", "", "Uncategorized"),
            ("HPCL Petrol Pump", "", "Transport"),
            ("Amazon Prime", "", "Entertainment"),
            ("Amazon", "", "Shopping"),
            ("Airtel", "", "Bills & Utilities"),
            ("Apollo Pharmacy", "", "Health"),
            ("LIC", "", "Insurance"),
            ("Zerodha", "", "Investments"),
            ("IndiGo", "", "Travel"),
            ("ATM Withdrawal", "", "Cash"),
            ("Something New", "", "Uncategorized"),
            # Person-to-person: never guessed as Food from a "dinner" remark.
            ("Rahul Sh", "UPI/DR/1/RAHUL SH/SBIN/rahul.sh@oksbi/dinner", "Transfers"),
            ("Suresh", "UPI/P2A/528012345678/SURESH/HDFC BANK/rent", "Rent"),
            ("Ramesh Kumar", "Paid to Ramesh Kumar", "Uncategorized"),
        ]
        for merchant, description, expected in cases:
            with self.subTest(merchant):
                self.assertEqual(categorize_transaction(merchant, description), expected)
        self.assertEqual(categorize_transaction("Refund", "refund", "income"), "Income")

    def test_summary(self):
        rows = parse_statement_text(PHONEPE).rows
        summary = summarize_statement(rows)
        self.assertEqual(summary["format"], "phonepe")
        self.assertEqual(summary["format_label"], "PhonePe")
        self.assertEqual(summary["count"], 7)
        self.assertEqual(summary["income_count"], 2)
        self.assertEqual(summary["date_from"], "2025-10-05")
        self.assertEqual(summary["date_to"], "2025-10-12")
        self.assertEqual(summary["period"], "05 Oct – 12 Oct 2025")
        self.assertEqual(summary["categories"][0], ("Food & Dining", 450.0))


class FileAndImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()

    def setUp(self):
        self.chat_id = str(uuid.uuid4().int)[:9]
        bot._pending_imports.clear()

    def bank_balance(self) -> float:
        accounts = execute_tool("get_pipe_balances", user_id=self.chat_id)["accounts"]
        return {a["name"]: a["balance"] for a in accounts}.get("Bank", 0.0)

    def test_multi_page_pdf_with_repeated_headers(self):
        page1 = [
            "HDFC BANK Ltd.",
            "Date Narration Chq./Ref.No. Value Dt Withdrawal Amt. Deposit Amt. Closing Balance",
            "Opening Balance 10,000.00",
            "01/10/25 UPI-SWIGGY-SWIGGY@YBL-YESB0YBLUPI-527412345678 01/10/25 450.00 9,550.00",
            "02/10/25 UPI-UBER-UBER@ICICI-ICIC0DC0099-527512345678 02/10/25 230.00 9,320.00",
            "Page 1 of 2",
        ]
        page2 = [
            "HDFC BANK Ltd.",
            "Date Narration Chq./Ref.No. Value Dt Withdrawal Amt. Deposit Amt. Closing Balance",
            "03/10/25 NEFT CR-ICIC0000104-ACME CORP-SALARY 03/10/25 40,000.00 49,320.00",
            "Page 2 of 2",
        ]
        rows = asyncio.run(parse_statement_file(make_pdf([page1, page2]), "pdf"))
        got = [(r["merchant"], r["type"], r["amount"], r["category"]) for r in rows]
        self.assertEqual(
            got,
            [
                ("Swiggy", "expense", 450.0, "Food & Dining"),
                ("Uber", "expense", 230.0, "Transport"),
                ("Acme Corp", "income", 40000.0, "Income"),
            ],
        )
        self.assertTrue(all(r["source"] == "pdf" and r["format"] == "bank_table" for r in rows))

    def test_empty_and_garbage_files(self):
        with self.assertRaises(ValueError):
            asyncio.run(parse_statement_file(b"", "pdf"))
        with self.assertRaises(ValueError):
            asyncio.run(parse_statement_file(b"%PDF-1.4 garbage", "pdf"))
        with self.assertRaises(ValueError):
            asyncio.run(parse_statement_file(make_pdf([[]]), "pdf"))  # no text at all
        no_rows = make_pdf([["Statement of account", "No transactions in this period", "Page 1"]])
        self.assertEqual(asyncio.run(parse_statement_file(no_rows, "pdf")), [])

    def test_preview_is_short_and_shows_format(self):
        pdf = make_pdf(
            [
                [
                    "Transaction Statement for 98XXXXXX10",
                    "Date Transaction Details Type Amount",
                    "Oct 05, 2025 Paid to SWIGGY DEBIT INR 450",
                    "Transaction ID T2510051032123456789 UTR No. 527812345678",
                    "Paid by XXXXXXXX1234",
                    "Oct 06, 2025 Received from Rahul Kumar CREDIT INR 1,500",
                    "This is a system generated statement - support.phonepe.com",
                ]
            ]
        )
        reply = asyncio.run(bot.handle_document(self.chat_id, "phonepe.pdf", pdf))
        self.assertIn("(PhonePe)", reply)
        self.assertIn("Found **2 transactions**", reply)
        self.assertIn("Swiggy (Food & Dining)", reply)
        self.assertLessEqual(len(reply.splitlines()), 14)

    def test_same_statement_imported_twice_is_not_double_counted(self):
        pdf = make_pdf([[line for line in HDFC.splitlines() if line.isascii()]])
        first = asyncio.run(bot.handle_document(self.chat_id, "hdfc.pdf", pdf))
        self.assertIn("Found **9 transactions**", first)
        done = asyncio.run(
            bot.answer_pending_import(self.chat_id, "apply")
        ).text  # history + balance
        self.assertIn("Imported 9 transactions", done)
        balance = self.bank_balance()
        self.assertAlmostEqual(balance, 75473.0 - 12599.0)

        asyncio.run(bot.handle_document(self.chat_id, "hdfc-again.pdf", pdf))
        with FakeLLM("confirm"):  # the typed yes is classified by the AI
            again = asyncio.run(bot.handle_text(self.chat_id, "yes"))
        self.assertIn("Imported 0 transactions", again)
        self.assertIn("Skipped 9", again)
        self.assertAlmostEqual(self.bank_balance(), balance)


if __name__ == "__main__":
    unittest.main()
