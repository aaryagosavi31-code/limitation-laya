"""
make_items.py - generate synthetic document-filing items for jev_probe.py.

    python make_items.py --n 100 --out items.jsonl --seed 1

Gold labels come from the generator (not from the text), so they are
independent of wording. About 30% of items are "borderline": they mix cues
from two categories, so the correct answer depends on the primary purpose.
Texts are kept short (<150 tokens) so padding tests fit Laya's context window.
"""
import argparse, json, random

CRIT = {"tax": "Tax documents: returns, withholding, tax notices, filings",
        "invoices": "Invoices: bills requesting payment for goods or services",
        "contracts": "Contracts: agreements defining terms between parties"}
INSTR = "Which folder should this document be filed in?"
CO = ["Northwind", "Apex Logistics", "Bluebird Foods", "Cedar Labs", "Harbor Print"]

def core(kind, rng):
    amt = rng.randrange(200, 9000)
    co = rng.choice(CO)
    if kind == "invoices":
        return f"Invoice #{rng.randrange(1000,9999)} from {co}. Amount due: ${amt:,}. Payment due in 30 days."
    if kind == "contracts":
        return f"Service agreement between {co} and the client. Term: {rng.randrange(1,5)} years. Either party may terminate with 60 days notice."
    return f"Notice of tax assessment for fiscal year {rng.randrange(2021,2026)}. Taxable income reported: ${amt:,}. Filing reference {rng.randrange(10000,99999)}."

DIST = {"tax": ["The company reported taxable income and withholding for the year.",
                "Please retain this notice for your tax filing records.",
                "Estimated tax payments are due each quarter."],
        "invoices": ["Payment is requested within 30 days of the invoice date.",
                     "The total amount due is listed in the attached bill.",
                     "Late payments may incur a service charge."],
        "contracts": ["Both parties agree to the terms set out in this agreement.",
                      "This agreement may be terminated with written notice.",
                      "The obligations of each party are described in the schedule."]}

CROSS = {"invoices": "Please note the attached agreement terms apply.",
         "contracts": "Fees under this agreement will be billed by separate invoice.",
         "tax": "A copy of the related invoice is enclosed for your records."}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="items.jsonl")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    kinds = list(CRIT)
    with open(a.out, "w") as f:
        for i in range(a.n):
            gold = kinds[i % 3]
            border = rng.random() < 0.3
            text = core(gold, rng) + (" " + CROSS[gold] if border else "")
            f.write(json.dumps({"id": f"d{i:03d}", "state": text,
                "instructions": INSTR, "criteria": CRIT, "gold": gold,
                "borderline": border,
                "distractors": [x for k in kinds if k != gold for x in DIST[k]]}) + "\n")

if __name__ == "__main__":
    main()