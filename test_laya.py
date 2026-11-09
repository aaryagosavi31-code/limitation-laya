from laya import Router

router = Router(preload=False)
q = {"q": {"type": "choice", "instructions": "Which folder?",
           "criteria": {"tax": "Tax docs", "invoices": "Invoices"}}}
print(router.predict({"body": "Invoice #4411, total $2,300"}, q, model="typed-decisions"))