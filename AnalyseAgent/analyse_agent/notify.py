"""Email the report via Gmail SMTP. Needs EMAIL_FROM, EMAIL_PASS (Gmail app password), EMAIL_TO."""
from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage
from pathlib import Path


CSS = """body{font-family:-apple-system,Segoe UI,Microsoft YaHei,sans-serif;font-size:14px;line-height:1.5;color:#222;max-width:900px}
table{border-collapse:collapse;margin:8px 0}th,td{border:1px solid #ccc;padding:4px 8px;text-align:left}
th{background:#f3f3f3}h1{font-size:20px}h2{font-size:17px;margin-top:22px}h3{font-size:15px}code{background:#f3f3f3;padding:0 3px}"""


def to_html(md_text: str) -> str:
    import markdown
    body = markdown.markdown(md_text, extensions=["tables", "sane_lists"])
    return f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"


def send_report(subject: str, body: str, attachments: list[Path]) -> None:
    """body is Markdown; sent as HTML (tables render) with the Markdown as plain-text fallback."""
    sender = os.environ["EMAIL_FROM"]
    password = os.environ["EMAIL_PASS"]
    to = [a.strip() for a in os.environ["EMAIL_TO"].split(",") if a.strip()]

    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, sender, ", ".join(to)
    msg.set_content(body, charset="utf-8")
    msg.add_alternative(to_html(body), subtype="html", charset="utf-8")
    for p in attachments:
        msg.add_attachment(p.read_bytes(), maintype="text", subtype="markdown", filename=p.name)

    with smtplib.SMTP("smtp.gmail.com", 587, timeout=60) as s:
        s.starttls()
        s.login(sender, password)
        s.send_message(msg)
