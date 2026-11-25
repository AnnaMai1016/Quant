"""Email the report via Gmail SMTP. Needs EMAIL_FROM, EMAIL_PASS (Gmail app password), EMAIL_TO."""
from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage
from pathlib import Path


def send_report(subject: str, body: str, attachments: list[Path]) -> None:
    sender = os.environ["EMAIL_FROM"]
    password = os.environ["EMAIL_PASS"]
    to = [a.strip() for a in os.environ["EMAIL_TO"].split(",") if a.strip()]

    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, sender, ", ".join(to)
    msg.set_content(body, charset="utf-8")
    for p in attachments:
        msg.add_attachment(p.read_bytes(), maintype="text", subtype="markdown", filename=p.name)

    with smtplib.SMTP("smtp.gmail.com", 587, timeout=60) as s:
        s.starttls()
        s.login(sender, password)
        s.send_message(msg)
