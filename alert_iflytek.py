"""
科大讯飞（002230.SZ）价格预警
每天运行一次，收盘价 > 50 元时发送邮件通知。
用法：python alert_iflytek.py
# 用github
git add alert_iflytek.py
git commit -m "lower alert threshold to 48"
git push

定时：crontab -e  加入  # 本地
TZ=Asia/Shanghai
0 16 * * 1-5 /opt/anaconda3/envs/qts/bin/python /Users/litingmai/Desktop/AQuant/QuantifyA/alert_iflytek.py >> /tmp/alert_iflytek.log 2>&1
"""

import os
import tushare as ts
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

# ── 配置区 ────────────────────────────────────────────────────────────────
TS_TOKEN    = '751c24755913f41ceb47461453aafde3a1e0e3e038ec11f6faa2efdf'
STOCK_CODE  = '002230.SZ'   # 科大讯飞
THRESHOLD   = 48.0           # 触发价格（元）

SMTP_HOST   = 'smtp.gmail.com'
SMTP_PORT   = 587
EMAIL_FROM  = os.environ['EMAIL_FROM']   # GitHub Secret: EMAIL_FROM
EMAIL_PASS  = os.environ['EMAIL_PASS']   # GitHub Secret: EMAIL_PASS
EMAIL_TO    = 'mlt17071348@gmail.com'
# ─────────────────────────────────────────────────────────────────────────


def get_latest_price(token: str, ts_code: str) -> tuple[float, str]:
    """从 tushare 获取最新收盘价，返回 (price, trade_date)"""
    pro = ts.pro_api(token)
    df  = pro.daily(ts_code=ts_code, limit=1)
    if df.empty:
        raise RuntimeError(f'tushare 未返回数据：{ts_code}')
    row   = df.iloc[0]
    price = float(row['close'])
    date  = str(row['trade_date'])
    return price, date


def send_email(subject: str, body: str) -> None:
    msg = MIMEText(body, 'plain', 'utf-8')
    msg['Subject'] = subject
    msg['From']    = EMAIL_FROM
    msg['To']      = EMAIL_TO

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.ehlo()
        server.starttls()
        server.login(EMAIL_FROM, EMAIL_PASS)
        server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())


def main():
    now   = datetime.now().strftime('%Y-%m-%d %H:%M')
    price, trade_date = get_latest_price(TS_TOKEN, STOCK_CODE)

    print(f'[{now}] {STOCK_CODE} 收盘价 = {price:.2f}  交易日 = {trade_date}')

    if price > THRESHOLD:
        subject = f'【价格预警】科大讯飞 {price:.2f} 元 > {THRESHOLD} 元'
        body    = (
            f'科大讯飞（{STOCK_CODE}）价格预警\n\n'
            f'交易日期：{trade_date}\n'
            f'收盘价格：{price:.2f} 元\n'
            f'触发阈值：{THRESHOLD} 元\n\n'
            f'预警时间：{now}'
        )
        send_email(subject, body)
        print(f'[ALERT] 已发送邮件至 {EMAIL_TO}')
    else:
        print(f'[OK] 未触发预警（阈值 {THRESHOLD} 元）')


if __name__ == '__main__':
    main()
