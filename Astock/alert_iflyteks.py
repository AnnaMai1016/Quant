"""
科大讯飞（002230.SZ）价格预警
每天运行一次，收盘价 > 50 元时发送邮件通知。
用法：python alert_iflytek.py
# 用github
cd /Users/litingmai/Desktop/AQuant/QuantifyA
git add alert_iflytek.py
git commit -m "lower alert threshold to 48"
git push

open https://github.com/AnnaMai1016/Quant/actions
左边点「科大讯飞价格预警」
右边点 Run workflow → Run workflow

wcaa ppmg ejuw yptq  # google pass
kjni djbi leiy fraa  # google pass

# 本地
crontab -e # 然后写
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
# 蓝筹股
stock_names = [
    # 科技
    "立讯精密", "中兴通讯", "歌尔股份", "科大讯飞", 
    # 能源
    "中国神华", "紫金矿业", "通威股份",
    # 电力
    "长江电力", "华能水电", "中国广核",
    # 半导体
    "中芯国际", "北方华创", "中微公司",
    # 机械制造
    "三一重工", "潍柴动力", "中国中车"
]
stock_codes = [
    # 科技
    "002475.SZ", "000063.SZ", "002241.SZ",
    # 能源
    "601088.SH", "601899.SH", "600438.SH",
    # 电力
    "600900.SH", "600025.SH", "003816.SZ",
    # 半导体
    "688981.SH", "002371.SZ", "688012.SH",
    # 机械制造
    "600031.SH", "000338.SZ", "601766.SH"
]

stock_by_sector = {
    "科技": ["002475.SZ", "000063.SZ", "002241.SZ", '002230.SZ'],
    "能源": ["601088.SH", "601899.SH", "600438.SH"],
    "电力": ["600900.SH", "600025.SH", "003816.SZ"],
    "半导体": ["688981.SH", "002371.SZ", "688012.SH"],
    "机械制造": ["600031.SH", "000338.SZ", "601766.SH"]
}

THRESHOLDs = {"002475.SZ": #  TakeProfit, StopLoss, TargetPrice

}
50.0           # 触发价格（元）

SMTP_HOST   = 'smtp.gmail.com'
SMTP_PORT   = 587
EMAIL_FROM  = os.environ['EMAIL_FROM']   # GitHub Secret: EMAIL_FROM
EMAIL_PASS  = os.environ['EMAIL_PASS']   # GitHub Secret: EMAIL_PASS
EMAIL_TO    = 'litingm2@illinois.edu'
# ─────────────────────────────────────────────────────────────────────────


def get_latest_price(token: str, ts_code: str) -> tuple[float, float, float, float, str]:
    """从 tushare 获取最新日线数据，返回 (open, high, low, close, trade_date)"""
    pro = ts.pro_api(token)
    df  = pro.daily(ts_code=ts_code, limit=1)
    if df.empty:
        raise RuntimeError(f'tushare 未返回数据：{ts_code}')
    row = df.iloc[0]
    return float(row['open']), float(row['high']), float(row['low']), float(row['close']), str(row['trade_date'])


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
    # ST监测

    #
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    open_, high, low, close, trade_date = get_latest_price(TS_TOKEN, STOCK_CODE)

    print(f'[{now}] {STOCK_CODE}  开={open_:.2f}  高={high:.2f}  低={low:.2f}  收={close:.2f}  日期={trade_date}')

    # email notice
    if close > THRESHOLD:
        subject = f'【价格预警】科大讯飞 {close:.2f} 元 > {THRESHOLD} 元'
        body    = (
            f'科大讯飞（{STOCK_CODE}）价格预警\n\n'
            f'交易日期：{trade_date}\n'
            f'开盘价格：{open_:.2f} 元\n'
            f'最高价格：{high:.2f} 元\n'
            f'最低价格：{low:.2f} 元\n'
            f'收盘价格：{close:.2f} 元\n'
            f'触发阈值：{THRESHOLD} 元\n\n'
            f'预警时间：{now}'
        )
        send_email(subject, body)
        print(f'[ALERT] 已发送邮件至 {EMAIL_TO}')
    else:
        print(f'[OK] 未触发预警（阈值 {THRESHOLD} 元）')

    # 每月初check基本面&价值因子&BAB因子&低波动因子&质量因子代码target价格调整

if __name__ == '__main__':
    main()
