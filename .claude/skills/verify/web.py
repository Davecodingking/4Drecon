#!/usr/bin/env python
"""当一回员工: 只通过网页点。用法:
  web.sh new <indoor|outdoor> <场景> <有人0/1> <png>     在 /new 页选好、点"开始"
  web.sh snap <url> <png>                                打开页面, 打印页面上看得到的状态, 截图
  web.sh go <url> <png>                                  在本轮页点"继续"
  web.sh dl <url> <保存目录>                             点本轮页所有下载链接
"""
import sys, time, os
from playwright.sync_api import sync_playwright
B = "http://127.0.0.1:8765"

def status(page):
    page.wait_for_function("document.getElementById('nowbig') && document.getElementById('nowbig').textContent.length > 1", timeout=20000)
    g = lambda s: page.eval_on_selector(s, "e => e.innerText") if page.query_selector(s) else ""
    print("标题:", g("#title"), "|", g("#kind"), "|", g("#cfgpill"))
    print("现在:", g("#nowbig"))
    print("说明:", g("#nowsub"))
    print("按钮:", g("#go"), "(禁用)" if page.eval_on_selector("#go", "e => e.disabled") else "(可点)")
    print("提示:", g("#gomsg"))
    print("进度:", " / ".join(t.replace("\n", " ") for t in page.eval_on_selector_all("#steps .st", "es => es.map(e => e.innerText)")))

with sync_playwright() as p:
    b = p.chromium.launch()
    ctx = b.new_context(viewport=dict(width=1280, height=900), accept_downloads=True)
    page = ctx.new_page()
    logs = []
    page.on("console", lambda m: logs.append(f"[console.{m.type}] {m.text}") if m.type in ("error", "warning") else None)
    page.on("pageerror", lambda e: logs.append(f"[pageerror] {e}"))
    page.on("requestfailed", lambda r: logs.append(f"[请求失败] {r.url}"))
    page.on("response", lambda r: logs.append(f"[HTTP {r.status}] {r.url}") if r.status >= 400 else None)
    cmd = sys.argv[1]
    if cmd == "new":
        kind, scene, people, png = sys.argv[2:6]
        page.goto(B + "/new"); page.wait_for_function("document.getElementById('scene').options.length > 0")
        page.check(f"input[name=k][value={kind}]")
        page.wait_for_timeout(300)
        print("场景下拉框:", page.eval_on_selector_all("#scene option", "es => es.map(e => e.textContent)"))
        page.select_option("#scene", scene)
        page.wait_for_timeout(200)
        print("页面给的轮次名:", page.input_value("#run"))
        if people == "1": page.check("#people")
        page.screenshot(path=png.replace(".png", "_before.png"), full_page=True)
        page.click("#go")
        try:
            page.wait_for_url("**/run?**", timeout=90000)
            print("跳到:", page.url); status(page)
        except Exception as e:
            print("没跳转:", page.inner_text("#msg"))
        page.screenshot(path=png, full_page=True)
    elif cmd == "snap":
        url, png = sys.argv[2:4]
        page.goto(B + url if url.startswith("/") else url)
        status(page); page.wait_for_timeout(1500)
        for d in page.query_selector_all("details.seg[open]"):
            print("--- 展开的段:", d.inner_text()[:int(os.environ.get("N", "1500"))])
        page.screenshot(path=png, full_page=True)
    elif cmd == "go":
        url, png = sys.argv[2:4]
        page.goto(B + url); status(page)
        page.on("dialog", lambda d: (print("弹窗:", d.message), d.accept()))
        page.click("#go"); page.wait_for_timeout(4000)
        print("== 点了之后"); status(page)
        page.screenshot(path=png, full_page=True)
    elif cmd == "dl":
        url, out = sys.argv[2:4]
        page.goto(B + url); status(page); page.wait_for_timeout(1000)
        links = page.query_selector_all("a.dl")
        print("下载链接数:", len(links))
        for a in links:
            print(" 链接文字:", a.inner_text().replace("\n", " "))
            with page.expect_download(timeout=120000) as di: a.click()
            d = di.value; dst = os.path.join(out, d.suggested_filename); d.save_as(dst)
            print("  存到", dst, os.path.getsize(dst), "字节, 开头", open(dst, "rb").read(20))
    for l in logs: print(l)
    b.close()
