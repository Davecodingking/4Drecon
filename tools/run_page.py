#!/usr/bin/env python
"""闸口网页的启动页 /new 和本轮页 /run(2026-09-24)。逻辑在 run_ctl.py, 这里只有页面和路由。

gate_ui.py 在每个请求进来时先问 handle(), 认得的路由在这里答完返回 True, 不认得的返回 False 交回去。
和 scene_view.py(3D 页)同一种挂法。

  GET  /new                启动页: 选类型/场景/轮次名, 有没有人, 开不开召回 -> 开跑
  GET  /run?s=&r=[&i=1]    本轮页: 配置(闸 1 之前可改) / 地基图 / 挑人(闸 2) / 开始·继续
  GET  /api/datasets       两个数据集根下的场景 + 下一个没用过的轮次名
  POST /api/new            {类型, 场景, 轮次, 有人, 召回}
  GET  /api/run/state      run_ctl.state()
  POST /api/run/config     {PEOPLE: "0"/"1", RECALL: ...}
  POST /api/run/people     {id: "keep"/"drop"}
  POST /api/run/go         替人敲一次 drive_go.sh
  GET  /run/img?what=ground|check|person(&id=)   只认这几种, 不收任意路径
  GET  /run/video          body4d 的跟踪体检视频 03_track.mp4
  GET  /run/media?what=    每道闸的图/视频(run_ctl.media_paths 的白名单)
"""
import json, os, urllib.parse

import numpy as np, cv2
from PIL import Image

import run_ctl
import people_decisions


def _kind(q):
    return "indoor" if q.get("i") in ("1", "true") else "outdoor"


def _q(kind, scene, run):
    return urllib.parse.urlencode(dict(s=scene, r=run, **({"i": 1} if kind == "indoor" else {})))


def _person_png(R1, pid):
    """挑人卡片的缩略图: 这个人面积最大那帧, 按 mask 外接框放宽 25% 裁出来, 高 180。"""
    d = people_decisions.load(R1)
    row = next((r for r in (d or {}).get("人", []) if str(r["id"]) == str(pid)), None)
    if row is None or row.get("最大面积帧") is None:
        return None
    trk, f = d["跟踪目录"], int(row["最大面积帧"])
    img = cv2.imread(f"{trk}/images/{f:08d}.jpg")
    # ★ mask 是调色板 PNG(像素值 = 人的 id)。cv2.imread 会按调色板把它展开成 3 通道彩色, 按 id 比就错了;
    #   PIL 读调色板图给的是索引本身。
    mp = f"{trk}/masks/{f:08d}.png"
    if img is None or not os.path.exists(mp):
        return None
    m = np.asarray(Image.open(mp))
    ys, xs = np.nonzero(m == int(pid))
    if not len(ys):
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    py, px = int((y1 - y0) * .25) + 4, int((x1 - x0) * .25) + 4
    y0, y1 = max(0, y0 - py), min(img.shape[0], y1 + py)
    x0, x1 = max(0, x0 - px), min(img.shape[1], x1 + px)
    crop = img[y0:y1, x0:x1].copy()
    edge = (m[y0:y1, x0:x1] == int(pid)).astype(np.uint8)
    cnt, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(crop, cnt, -1, (0, 200, 255), 1)
    h = 180
    return cv2.resize(crop, (max(1, round(crop.shape[1] * h / crop.shape[0])), h), interpolation=cv2.INTER_AREA)


def handle(h, method, path, q, body=None):
    """认得的路由答完返回 True。h 是 gate_ui 的请求处理器(用它的 _send / _json)。"""
    if path == "/new" and method == "GET":
        h._send(200, "text/html; charset=utf-8", NEW_PAGE.encode()); return True
    if path == "/run" and method == "GET":
        h._send(200, "text/html; charset=utf-8", RUN_PAGE.encode()); return True
    if path == "/api/datasets" and method == "GET":
        h._json(run_ctl.datasets()); return True
    if path not in ("/api/new", "/api/run/state", "/api/run/config", "/api/run/people",
                    "/api/run/go", "/run/img", "/run/video", "/run/media"):
        return False
    try:
        if path == "/api/new":
            b = body or {}
            kind, scene, run = b.get("类型"), b.get("场景", ""), b.get("轮次", "")
            run_ctl.new_run(kind, scene, run, "1" if b.get("有人") else "0", "1" if b.get("召回") else "0")
            h._json({"ok": True, "url": "/run?" + _q(kind, scene, run)}); return True
        kind, scene, run = _kind(q), q.get("s", ""), q.get("r", "")
        run_ctl._check_name(scene, run)
        R1 = run_ctl.r1_of(kind, scene, run)
        if path == "/api/run/state":
            h._json(run_ctl.state(kind, scene, run)); return True
        if path == "/api/run/config":
            rc = run_ctl.set_config(kind, scene, run, body or {})
            print(f"[本轮配置] {scene}/{run} -> {rc['覆盖']}", flush=True)
            h._json({"ok": True, "覆盖": rc["覆盖"]}); return True
        if path == "/api/run/people":
            d = run_ctl.set_people(kind, scene, run, body or {})
            print(f"[挑人] {scene}/{run} -> 勾了 {people_decisions.keep_ids(R1)}", flush=True)
            h._json({"ok": True, "勾了": [r["id"] for r in d["人"] if r["决定"] == "keep"]}); return True
        if path == "/api/run/go":
            out = run_ctl.go(kind, scene, run)
            print(f"[开跑] {scene}/{run}: {out}", flush=True)
            h._json({"ok": True, "输出": out}); return True
        body4d = people_decisions.auto_dir(R1)
        if path == "/run/media":
            # 每道闸的图/视频: 只认 run_ctl.media_paths 里的键, 不收任意路径
            p = run_ctl.media_paths(kind, scene, run).get(q.get("what", ""), (None,))[0]
            if not p or not os.path.exists(p):
                h._json({"error": "no such media"}, 404); return True
            ctype = {".mp4": "video/mp4", ".fbx": "application/octet-stream"}.get(os.path.splitext(p)[1], "image/jpeg")
            h._send(200, ctype, open(p, "rb").read()); return True
        if path == "/run/video":
            p = f"{body4d}/03_track.mp4"
            if not os.path.exists(p):
                h._json({"error": "no tracking video"}, 404); return True
            h._send(200, "video/mp4", open(p, "rb").read()); return True
        what = q.get("what")
        if what == "person":
            png = _person_png(R1, q.get("id"))
            if png is None:
                h._json({"error": "no such person"}, 404); return True
            h._send(200, "image/png", cv2.imencode(".png", png)[1].tobytes()); return True
        p = {"ground": f"{R1}/02_ground_check.jpg", "check": f"{body4d}/03_check.jpg"}.get(what)
        if not p or not os.path.exists(p):
            h._json({"error": "no such image"}, 404); return True
        h._send(200, "image/jpeg", open(p, "rb").read()); return True
    except (ValueError, SystemExit) as e:
        # ★ SystemExit 也要接住: 它不是 Exception, 漏到 gate_ui 那层会直接断连接,
        #   浏览器拿到的是代理的 HTML 错误页, 页面上只看得到一句"不是合法 JSON"
        h._json({"error": str(e)}, 400); return True


# ------------------------------------------------------------------ 页面
_STYLE = """<meta name=viewport content="width=device-width,initial-scale=1">
<style>
 :root{--bg:#faf9f5;--panel:#fff;--ink:#141413;--dim:#6b6a65;--line:#e3e1da;--hi:#1a6acb;--red:#d0000e;--ok:#2f7d32}
 @media (prefers-color-scheme:dark){:root{--bg:#1b1b19;--panel:#232320;--ink:#eceae4;--dim:#9a988f;--line:#36352f;--hi:#7fb2f0;--red:#ff6b6b;--ok:#7bc47f}}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 ui-sans-serif,system-ui,"PingFang SC","Microsoft YaHei","Noto Sans CJK SC",sans-serif;padding:20px 16px 48px}
 .wrap{max-width:980px;margin:0 auto}
 a{color:var(--hi)} h1{font-size:19px;margin:0} .dim{color:var(--dim)}
 h2{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--dim);margin:26px 0 8px;font-weight:600}
 .top{display:flex;flex-wrap:wrap;align-items:center;gap:10px 14px;margin-bottom:6px}
 .sp{flex:1}
 .panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px 16px}
 button{font:inherit;padding:6px 14px;border-radius:6px;border:1px solid var(--line);background:var(--panel);color:var(--ink);cursor:pointer}
 button.primary{background:var(--hi);border-color:var(--hi);color:#fff}
 button:disabled{opacity:.45;cursor:default}
 .row{display:flex;flex-wrap:wrap;align-items:center;gap:8px 18px;margin:6px 0}
 label.tg{display:flex;align-items:center;gap:8px}
 .note{color:var(--dim);font-size:12.5px}
 .msg{margin-top:8px;font-size:13px;white-space:pre-wrap} .msg.err{color:var(--red)} .msg.ok{color:var(--ok)}
 .pill{display:inline-block;padding:1px 8px;border-radius:99px;border:1px solid var(--line);font-size:12px;color:var(--dim)}
 .pill.run{border-color:var(--hi);color:var(--hi)}
 img.wide{max-width:100%;border-radius:6px;border:1px solid var(--line);display:block}
 .people{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}
 .card{border:1px solid var(--line);border-radius:8px;overflow:hidden;background:var(--panel);cursor:pointer}
 .card img{display:block;height:180px;margin:0 auto;max-width:100%;object-fit:contain;background:#0001}
 .card .cap{padding:6px 10px;font-size:12.5px;font-variant-numeric:tabular-nums}
 .card.off{opacity:.45} .card.off img{filter:grayscale(1)}
 .card .id{font-weight:600} .warn{color:var(--red)}
 select,input[type=text]{font:inherit;padding:5px 8px;border-radius:6px;border:1px solid var(--line);background:var(--panel);color:var(--ink)}
 details summary{cursor:pointer;color:var(--dim)}
</style>"""

NEW_PAGE = """<!doctype html><meta charset=utf-8><title>新建一轮</title>""" + _STYLE + """
<div class=wrap>
<div class=top><a href="/runs">轮次 ▾</a><h1>新建一轮</h1></div>
<p class=dim>选视频、说清是室内还是室外, 然后开始。驱动在后台跑, 自动挑一张空闲的卡, 跑到第 1 道闸(地基)停下 ——
那里还能改"有人"和"召回"。视频可以是数据集目录下的场景文件夹, 也可以是直接放在数据集目录里的 mp4
(标着"新视频", 开跑时自动建场景文件夹)。</p>
<div class=panel>
 <div class=row><span style="width:90px">类型</span>
  <label class=tg><input type=radio name=k value=outdoor checked> 室外</label>
  <label class=tg><input type=radio name=k value=indoor> 室内</label>
  <span class=note>开跑后不能改。</span></div>
 <div class=row><span style="width:90px">场景</span><select id=scene></select>
  <span class=note id=nsc></span></div>
 <div class=row><span style="width:90px">轮次名</span><input type=text id=run size=14></div>
 <div class=row><span style="width:90px">有人</span>
  <label class=tg><input type=checkbox id=people> 视频里有人: 跟踪人、第 2 道闸挑人、拿人当尺子</label></div>
 <div class=row><span style="width:90px">召回</span>
  <label class=tg><input type=checkbox id=recall> 跑 WildDet3D 召回支路(默认关)</label></div>
 <div class=row style="margin-top:12px"><button class=primary id=go>开始 ▸</button></div>
 <div class=msg id=msg></div>
</div></div>
<script>
let D = null;
const kind = () => document.querySelector('input[name=k]:checked').value;
function fill(){
  const sel = document.getElementById('scene'), rows = (D && D[kind()]) || [];
  sel.innerHTML = rows.map(r => '<option value="'+r.场景+'">'+r.场景+(r.新视频 ? '  (新视频 '+r.新视频+')' : '')+'</option>').join('');
  document.getElementById('nsc').textContent = rows.length ? '' : '这个数据集目录下没有视频';
  nextRun();
}
function nextRun(){
  const rows = (D && D[kind()]) || [], s = document.getElementById('scene').value;
  const r = rows.find(x => x.场景 === s);
  document.getElementById('run').value = r ? r.下一轮 : '';
}
document.querySelectorAll('input[name=k]').forEach(e => e.onchange = fill);
document.getElementById('scene').onchange = nextRun;
fetch('/api/datasets').then(r => r.json()).then(d => { D = d; fill(); });
document.getElementById('go').onclick = async () => {
  const b = document.getElementById('go'), m = document.getElementById('msg');
  b.disabled = true; m.className = 'msg'; m.textContent = '开跑中…';
  const r = await (await fetch('/api/new', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({类型: kind(), 场景: document.getElementById('scene').value,
      轮次: document.getElementById('run').value.trim(),
      有人: document.getElementById('people').checked, 召回: document.getElementById('recall').checked})})).json();
  if (r.error){ m.className = 'msg err'; m.textContent = r.error; b.disabled = false; return; }
  location.href = r.url;
};
</script>"""

RUN_PAGE = """<!doctype html><meta charset=utf-8><title>本轮</title>""" + _STYLE + """
<style>
 .steps{display:flex;gap:6px;flex-wrap:wrap;margin:14px 0 10px}
 .st{flex:1 1 130px;border:1px solid var(--line);border-radius:8px;padding:8px 10px;background:var(--panel);cursor:pointer;min-width:120px}
 .st .n{font-size:12px;color:var(--dim)} .st .t{font-weight:600} .st .g{font-size:12px;color:var(--dim)}
 .st.done{border-color:color-mix(in srgb,var(--ok) 45%,var(--line))} .st.done .t::before{content:"✓ ";color:var(--ok)}
 .st.waiting{border:2px solid var(--hi);padding:7px 9px} .st.waiting .g{color:var(--hi);font-weight:600}
 .st.running{border:2px dashed var(--hi);padding:7px 9px} .st.running .t::before{content:"⟳ ";color:var(--hi)}
 .st.stopped{border-color:var(--red)} .st.stopped .g{color:var(--red)}
 .st.pending{opacity:.55}
 .bar{display:flex;flex-wrap:wrap;align-items:center;gap:10px 16px}
 .bar .what{flex:1 1 320px} .bar .big{font-size:16px;font-weight:600} .bar .sub{color:var(--dim);font-size:13px;margin-top:2px}
 details.seg{background:var(--panel);border:1px solid var(--line);border-radius:8px;margin:10px 0}
 details.seg>summary{padding:10px 14px;cursor:pointer;list-style:none;display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}
 details.seg>summary::-webkit-details-marker{display:none}
 details.seg>summary .nm{font-weight:600} details.seg>summary .meta{color:var(--dim);font-size:12.5px}
 details.seg.waiting{border:2px solid var(--hi)} details.seg.running{border:2px dashed var(--hi)}
 .body{padding:0 14px 14px}
 .body h3{font-size:12px;letter-spacing:.06em;color:var(--dim);margin:14px 0 6px;font-weight:600}
 table.stp{border-collapse:collapse;width:100%;font-size:13px}
 table.stp td{padding:3px 8px 3px 0;border-top:1px solid var(--line);vertical-align:top}
 table.stp td.k{font-family:ui-monospace,monospace;white-space:nowrap} table.stp td.s{white-space:nowrap;width:1%}
 table.stp td.d{color:var(--dim)} table.stp td.tm{text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums;color:var(--dim)}
 tr.running td{color:var(--hi);font-weight:600} tr.pending td{opacity:.5}
 .rep{white-space:pre-wrap;font:12.5px/1.55 ui-monospace,monospace;background:color-mix(in srgb,var(--line) 35%,transparent);border-radius:6px;padding:8px 10px}
 .media{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:10px}
 .media figure{margin:0} .media img,.media video{width:100%;border-radius:6px;border:1px solid var(--line);display:block;background:#000}
 .media figcaption{font-size:12px;color:var(--dim);margin-top:3px}
 a.dl{display:inline-block;padding:7px 12px;border:1px solid var(--line);border-radius:6px;text-decoration:none;color:inherit}
 a.dl:hover{border-color:currentColor}
 .decide{border-left:3px solid var(--hi);padding:2px 0 2px 12px;margin-top:14px}
 table.kv{border-collapse:collapse;font-size:13px} table.kv td{padding:3px 14px 3px 0;border-top:1px solid var(--line)}
 .thumb{height:64px;border-radius:4px;vertical-align:middle;margin-right:6px}
</style>
<div class=wrap>
<div class=top>
 <a href="/runs">轮次 ▾</a><h1 id=title>…</h1><span class=pill id=kind></span><span class=pill id=cfgpill></span>
 <span class=sp></span><a id=cands hidden>候选页 ↗</a><a id=to3d target=_blank hidden>3D ↗</a>
</div>
<div class=steps id=steps></div>
<div class=panel><div class=bar>
  <div class=what><div class=big id=nowbig>…</div><div class=sub id=nowsub></div></div>
  <button class=primary id=go disabled>继续</button>
</div><div class=msg id=gomsg></div></div>
<div id=segs></div>
</div>
<script>
const P = new URLSearchParams(location.search), RUN = {};
for (const k of ['s','r','i']) if (P.get(k)) RUN[k] = P.get(k);
const U = (path, o) => path + '?' + new URLSearchParams({...RUN, ...(o||{})}).toString();
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const dur = s => s == null ? '' : s < 60 ? Math.round(s) + ' 秒' : (s/60).toFixed(1) + ' 分钟';
const STEPICON = {done:'✓', running:'⟳', pending:'·'};
let S = null, cfgDirty = false, peopleDirty = false, dec = {}, open = {};

async function post(path, body){
  const r = await fetch(U(path), {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  return r.json();
}
function say(id, text, err){ const m = $(id); if (!m) return; m.className = 'msg ' + (err ? 'err' : 'ok'); m.textContent = text; }

async function load(){
  let d;
  try {
    const r = await fetch(U('/api/run/state'));
    const t = await r.text();
    try { d = JSON.parse(t); }
    catch(e){ throw new Error('服务返回的不是 JSON(HTTP ' + r.status + ')'); }
  } catch(e){
    // 网页进程重启的那几秒、或者端口转发断了一下: 别停, 自己重试
    say('gomsg', '服务暂时没响应(可能在重启), 3 秒后自动重试… ' + e.message, true);
    setTimeout(load, 3000); return;
  }
  if (d.error){ say('gomsg', d.error, true); return; }
  if ($('gomsg').classList.contains('err') && $('gomsg').textContent.startsWith('服务暂时没响应')) say('gomsg', '已重新连上');
  S = d; render();
}

function segState(g){
  return {done:'完成', waiting:'停在闸 ' + g.段 + ' 等你', running:'正在跑', stopped:'没跑完(看日志)', pending:'未开始'}[g.状态];
}

function render(){
  $('title').textContent = S.场景 + ' / ' + S.轮次;
  document.title = '本轮 ' + S.场景 + '/' + S.轮次;
  $('kind').textContent = S.类型 === 'indoor' ? '室内' : '室外';
  $('cfgpill').textContent = (S.有人 === '1' ? '有人' : '没有人') + ' · 召回' + (S.召回 === '1' ? '开' : '关');
  const c = $('cands'); c.hidden = !(S.段 >= 1); c.href = U('/');
  const v = $('to3d'); v.hidden = !(S.段 >= 2); v.href = U('/3d');

  // ---- 进度条
  $('steps').innerHTML = S.段列表.map(g =>
    '<div class="st ' + g.状态 + '" data-seg="' + g.段 + '"><div class=n>第 ' + g.段 + ' 段</div>' +
    '<div class=t>' + esc(g.名) + '</div><div class=g>' +
    (g.闸 === '跳过' ? '闸 ' + g.段 + ' 跳过(召回关)' : segState(g)) + '</div></div>').join('');
  $('steps').querySelectorAll('.st').forEach(e => e.onclick = () => {
    const d = document.querySelector('details.seg[data-seg="' + e.dataset.seg + '"]');
    if (d){ d.open = true; open[e.dataset.seg] = true; d.scrollIntoView({behavior:'smooth', block:'start'}); }
  });

  // ---- 现在在哪 / 点继续会怎样
  const cur = S.段列表.find(g => g.状态 === 'running' || g.状态 === 'waiting' || g.状态 === 'stopped');
  const nx = S.下一步;
  let big, sub = '';
  if (S.在跑 && cur){
    // 并行之后一段里可以有好几步同时在跑, 各用各的卡
    const rs = cur.步骤.filter(x => x.状态 === 'running');
    big = '正在跑第 ' + cur.段 + ' 段「' + cur.名 + '」' + (S.GPU ? ' · GPU ' + S.GPU : '');
    sub = rs.length ? '当前步骤 ' + rs.map(x => x.步骤).join('、') + (rs.length === 1 ? ' — ' + rs[0].说明 : '') +
          ((S.任务 || []).length ? ' · ' + S.任务.map(t => '卡 ' + t.卡 + ': ' + t.步骤).join(', ') : '') +
          (cur.已用秒 != null ? ' · 已用 ' + dur(cur.已用秒) : '') : '';
  } else if (S.停在闸口){
    big = '停在第 ' + S.闸 + ' 道闸「' + cur.名 + '」: 看下面这一段的结果, 需要的话改决定, 再点继续';
  } else if (S.段 >= S.总段数){
    big = '全部跑完了';
  } else if (S.段 === 0){
    big = '还没开跑';
  } else {
    big = '第 ' + (S.段 + 1) + ' 段没跑完就停了(多半是失败), 看日志; 点继续会重跑这一段';
  }
  if (!S.在跑 && nx) sub = '点继续 → 跑第 ' + nx.段 + ' 段「' + nx.名 + '」: ' + nx.步骤.join(' → ') +
      ', 跑完停在第 ' + nx.停在 + ' 道闸' + (nx.停在 > nx.段 ? '(召回关, 中间的闸不停)' : '');
  $('nowbig').textContent = big; $('nowsub').textContent = sub;
  const go = $('go');
  go.disabled = !S.能继续;
  go.textContent = S.在跑 ? '运行中…' : (S.段 === 0 && !S.停在闸口) ? '开始 ▸' : '继续 ▸';

  // ---- 每一段
  const keepOpen = {};
  document.querySelectorAll('details.seg').forEach(d => keepOpen[d.dataset.seg] = d.open);
  $('segs').innerHTML = S.段列表.map(segHTML).join('');
  document.querySelectorAll('details.seg').forEach(d => {
    const g = S.段列表[+d.dataset.seg - 1];
    d.open = d.dataset.seg in keepOpen ? keepOpen[d.dataset.seg] : ['waiting','running','stopped'].includes(g.状态);
  });
  wire();
}

function segHTML(g){
  const tot = g.步骤.reduce((a, x) => a + (x.耗时 || 0), 0);
  let h = '<details class="seg ' + g.状态 + '" data-seg="' + g.段 + '"><summary><span class=nm>第 ' + g.段 + ' 段 ' + esc(g.名) +
    '</span><span class=meta>' + (g.闸 === '跳过' ? '闸 ' + g.段 + ' 跳过(召回关) · ' : segState(g) + ' · ') +
    g.步骤.length + ' 步' + (tot ? ' · ' + dur(tot) : '') + '</span></summary><div class=body>';
  h += '<h3>跑了什么</h3><table class=stp>' + g.步骤.map(x =>
    '<tr class="' + x.状态 + '"><td class=s>' + (STEPICON[x.状态] || '') + '</td><td class=k>' + esc(x.步骤) +
    '</td><td class=d>' + esc(x.说明) + '</td><td class=tm>' + dur(x.耗时) + '</td></tr>').join('') + '</table>';
  if (g.候选决定){
    const c = g.候选决定, per = Object.entries(c.按词);
    h += '<h3>按你现在保存的决定</h3><p><b>会生成 ' + c.会生成 + ' 个 mesh</b>' +
      (per.length ? ': ' + per.map(([w, n]) => esc(w) + ' ×' + n).join(', ') : '') +
      '<span class=dim> (共 ' + c.候选 + ' 个候选' + (c.作废词.length ? ', 整词作废: ' + c.作废词.map(esc).join(', ') : '') +
      ')</span></p>';
  }
  if (g.报告 && g.报告.length) h += '<h3>闸口报告' + (g.候选决定 ? '(到这道闸时的快照, 之后改的决定不在里面)' : '') +
    '</h3><div class=rep>' + esc(g.报告.join('\\n')) + '</div>';
  if (g.尺度) h += scaleHTML(g.尺度);
  if (g.人表) h += peopleHTML(g.人表);
  if (g.跳过的物体 && g.跳过的物体.length) h += '<h3>没摆进场景的物体</h3><table class=kv>' + g.跳过的物体.map(x =>
    '<tr><td>' + esc(x.词) + ' #' + x.实例 + '</td><td>' + esc(x.原因) + '</td><td class=dim>' +
    (x.尺寸 ? x.尺寸.map(v => v.toFixed(1)).join(' × ') + ' m' : '') + '</td></tr>').join('') + '</table>';
  const dl = g.产出.filter(m => m.下载), shown = g.产出.filter(m => !m.下载);
  if (dl.length) h += '<h3>下载</h3><div class=row>' + dl.map(m =>
    '<a class=dl href="' + U('/run/media', {what:m.what}) + '" download="' + esc(m.下载) + '">⬇ ' + esc(m.说明) +
    ' <span class=dim>' + esc(m.下载) + ' · ' + m.大小MB + ' MB</span></a>').join('') + '</div>';
  if (shown.length) h += '<h3>新产出</h3><div class=media>' + shown.map(m =>
    '<figure>' + (m.视频 ? '<video controls preload=metadata src="' + U('/run/media', {what:m.what}) + '"></video>'
      : '<a href="' + U('/run/media', {what:m.what}) + '" target=_blank><img src="' + U('/run/media', {what:m.what}) + '"></a>') +
    '<figcaption>' + esc(m.说明) + '</figcaption></figure>').join('') + '</div>';
  h += decideHTML(g) + '</div></details>';
  return h;
}

function scaleHTML(sc){
  const rows = Object.entries(sc.每人 || {});
  return '<h3>尺度</h3><table class=kv><tr><td>倍数 k</td><td><b>' + sc.k + '</b></td></tr><tr><td>尺子</td><td>' +
    esc(sc.来源) + '</td></tr></table>' + (rows.length ? '<table class=kv style="margin-top:6px">' + rows.map(([i, r]) =>
    '<tr><td><img class=thumb src="' + U('/run/img', {what:'person', id:i}) + '">#' + esc(i) + '</td><td>场景深度 / 人体模型 = ' +
    r + '</td></tr>').join('') + '</table><p class=note>每个人给一个比例, 取中位数当尺子。离得远、人小的那个最不准。</p>' : '');
}

// 一个跟踪编号最后算谁(和 people_decisions.fate_text 是同一句)
function fateText(r){
  if (!r.去向.length) return r.决定 !== 'keep' ? '没出网格(闸口上没勾)' : '摆人那一次还没勾上';
  return r.去向.map(f => {
    const span = f.视频帧 ? '视频第 ' + f.视频帧[0] + '~' + f.视频帧[1] + ' 帧' : '';
    if (f.事 === '人') return '人 ' + f.人 + (f.帧数 ? '(' + f.帧数 + ' 帧)' : '') + (r.坐着 ? ', 坐着' : '');
    if (f.事 === '并进别人') return span + '(' + f.帧数 + ' 帧)并进人 ' + f.人;
    return span + '(' + f.帧数 + ' 帧)不算人';
  }).join('; ');
}

function peopleHTML(t){
  const sus = t.行.filter(r => r.可疑.length && r.决定 === 'keep').length;
  return '<h3>人(摆进场景 ' + t.人数 + ' 个)</h3>' +
    (t.重跑没完 ? '<p class=warn>上一次按新挑的人重跑没跑完(失败了, 看日志)。下面还是之前的结果; 点继续会接着重跑。</p>' :
     t.过期 ? '<p class=warn>下面是按上一次勾的人算的。点继续会按现在勾的重出网格、重摆、重导出。</p>' : '') +
    '<table class=kv>' + t.行.map(r =>
      '<tr' + (r.决定 === 'drop' ? ' class=dim' : '') + '><td style="width:112px">' +
      (r.新号 ? '' : '<img class=thumb src="' + U('/run/img', {what:'person', id:r.id}) + '">') + '</td><td>#' + r.id + '</td><td>' +
      (r.决定 === 'drop' ? '不要' : r.决定 === 'keep' ? '留下' : '') + '</td><td>' + esc(fateText(r)) +
      (r.可疑.length ? '<br><span class=warn>可疑: ' + esc(r.可疑.join('; ')) + '</span>' : '') + '</td></tr>').join('') +
    '</table><p class=note>编号 = 挑人卡片上的数字, 验收视频和 FBX 里的人(P&lt;号&gt;)也是这个号。' +
    (sus ? '<b>有 ' + sus + ' 个可疑的</b>: 看截图, 不是人(衣服、手、别人身上的一块)就去' : '要去掉谁, 去') +
    '<a href="#people" onclick="openPeople()">第 2 段的"人"</a>把他点成不要、保存, 再点继续 —— 会重出网格、重摆、重导出 FBX。</p>';
}

function openPeople(){
  const d = document.querySelector('details.seg[data-seg="2"]');
  if (d){ d.open = true; setTimeout(() => d.scrollIntoView(), 50); }
}

function decideHTML(g){
  const waiting = g.状态 === 'waiting';
  if (g.段 === 1){
    return '<div class=decide><h3>' + (waiting ? '你要决定' : '本轮配置') + '</h3>' +
      '<div class=row><label class=tg><input type=checkbox id=people> 有人(跟踪人、第 2 道闸挑人、拿人当尺子)</label><span class=note id=pnote></span></div>' +
      '<div class=row><label class=tg><input type=checkbox id=recall> 召回(WildDet3D 捡漏)</label><span class=note id=rnote></span></div>' +
      '<div class=row><button id=savecfg disabled>保存配置</button></div><div class=msg id=cfgmsg></div></div>';
  }
  if (g.段 === 2){
    let h = '<div class=decide><h3>' + (waiting ? '你要决定' : '这一闸的决定(改了点继续会重跑相关步骤)') + '</h3>' +
      '<p>① 物体: <a href="' + U('/') + '">去候选页</a> 定哪些生成、哪些作废, 在那边点 Save。</p>';
    if (!S.人) h += '<p class=dim>② 人: ' + (S.有人 !== '1' ? '这一轮没有人。' : S.手工人体 ? '人来自手工配置的 BODY_RUN, 不在这里挑。' : '人体跟踪还没跑。') + '</p>';
    else h += '<p>② 人: 点卡片切换留下 / 不要, 然后保存。</p><div class=people id=cards></div>' +
      '<div class=row style="margin-top:10px"><button id=savepeople disabled>保存挑人</button><span class=note id=pcount></span></div><div class=msg id=pmsg></div>';
    return h + '</div>';
  }
  if (g.段 === 3 && waiting) return '<div class=decide><h3>你要决定</h3><p>看上面的尺度和人体网格视频, 也可以点顶上 3D 看盒子。没问题就点继续。</p></div>';
  if (g.状态 === 'waiting') return '<div class=decide><h3>你要决定</h3><p>看上面的图和视频, 没问题就点继续。</p></div>';
  return '';
}

// 挑人卡片上多一行: 摆人之后才知道的事(这个编号最后算谁、可不可疑)。摆人还没跑过就没有这一行
function cardFate(id){
  const r = S.人表 && S.人表.行.find(x => x.id === id && !x.新号);
  if (!r || !r.去向.length) return '';
  return '<br><span class=dim>→ ' + esc(fateText(r)) + '</span>' +
    (r.可疑.length ? '<br><span class=warn>可疑: ' + esc(r.可疑.join('; ')) + '</span>' : '');
}

function wire(){
  const pe = $('people'), rc = $('recall');
  if (pe){
    if (!cfgDirty){ pe.checked = S.有人 === '1'; rc.checked = S.召回 === '1'; }
    pe.disabled = !S.可改.PEOPLE; rc.disabled = !S.可改.RECALL;
    $('pnote').textContent = S.手工人体 ? '手工配置的 BODY_RUN, 这里改不了' : S.可改.PEOPLE ? '过第 1 道闸之前能改' :
      S.在跑 ? '跑的时候锁住' : '人体跟踪已经跑过(第 2 段), 要改就新开一轮';
    $('rnote').textContent = S.可改.RECALL ? '过第 3 道闸之前能改' : S.在跑 ? '跑的时候锁住' : '召回那段已经跑过';
    pe.onchange = rc.onchange = () => { cfgDirty = true; $('savecfg').disabled = false; };
    $('savecfg').onclick = async () => {
      const body = {};
      if (S.可改.PEOPLE) body.PEOPLE = pe.checked ? '1' : '0';
      if (S.可改.RECALL) body.RECALL = rc.checked ? '1' : '0';
      const r = await post('/api/run/config', body);
      if (r.error) return say('cfgmsg', r.error, true);
      cfgDirty = false; say('cfgmsg', '已保存'); load();
    };
  }
  const cards = $('cards');
  if (cards && S.人){
    if (!peopleDirty){ dec = {}; S.人.人.forEach(r => dec[r.id] = r.决定); }
    cards.innerHTML = S.人.人.map(r =>
      '<div class="card' + (dec[r.id] === 'keep' ? '' : ' off') + '" data-id="' + r.id + '">' +
      '<img src="' + U('/run/img', {what:'person', id:r.id}) + '" alt="' + r.id + '">' +
      '<div class=cap><span class=id>#' + r.id + '</span> · ' + (dec[r.id] === 'keep' ? '留下' : '<b>不要</b>') +
      '<br>' + r.帧数 + ' 帧 · 最大 ' + r['最大面积%'] + '%' + (r.告警 ? '<br><span class=warn>' + r.告警 + ' 处串号告警</span>' : '') +
      cardFate(r.id) + '</div></div>').join('');
    cards.querySelectorAll('.card').forEach(el => el.onclick = () => {
      const id = +el.dataset.id; dec[id] = dec[id] === 'keep' ? 'drop' : 'keep'; peopleDirty = true; wire();
    });
    const n = Object.values(dec).filter(x => x === 'keep').length;
    $('pcount').textContent = '留下 ' + n + ' / ' + S.人.人.length + (peopleDirty ? ' · 没保存' : '');
    $('savepeople').disabled = !peopleDirty || S.在跑;
    $('savepeople').onclick = async () => {
      const r = await post('/api/run/people', dec);
      if (r.error) return say('pmsg', r.error, true);
      peopleDirty = false; say('pmsg', '已保存, 留下 ' + JSON.stringify(r.勾了) + '。人体网格已经出过的话, 点继续会按新的选择重出。'); load();
    };
  }
}

$('go').onclick = async () => {
  if ((cfgDirty || peopleDirty) && !confirm('有没保存的改动, 不保存直接继续吗?')) return;
  $('go').disabled = true;
  const r = await post('/api/run/go', {});
  if (r.error){ say('gomsg', r.error, true); $('go').disabled = false; return; }
  say('gomsg', r.输出 || '已开跑');
  setTimeout(load, 1500);
};
if (location.hash === '#people') setTimeout(() => { const d = document.querySelector('details.seg[data-seg="2"]');
  if (d){ d.open = true; d.scrollIntoView(); } }, 400);
load();
setInterval(() => { if (S && S.在跑) load(); }, 5000);
</script>"""
