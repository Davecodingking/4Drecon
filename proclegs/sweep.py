"""一次试几组参数(rohm 环境): 每组跑一遍 gate.py(考场 + office 关卡), 最后一行一组并排比。
用法: python sweep.py prefix '{"EDGE_GATE":0.7}' '{"WIDTH_ON":0}' ...   (每组 tag = prefix_序号)
      python sweep.py --show tagA tagB ...    只并排看已有结果
"""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OFF = [('qa_off_center', '偏'), ('qa_wide_stance', '叉'), ('qa_leg_jitter', '抖'), ('sink_cm', '沉'), ('toe_in_L', '内八L'), ('toe_in_R', '内八R'), ('same_foot', '同脚'), ('miss_cm', '够不着')]
BEN = [('off_center', '偏'), ('wide_stance', '叉'), ('leg_jitter', '抖'), ('knee_jump', '膝跳')]


def show(tags):
    base = json.load(open(f'{HERE}/outputs/office_gate_base.json'))
    hdr = f'{"tag":<14}{"过":>3} | 像office ' + ' '.join(f'{n:>4}' for _, n in BEN) + ' | 正常 ' + ' '.join(f'{n:>4}' for _, n in BEN) + ' | office ' + ' '.join(f'{n:>5}' for _, n in OFF)
    print(hdr)
    print(f'{"v10 (office)":<14}{"":>3} | {"":>9}' + ' ' * 20 + ' | ' + ' ' * 24 + ' | office ' + ' '.join(f'{base[k]:>5.1f}' for k, _ in OFF))
    for t in tags:
        r = json.load(open(f'{HERE}/outputs/gate_{t}.json'))
        print(f'{t:<14}{"是" if r["pass"] else "否":>3} | 像office ' + ' '.join(f'{r["像office"]["ours"][k]:>4.1f}' for k, _ in BEN)
              + ' | 正常 ' + ' '.join(f'{r["正常走路"]["ours"][k]:>4.1f}' for k, _ in BEN) + ' | office ' + ' '.join(f'{r["office"][k]:>5.1f}' for k, _ in OFF)
              + (f'   {r["opt"]}' if r.get('opt') else ''))


def main():
    if sys.argv[1] == '--show':
        return show(sys.argv[2:])
    prefix, opts = sys.argv[1], sys.argv[2:]
    tags = []
    for j, o in enumerate(opts):
        tag = f'{prefix}_{j}'; tags.append(tag)
        subprocess.run([sys.executable, f'{HERE}/gate.py', '--tag', tag, '--opt', o], stdout=subprocess.DEVNULL, check=True, cwd=HERE)
        print('done', tag, o, flush=True)
    show(tags)


if __name__ == '__main__':
    main()
