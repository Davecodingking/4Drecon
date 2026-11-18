from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'pipeline'))
from legs_proc import LEG_COLS, BALL_TWIST, distal_prior

def test_local_foot_changes():
    cols=LEG_COLS['l']+LEG_COLS['r']
    source=torch.zeros(3,204)
    source[:,67]=.95  # A measured toe-off.
    for selected in (cols,[c for c in cols if c not in BALL_TWIST]):
        ix,ref,slack=distal_prior(source,selected)
        pose=source[:,selected].clone()
        def cost():
            return ((pose[:,ix]-ref).abs()-slack).clamp_min(0).square().mean(-1)
        assert cost().max()==0
        pose[0,selected.index(67)]-=1.2  # Forefoot curls even if the ankle counter-rotates.
        pose[0,selected.index(64)]+=.7
        pose[1,selected.index(67)]+=.08
        pose[2,selected.index(62)]+=.8
        assert cost()[0]>0 and cost()[1]==0 and cost()[2]==0

if __name__=='__main__':
    test_local_foot_changes()
    print('PASS measured roll, inferred toe curl, and upstream joints')
