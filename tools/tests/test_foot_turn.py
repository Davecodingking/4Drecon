from pathlib import Path
import sys,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"pipeline"))
from legs_proc import turning_intervals


def test_turns_require_continuous_evidence():
    for fps in (12,30,60):
        t = np.arange(0,2,1/fps)
        hidden = np.zeros(len(t),bool)
        shot = np.zeros(len(t),int)
        yaw = np.clip((t-.2)/1.2,0,1)*2*np.pi
        assert turning_intervals(yaw,hidden,t,shot).all()
        assert not turning_intervals(yaw,~hidden,t,shot).any()
        # Reframing or a missing interval cannot turn a stationary actor.
        jump = (t>=1).astype(int)
        assert not turning_intervals(jump*np.pi,hidden,t,jump).any()
        assert not turning_intervals(jump*np.pi,hidden,t+jump*2,shot).any()


if __name__=="__main__":
    test_turns_require_continuous_evidence()
    print("PASS")
