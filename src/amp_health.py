"""Monitor persistent AMP failure, not normal sparse loss-scale recalibration."""
import math
from collections import deque


class AMPHealth:
    def __init__(self, policy):
        self.policy=dict(policy)
        self.window=deque(maxlen=policy['window'])
        self.consecutive=0
        self.total_skips=0
        self.slots=0
        self.current_scale=None
        self.minimum_scale=None

    def update(self, skipped, scale):
        self.slots+=1
        self.total_skips+=int(skipped)
        self.consecutive=self.consecutive+1 if skipped else 0
        self.window.append(int(skipped))
        self.current_scale=float(scale)
        self.minimum_scale=self.current_scale if self.minimum_scale is None else min(self.minimum_scale,self.current_scale)
        if not math.isfinite(scale) or scale<self.policy['minimum_scale']:
            raise RuntimeError('AMP scale non-finite or below project conservative minimum')
        if self.consecutive>=self.policy['max_consecutive_skips']:
            raise RuntimeError('Persistent consecutive AMP skipped steps')
        if len(self.window)==self.window.maxlen and sum(self.window)>self.policy['max_window_skip_fraction']*len(self.window):
            raise RuntimeError('AMP rolling-window skip frequency exceeded declared threshold')

    def state_dict(self):
        return {'policy':self.policy,'window':list(self.window),'consecutive':self.consecutive,
                'total_skips':self.total_skips,'slots':self.slots,'current_scale':self.current_scale,'minimum_scale':self.minimum_scale}

    def load_state_dict(self, state):
        if state['policy']!=self.policy:raise RuntimeError('AMP monitor policy mismatch on resume')
        self.window=deque(state['window'],maxlen=self.policy['window'])
        for name in ('consecutive','total_skips','slots','current_scale','minimum_scale'):
            setattr(self,name,state[name])
