import torch


class Channel:
    # returns the message when passed through a channel.
    # AGWN, Fading
    # Note that we need to make sure that the colle map will not change in this
    # step, thus we should not use *= and +=.
    def __init__(self, _iscomplex):
        self._iscomplex = _iscomplex
        self.device = torch.device("cuda:0" )

    def ideal_channel(self, _input):
        return _input

    
    def agwn(self, _input, _snr):
        _std = (10**(-_snr/10.)/2)**0.5
        _dim = _input.shape[0]*_input.shape[1]*_input.shape[2]
        spow = torch.sqrt(torch.sum(_input**2))/_dim**0.5
        _input = _input + torch.randn_like(_input) * _std*spow
        return _input

    def agwn_physical_layer(self, _input, _snr):
        _std = (10**(-_snr/10.)/2)**0.5
        _input = _input + torch.randn_like(_input) * _std
        return _input

    def phase_invariant_fading(self, _input, _snr):
        # ref from JSCC
        _dim = _input.shape[0]*_input.shape[1]*_input.shape[2]
        spow = torch.sqrt(torch.sum(_input**2))/_dim**0.5
        _std = (10**(-_snr/10.)/2)**0.5 if self._iscomplex else (10**(-_snr/10.))**0.5
        _mul = (torch.randn(_input.shape[0], 1)**2 + torch.randn(_input.shape[0], 1)**2)**0.5
        _input = _input * _mul.view(-1,1,1).to(self.device)
        _input = _input +  torch.randn_like(_input) * _std*spow
        return _input

    def phase_invariant_fading_physical_layer(self, _input, _snr):
        # ref from JSCC
        _std = (10**(-_snr/10.)/2)**0.5 if self._iscomplex else (10**(-_snr/10.))**0.5
        _mul = (torch.randn(_input.shape[0], 1)**2 + torch.randn(_input.shape[0], 1)**2)**0.5
        _input = _input * _mul.view(-1,1,1).to(self.device)
        _input = _input +  torch.randn_like(_input) * _std
        #print(_std)
        return _input

# Other Utils:

# RL criteria
class Crit:
    def __init__(self):
        pass

    def __call__(self, mode, *args):
        return getattr(self, '_' + mode)(*args)

    def _xe(self, pred, target, lengths):  # length=16
        mask = pred.new_zeros(len(lengths), target.size(1))
        for i, l in enumerate(lengths):
            mask[i, :l] = 1

        loss = - pred.gather(2, target.unsqueeze(2)).squeeze(2) * mask   # log counted.
        loss = torch.sum(loss) / torch.sum(mask)

        return loss

# others

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


