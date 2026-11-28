from torch.nn.functional import mse_loss

def world_loss(z,s,b):
    return mse_loss(z.float(),b['future_z'].float()) + .1*mse_loss(s.float(),b['future_state'].float())

def action_loss(a,b):
    return mse_loss(a.float(),b['actions'].float())
