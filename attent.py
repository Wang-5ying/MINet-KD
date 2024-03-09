import torch
import torch.nn.functional as F

def get_attention(preds, temp):
    """ preds: Bs*C*W*H """
    N, C, H, W = preds.shape

    value = torch.abs(preds)
    print(value.size())
    # Bs*W*H
    fea_map = value.mean(axis=1, keepdim=True)
    print(fea_map.size())
    S_attention = (H * W * F.softmax((fea_map / temp).view(N, -1), dim=1)).view(N, H, W)

    # Bs*C
    channel_map = value.mean(axis=2, keepdim=False).mean(axis=2, keepdim=False)
    C_attention = C * F.softmax(channel_map / temp, dim=1)
    print(S_attention.size(), C_attention.size())
    return S_attention, C_attention

a = torch.randn(2, 3, 256, 256).cuda()
b = torch.randn(2, 3, 256, 256).cuda()
get_attention(a, 3)