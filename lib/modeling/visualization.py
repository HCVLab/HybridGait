import torch
import numpy as np

from PIL import Image, ImageDraw
from matplotlib import cm
from einops import rearrange


def array_to_cam(arr):
    tmp = np.uint8(cm.gist_earth(arr[184])*255)
    tmp = Image.fromarray(tmp).convert("RGB")
    # cam_pil = Image.fromarray(np.uint8(cm.gist_earth(arr)*255)).convert("RGB")
    return tmp

# relevant
def _get_p_n(N, dtype, kernel_size):
    p_n_x, p_n_y = torch.meshgrid(
        torch.arange(-(kernel_size-1)//2, (kernel_size-1)//2+1),
        torch.arange(-(kernel_size-1)//2, (kernel_size-1)//2+1)
    )
    p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
    p_n = p_n.view(1, 2*N, 1, 1).type(dtype)
    return p_n

# absolute
def _get_p_0(h, w, N, dtype, stride, offset, p_n):
    p_0_x, p_0_y = torch.meshgrid(
        torch.arange(1, h*stride+1, stride),
        torch.arange(1, w*stride+1, stride)
    )
    p_0_x = torch.flatten(p_0_x).view(1,1,h,w).repeat(1,N,1,1)
    p_0_y = torch.flatten(p_0_y).view(1,1,h,w).repeat(1,N,1,1)
    p_0 = torch.cat([p_0_x, p_0_y], 1)
    # b c h w
    p = p_0 + p_n + offset
    return p


def def_visualization(offset):
    # offset: n c h w
    N = offset.size(1)//2
    h = offset.size(2)
    w = offset.size(3)
    p_n = _get_p_n(N, float, 3)
    p = _get_p_0(h, w, N, float, 1, offset, p_n)
    
    return

def deform_visualization(offset):
    # offset: n c h w
    n,c,h,w = offset.shape
    part_offset = offset[:,8:10,:,:]
    p_0_x, p_0_y = torch.meshgrid(
        torch.arange(0, 16, 1),
        torch.arange(0, 11, 1)
    )
    p_0_x = torch.flatten(p_0_x).view(1,1,h,w)
    # .repeat(1,2,1,1)
    p_0_y = torch.flatten(p_0_y).view(1,1,h,w)
    # .repeat(1,2,1,1)
    p_0 = torch.cat([p_0_x, p_0_y], 1).cuda()
    p = (p_0 + part_offset)
    # .floor()
    # 1, 2, h, w
    p = rearrange(p, 'n c h w -> n h w c')
    img = Image.new("RGB", (11, 16), "white")
    draw = ImageDraw.Draw(img)
    for i in range(h):
        for j in range(w):
            x, y = int(p[0][i][j][0] + 0.5), int(p[0][i][j][1] + 0.5)
            print("x:{}, y:{}".format(x,y))
            draw.point((y,x),(255,0,0))
    img.resize((44, 64), Image.BILINEAR)
    img.save("img.jpg")
    image = Image.open("img.jpg")
    image = image.resize((44, 64), Image.BILINEAR)
    image.save("image.jpg")



