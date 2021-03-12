import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import math
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
print(device)

def gabor_kernel_init_seq(weight,lambd = 16.0, nt = 8, n = 0, sl = 0.7, st = 1.4, nl = 4.0):
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.time()
    if n <= 0: n = 1+2*int(nl*lambd)
    gl = -0.5/(sl*sl)  # lambda direction scale factor
    gt = -0.5/(st*st)  # theta direction scale factor
    for t in range (0, nt):
        theta = t*math.pi/nt  # orientation
        c = math.cos(theta)/lambd  # rotation and scaling
        s = math.sin(theta)/lambd  # parameters
        x0 = 0.5*(n-1)*(c+s)  # translation
        y0 = 0.5*(n-1)*(c-s)  # parameters
        sc = 1.0/(2*math.pi*sl*st*lambd*lambd)  # Gaussian normalization factor
        for y in range (0,n):
            for x in range (0,n):
                xr = c*x+s*y-x0  # centering, rotation and scaling
                yr = c*y-s*x-y0  # centering, rotation and scaling
                a = 2.0*math.pi*xr  # wave phase
                g = sc*math.exp(gl*xr*xr+gt*yr*yr)  # Gaussian amplitude
                weight[t+0*nt, 0, y, x] = g*math.cos(a)  # real component
                weight[t+1*nt, 0, y, x] = g*math.sin(a)  # imaginary component
    if torch.cuda.is_available(): torch.cuda.synchronize()
    print("kernel init sequential %dx%d: %.2f ms"% (n, n, 1000*(time.time()-t0)))

def gabor_kernel_init(weight,lambd = 16.0, nt = 8, n = 0, sl = 0.7, st = 1.4, nl = 4.0):
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.time()
    if n <= 0: n = 1+2*int(nl*lambd)
    gl = -0.5/(sl*sl)  # lambda direction scale factor
    gt = -0.5/(st*st)  # theta direction scale factor
    x = torch.tensor(range(n)).unsqueeze_(0).expand(n,n)  # x coordinate
    y = torch.tensor(range(n)).unsqueeze_(1).expand(n,n)  # y coordinate
    for t in range (0, nt):
        theta = t*math.pi/nt  # orientation
        c = math.cos(theta)/lambd  # rotation and scaling
        s = math.sin(theta)/lambd  # parameters
        x0 = 0.5*(n-1)*(c+s)  # translation
        y0 = 0.5*(n-1)*(c-s)  # parameters
        sc = 1.0/(2*math.pi*sl*st*lambd*lambd)  # Gaussian normalization factor
        xr = c*x+s*y-x0  # centering, rotation and scaling
        yr = c*y-s*x-y0  # centering, rotation and scaling
        a = 2.0*math.pi*xr  # wave phase
        g = sc*torch.exp(gl*xr*xr+gt*yr*yr)  # Gaussian amplitude
        weight[t+0*nt, 0] = g*torch.cos(a)  # real component
        weight[t+1*nt, 0] = g*torch.sin(a)  # imaginary component
    if torch.cuda.is_available(): torch.cuda.synchronize()
    print("kernel init %dx%d: %.2f ms"% (n, n, 1000*(time.time()-t0)))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv0 = nn.Conv2d(1, 16, 25, bias=False)
        self.conv1 = nn.Conv2d(1, 16, 49, bias=False)
        self.conv2 = nn.Conv2d(1, 16, 97, bias=False)
        self.conv3 = nn.Conv2d(1, 16, 193, bias=False)
 
    def forward(self, x):
        x0 = torch.squeeze(F.adaptive_avg_pool2d(self.conv0(x)**2, (1, 1)),3)/1
        x1 = torch.squeeze(F.adaptive_avg_pool2d(self.conv1(x)**2, (1, 1)),3)/2
        x2 = torch.squeeze(F.adaptive_avg_pool2d(self.conv2(x)**2, (1, 1)),3)/4
        x3 = torch.squeeze(F.adaptive_avg_pool2d(self.conv3(x)**2, (1, 1)),3)/8
        y = torch.cat((x0, x1, x2, x3), dim = 2)
        y1, y2 = torch.split(y, 8, dim = 1)
        return y1+y2

net = Net()
net.to(device)
if torch.cuda.is_available(): torch.cuda.synchronize()
t0 = time.time()
gabor_kernel_init(net.conv0.weight,lambd = 3.0) # slower due to initializations
gabor_kernel_init(net.conv0.weight,lambd = 3.0)
gabor_kernel_init(net.conv1.weight,lambd = 6.0)
gabor_kernel_init(net.conv2.weight,lambd = 12.0)
gabor_kernel_init(net.conv3.weight,lambd = 24.0)
if torch.cuda.is_available(): torch.cuda.synchronize()
print("total init time: %.2f ms"% (1000*(time.time()-t0)))
#net.to(device)

from PIL import Image
import torchvision.transforms.functional as TF
x = torch.cat((255*TF.to_tensor(Image.open('cat.jpg')).unsqueeze_(0),
               255*TF.to_tensor(Image.open('houses.jpg')).unsqueeze_(0),
               255*TF.to_tensor(Image.open('bur0.jpg')).unsqueeze_(0),
               255*TF.to_tensor(Image.open('bur1.jpg')).unsqueeze_(0))).to(device)

b = 4
#x = torch.randn(b,1,256,256).to(device)
print(x.shape)

nops = ((64.0*64*193*193)+(160.0*160*97*97)+(208.0*208*49*49)+(232*232*25*25))*16.0*2.0
print("Gflops/image: %.3f"% (nops/1000000000.0))

tt = 0
for i in range(20): 
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.time()
    y = net(x)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    dt = time.time()-t0
    print("iter %2d: %.2f ms, %.3f Tflops"% (i, dt*1000, b*nops/dt/1000000000000))
    if (i > 3): tt += dt

dt = tt/16
print("Average: %.2f ms, %.3f Tflops"% (dt*1000, b*nops/dt/1000000000000))

print(y.shape)
print(y)

