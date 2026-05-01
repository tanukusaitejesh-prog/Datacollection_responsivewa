import sys
sys.path.append(r'c:\Users\saite\OneDrive\Desktop\ASD')
from spatial_processor import SpatialSkeletonProcessor
import numpy as np

# Create fake data: 10 frames, 33 joints, 3 dims
data = np.random.rand(10, 33, 3)

proc = SpatialSkeletonProcessor()
# Fake some confidence
proc.handle_missing_joints = lambda x, y: x  # bypass
# Run the core methods
data = proc.center_root(data)
data = proc.align_orientation_and_scale(data)

# Now check metrics like JS validator
T = data.shape[0]
L_HIP, R_HIP = 23, 24
L_SHOULDER, R_SHOULDER = 11, 12

rx, ry, rz, length, dx, dy, valid = 0, 0, 0, 0, 0, 0, 0

for t in range(T):
    l_hip = data[t, L_HIP]
    r_hip = data[t, R_HIP]
    l_sho = data[t, L_SHOULDER]
    r_sho = data[t, R_SHOULDER]
    
    root = (l_hip + r_hip) / 2
    sho = (l_sho + r_sho) / 2
    
    rx += abs(root[0])
    ry += abs(root[1])
    rz += abs(root[2])
    
    pdx = sho[0] - root[0]
    pdy = sho[1] - root[1]
    pdz = sho[2] - root[2]
    
    length += np.sqrt(pdx**2 + pdy**2 + pdz**2)
    dx += pdx
    dy += pdy
    valid += 1

print("rootOffset:", (rx+ry+rz)/(3*valid))
print("scale:", length/valid)
print("dx:", dx/valid)
print("dy:", dy/valid)
