# RiseOdom
Stereo Visual Odometry Algorithm

# Install Instructions
OpenCV 2.4 Is required for this algorithm, make sure to install it first.
To build follow the general cmake build step. This code makes use of intel intrinsic instructions. For PCs with only SSE2 available use USE_PENTIUM4=ON option during cmake configuration.
<code>
mkdir build
cmake ../
make
make install
</code>