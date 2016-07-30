Supervised Hashing with Kernels [Code]
--
**Terms of Use**

Copyright (c) 2012 by

DVMM Laboratory

Department of Electrical Engineering

Columbia University

Rm 1312 S.W. Mudd, 500 West 120th Street

New York, NY 10027

USA

If it is your intention to use this code for non-commercial purposes, such as in academic research, this code is free.

If you use this code in your research, please acknowledge the authors, and cite our related publication:

Wei Liu, Jun Wang, Rongrong Ji, Yu-Gang Jiang, Shih-Fu Chang. Supervised Hashing with Kernels. In IEEE International Conference on Computer Vision and Pattern Recognition (CVPR), 2012.  

**Instruction**

Please first see KSH_demo.m to find how my codes work.

For a quick illustration, I used a sampled subset (9K) of the Photo Tourism (Notre Dame part) image patch 
dataset. Each image patch was represented by a 512-dimensional GIST feature vector. The raw database can 
be found from http://phototour.cs.washington.edu/patches/default.htm

One possible issue is kernel. I used Gaussian RBF kernel throughout my paper, but any other kernels are
admittable. Please ask me if you do not know how to incorporate other kernels. For the important parameter 
m used in my method, I just simply fix m=300.

For any problem with my codes, feel free to drop me a message via wliu@ee.columbia.edu. Also, I politely ask
you to cite my CVPR'12 paper in your publications.

Wei Liu
October 3, 2012

  






 
