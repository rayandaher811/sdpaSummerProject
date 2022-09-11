#!/bin/bash
g++ -c -O3 -funroll-all-loops -D_REENTRANT -Wall -fPIC -funroll-all-loops          -I/usr/local/include -I/home/yuval/sdpa/mumps/build/include -I/usr/include         -o solveSdp.o solveSdp.cpp
g++ -O3 -funroll-all-loops -D_REENTRANT -Wall -fPIC -funroll-all-loops   -o solveSdp.exe solveSdp.o /usr/local/lib/libsdpa.a -L/home/yuval/sdpa/mumps/build/lib -ldmumps -lmumps_common -lpord -L/home/yuval/sdpa/mumps/build/libseq -lmpiseq -llapack -lblas -L/usr/lib/gcc/x86_64-linux-gnu/10 -L/usr/lib/gcc/x86_64-linux-gnu/10/../../../x86_64-linux-gnu -L/usr/lib/gcc/x86_64-linux-gnu/10/../../../../lib -L/lib/x86_64-linux-gnu -L/lib/../lib -L/usr/lib/x86_64-linux-gnu -L/usr/lib/../lib -L/usr/lib/gcc/x86_64-linux-gnu/10/../../.. -lgfortran -lm -lquadmath -lpthread
rm solveSdp.o