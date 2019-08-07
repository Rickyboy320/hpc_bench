CC = mpic++

SRC = vector.cpp

CC_FLAGS = -O3 -I/cm/shared/apps/cuda10.0/toolkit/10.0.130/include/ -L/cm/shared/apps/cuda10.0/toolkit/10.0.130/lib64
LL_FLAGS = -fopenmp -lcudart

EXE = vector

debug: $(SRC)
	$(CC) $(CC_FLAGS) $(SRC) -o $(EXE) $(LL_FLAGS)

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result*
