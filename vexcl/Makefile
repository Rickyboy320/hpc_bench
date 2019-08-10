VEXCL_ROOT=/home/watertor/packages/vexcl-1.4.1/
SRC = vector.cpp variants.cpp

vexcl: vexcl.cpp
	g++ -std=c++0x -o vexcl.out vexcl.cpp -I$(VEXCL_ROOT) -lOpenCL -lboost_system

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo *.o *.out
