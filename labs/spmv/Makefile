# ----- Make Macros -----

CXX = g++
CXXFLAGS = -std=c++11 -fopenmp
OPTFLAGS = -O3

NVCC = nvcc
NVCCFLAGS = -O3 -w -std=c++11 -Xcompiler -fopenmp -ccbin=g++ #-gencode arch=compute_70,code=sm_70 -Xptxas="-v"

LD_FLAGS = -lgomp -InvToolsExt

TARGETS = spmv
OBJECTS = template.o

# ----- Make Rules -----

all:	$(TARGETS)

%.o: %.cpp
	${CXX} ${CXXFLAGS} ${OPTFLAGS} $^ -c -o $@

%.o : %.cu
	${NVCC} ${NVCCFLAGS} $^ -c -o $@ 

spmv: $(OBJECTS)
	$(NVCC) -o $@ $(OBJECTS) $(LD_FLAGS)

clean:
	rm -f $(TARGETS) *.o *.o.* *.txt *.bin core
