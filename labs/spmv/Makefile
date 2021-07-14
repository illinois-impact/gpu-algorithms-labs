# ----- Make Macros -----

CXX = g++
CXXFLAGS = -std=c++11 -fopenmp
OPTFLAGS = -O3

NVCC = nvcc
NVCCFLAGS = -O3 -std=c++11 -Xcompiler -fopenmp -ccbin=g++-8 #-gencode arch=compute_70,code=sm_70 -ccbin=CC

LD_FLAGS = -lgomp -InvToolsExt

.PHONY: libwb

TARGETS = spmv libwb
OBJECTS = template.o

# ----- Make Rules -----

all:	$(TARGETS)

%.o: %.cpp
	${CXX} ${CXXFLAGS} ${OPTFLAGS} $^ -c -o $@

%.o : %.cu
	${NVCC} ${NVCCFLAGS} $^ -c -o $@ -Ilibwb

spmv: $(OBJECTS) libwb
	$(NVCC) -o $@ $(OBJECTS) $(LD_FLAGS) -Llibwb/lib -lwb

libwb:
	cd libwb; make

clean:
	rm -f $(TARGETS) src/*.o src/*.o.* *.txt *.bin core
