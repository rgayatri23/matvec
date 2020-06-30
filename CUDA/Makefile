CXX=nvcc
CXXFLAGS = -O0 -g
LINK = ${CXX}
LINKFLAGS =

EXE = "matvec"

SRC_CPP = matvec.cpp
SRC_CU = matvec_cudaKernels.cu

default: build
	echo "Start Build"

OBJ_CPP = $(SRC_CPP:.cpp=.o)
OBJ_CU = $(SRC_CU:.cu=.o)

CXXFLAGS += -I$(CUDA_ROOT)/include
CXXFLAGS += -I$(CUDA_ROOT)/samples/common/inc

build: $(EXE)

$(EXE): $(OBJ_CPP) $(OBJ_CU)
	$(LINK) $(LINKFLAGS) $(OBJ_CPP) $(OBJ_CU) -o $(EXE).ex

$(OBJ_CPP): $(SRC_CPP)
		$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_CU): $(SRC_CU)
		$(CXX) $(CXXFLAGS) -gencode arch=compute_70,code=sm_70 -c $< -o $@

clean:
	rm -f *.ex *.o

# Compilation rules

%.o:%.cpp
	$(CXX) $(CXXFLAGS) -c $<
