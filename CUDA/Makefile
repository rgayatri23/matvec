CXX=nvcc
CXXFLAGS = -O3
LINK = ${CXX}
LINKFLAGS =

EXE = "matvec"

SRC = matvec.cpp

default: build
	echo "Start Build"

OBJ = $(SRC:.cpp=.o)

CXXFLAGS += -I$(CUDA_ROOT)/include
CXXFLAGS += -I$(CUDA_ROOT)/samples/common/inc

build: $(EXE)

$(EXE): $(OBJ)
	$(LINK) $(LINKFLAGS) $(OBJ) -o $(EXE).ex

$(OBJ): $(SRC)
		$(CXX) $(CXXFLAGS) -x cu -gencode arch=compute_70,code=sm_70 -c $< -o $@

clean:
	rm -f *.ex *.o

# Compilation rules

%.o:%.cpp
	$(CXX) $(CXXFLAGS) -c $<
