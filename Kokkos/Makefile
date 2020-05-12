KOKKOS_PATH = $(HOME)/Kokkos/kokkos

DEFINE = -O3
CXXFLAGS = $(DEFINE) -lineinfo
LINK = ${CXX}
LINKFLAGS =

ifeq ($(DEVICE),openmp)
	KOKKOS_DEVICES = "OpenMP"
	DEFINE += -Dopenmp_space
	CXX = CC
	CXXFLAGS += -qopenmp
	LINKFLAGS += -qopenmp
	EXE = ${EXE_NAME}.openmp
else ifeq ($(DEVICE),cudauvm)
	KOKKOS_DEVICES = "Cuda"
	DEFINE += -DcudaUVM_space
	CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper
	EXE = ${EXE_NAME}.cuda
	KOKKOS_ARCH = "Volta70"
	KOKKOS_CUDA_OPTIONS = "force_uvm,enable_lambda"
else ifeq ($(DEVICE),cuda)
	KOKKOS_DEVICES = "Cuda"
	DEFINE += -Dcuda_space
	CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper
	EXE = ${EXE_NAME}.cuda
	KOKKOS_ARCH = "Volta70"
	KOKKOS_CUDA_OPTIONS = "enable_lambda -lineinfo"
else
	KOKKOS_DEVICES = "Serial"
	DEFINE += -Dserial_space
	CXX = CC
	CXXFLAGS += -xcore-avx2
	EXE = ${EXE_NAME}.host
endif

EXE_NAME = "matvec"

SRC = $(wildcard *.cpp)

default: build
	echo "Start Build"

DEPFLAGS = -M

OBJ = $(SRC:.cpp=.o)
LIB =

include $(KOKKOS_PATH)/Makefile.kokkos

build: $(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

clean: kokkos-clean
	rm -f *.openmp *.o *.cuda *.host

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $<

test: $(EXE)
	./$(EXE)
