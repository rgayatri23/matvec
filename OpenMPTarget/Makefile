SHELL = /bin/sh

#==========================
# Files
#==========================
EXE = matvec.ex
SRC = *.cpp

timemory_profile := n

CPP_OBJ = sna.o test_snap.o

COMP ?= clang
COMP_CHOICE ?= $(COMP)

OPENMP := n
OPENMP_TARGET := n

ifeq ($(LMOD_SYSTEM_NAME),summit)
CUDA_PATH?=$(OLCF_CUDA_ROOT)
DEFINE ?= -DON_SUMMIT
else
CUDA_PATH?=$(CUDA_ROOT)
endif

#==========================
# Compilers
#==========================
#CXX = mpicxx

ifeq ($(LMOD_SYSTEM_NAME),summit)
	CUDA_PATH ?= $(OLCF_CUDA_ROOT)
	DEFINE += -DON_SUMMIT
else
	CUDA_PATH?=$(CUDA_ROOT)
endif

CXXFLAGS = -O3

#==========================
# Machine specific info
# compilers and options
#==========================
ifeq ($(COMP),xl)
CC=xlc_r
CXX=xlC_r
CFLAGS += -qsmp=omp:noauto
CFLAGS += -qoffload
CFLAGS += -qtgtarch=sm_70
ifeq ($(LMOD_SYSTEM_NAME),summit)
	CFLAGS += -qarch=pwr9 -qtune=pwr9
endif
	CXXFLAGS += $(CFLAGS) -std=c++11
	CXXFLAGS += $(DEFINE)
endif

ifeq ($(COMP),clang)
CC=clang
CXX=clang++
#CXXFLAGS = -O2 #-ffast-math -ffp-contract=fast -fstrict-aliasing -Werror -Wall -Wno-unused-variable
CXXFLAGS += -fstrict-aliasing #-ffast-math
CXXFLAGS += -Wno-openmp-target -Wall -Wno-unused-variable
CXXFLAGS += $(DEFINE)
CXXFLAGS += -std=c++11
CXXFLAGS += -lm
#CXXFLAGS += -v #verbose flag
ifeq ($(OPENMP),y)
	CXXFLAGS += -fopenmp
endif
	ifeq ($(OPENMP_TARGET),y)
		CXXFLAGS += -fopenmp
		CXXFLAGS += -fopenmp-targets=nvptx64-nvidia-cuda --cuda-path=${CUDA_PATH} -I${CUDA_PATH}/include -ffp-contract=fast
		CXXFLAGS += -DOPENMP_TARGET
	endif
endif

ifeq ($(COMP),pgi)
	CXX=pgc++
	CXXFLAGS += -fast -Mlarge_arrays $(DEFINE) -std=c++11
  ifeq ($(OPENMP),y)
    CFLAGS += -mp
	endif
  ifeq ($(OPENMP_TARGET),y)
    CFLAGS += -mp=tesla:cc70 --no_exceptions
    CFLAGS += -Minfo=accel #-ta=tesla:managed
		CFLAGS += -mp=tesla:cc70
  endif
  CXXFLAGS += $(CFLAGS) -std=c++11 $(DEFINE)
endif

ifeq ($(COMP),gnu)
CC=gcc
CXX=g++
COMP_CHOICE=gcc
CFLAGS += -Ofast -fstrict-aliasing
CFLAGS += -msse -msse2 -mssse3 -msse4.1 -msse4.2 -mfma -mavx -mavx2 -mavx512f -mavx512cd -DTIMEMORY_VEC=512
CFLAGS += -finline-functions -funroll-loops -ftree-vectorize -ftree-loop-optimize -ftree-loop-vectorize -faligned-new
  ifeq ($(OPENMP),y)
    CFLAGS += -fopenmp -pthread
  else ifeq ($(OPENMP_TARGET),y)
    CFLAGS += -fopenmp
    CFLAGS += -foffload=nvptx-none="-Ofast -lm -misa=sm_35"
  else ifeq ($(OPENACC),y)
    CFLAGS += -fopenacc
  endif

  CXXFLAGS += $(CFLAGS) -std=c++11 $(DEFINE)
  FCFLAGS += $(CFLAGS)
endif


#==========================
# Compiler commands
#==========================
CXXOBJ       = $(CXX) $(CXXFLAGS) -c
CXXLD         = $(CXX)


#==========================
# Make the executable
#==========================
$(EXE): $(SRC) $(INC)
	echo $(SRC)
	$(CXXLD) $(CXXFLAGS) $(SRC) $(PAPI_POST_LINK_OPTS) -o $(EXE)
#==========================
#remove all objs
#==========================
clean:
	/bin/rm -f *.o $(EXE)
