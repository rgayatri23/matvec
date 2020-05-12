CXXFLAGS = -O3
LINK = ${CXX}
LINKFLAGS =

EXE = "matvec"

SRC = $(wildcard *.cpp)

default: build
	echo "Start Build"

OBJ = $(SRC:.cpp=.o)

build: $(EXE)

$(EXE): $(OBJ)
	$(LINK) $(LINKFLAGS) $(OBJ) -o $(EXE).ex

clean:
	rm -f *.ex *.o

# Compilation rules

%.o:%.cpp
	$(CXX) $(CXXFLAGS) -c $<
