CXX = g++

SRC = main.cpp
OBJ = build/main.o
EXEC = build/main
BUILD_DIR = build

FLOAT_TYPE = float

ifeq ($(FLOAT_TYPE), double)
    CXXFLAGS += -DFLOAT_TYPE=double
else
    CXXFLAGS += -DFLOAT_TYPE=float
endif

$(OBJ): $(SRC)
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $(SRC) -o $(OBJ)

$(EXEC): $(OBJ)
	$(CXX) $(OBJ) -o $(EXEC)

run: $(EXEC)
	./$(EXEC)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: run clean
