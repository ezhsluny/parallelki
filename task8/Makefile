# Компилятор
CXX = nvc++

# Флаги компиляции
CXXFLAGS = -std=c++17
LIBS = -cudalib=cublas,cub -lboost_program_options
FLAGS = -fast

# Исходные файлы
SRC = gpu_cub.cu
TARGET = gpu_cub

# Правила сборки
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(FLAGS) $< -o $@ $(LIBS)

run: $(TARGET)
	./$(TARGET) --nx 512 --eps 1e-6 --iters 1000000

clean:
	rm -f $(TARGET)

.PHONY: all run clean