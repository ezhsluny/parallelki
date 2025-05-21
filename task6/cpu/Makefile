# Компилятор
CXX = pgc++

# Флаги компиляции
CXXFLAGS = -std=c++11
LDFLAGS = -lboost_program_options
ACC_FLAGS = -acc=host -Minfo=all

# Исходные файлы
SRC = equations.cpp
TARGET = equations

# Правила сборки
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(ACC_FLAGS) $< -o $@ $(LDFLAGS)

run: $(TARGET)
	./$(TARGET) --n 512 --eps 1e-6 --iter 1000000

clean:
	rm -f $(TARGET)

.PHONY: all run clean