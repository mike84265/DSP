.PHONY: all clean

CFLAGS+=-std=c++11 -Wall -DDEBUG -g
LDFLAGS+=-lm     # link to math library
CXX = g++

TARGET=train_hmm.o train test_hmm.o test

all: $(TARGET)
	# type make/make all to compile test_hmm
	@make train
	@make test

train: train_hmm.h train_hmm.cpp train.cpp
	@make train_hmm.o
	@echo "compiling train ..."
	@$(CXX) $(CFLAGS) train.cpp train_hmm.o -o train

test: test_hmm.h test_hmm.cpp test.cpp
	@make test_hmm.o
	@echo "compiling test ..."
	@$(CXX) $(CFLAGS) test.cpp test_hmm.o -o test 

%.o: %.cpp
	@echo "compiling $< ..."
	@$(CXX) $(CFLAGS) $< -c -o $@

clean:
	$(RM) $(TARGET)   
	# type make clean to remove the compiled file
