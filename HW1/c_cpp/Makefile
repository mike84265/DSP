.PHONY: all clean

CFLAGS+=-std=c99 -Wall -DDEBUG -O3
LDFLAGS+=-lm     # link to math library
CXX = gcc

TARGET=train_hmm.o train test_hmm.o test

all: $(TARGET)
	# type make/make all to compile test_hmm
	@make train
	@make test

train: train_hmm.h train_hmm.c train.c
	@make train_hmm.o
	@echo "compiling train ..."
	@$(CXX) $(CFLAGS) train.c train_hmm.o -o train

test: test_hmm.h test_hmm.c test.c
	@make test_hmm.o
	@echo "compiling test ..."
	@$(CXX) $(CFLAGS) test.c test_hmm.o -o test 

%.o: %.c
	@echo "compiling $< ..."
	@$(CXX) $(CFLAGS) $< -c -o $@

clean:
	$(RM) $(TARGET)   
	# type make clean to remove the compiled file
