.PHONY: all clean

CFLAGS+=-std=c99 -Wall -O3 -DDEBUG
LDFLAGS+=-lm     # link to math library
CXX = g++
CC = gcc

TARGET=train test train_hmm.o test_hmm.o

all: $(TARGET)
	# type make/make all to compile test_hmm

train: train_hmm.h train_hmm.c train.c
	@make train_hmm.o
	@echo "compiling train ..."
	@$(CC) $(CFLAGS) train.c train_hmm.o -o train

test: test_hmm.h test_hmm.c test.c
	@make test_hmm.o
	@echo "compiling test ..."
	@$(CC) $(CFLAGS) test.c test_hmm.o -o test 

%.o: %.c
	@echo "compiling $< ..."
	@$(CC) $(CFLAGS) $< -c -o $@

clean:
	$(RM) $(TARGET)   
	# type make clean to remove the compiled file
