CC=gcc

all: cdnn.so

cdnn.o: cdnn.c cdnn.h cdnn_core.h
	${CC} -fPIC -c -Wall cdnn.c -o cdnn.o

cdnn.so: cdnn.o
	${CC} --shared -fPIC -W1,-soname,libcdnn.so -o libcdnn.so cdnn.o -lc

clean:
	rm *.o
	rm *.so
