CC=gcc

all: cdnn.so

cdnn.o: cdnn.c
	${CC} -fPIC -c -Wall $^ -o $@

tensor.o: cdnn_tensor.c
	${CC} -fPIC -c -Wall $^ -o $@

convolution.o: cdnn_convolution.c
	${CC} -fPIC -c -Wall $^ -o $@

pooling.o: cdnn_pooling.c
	${CC} -fPIC -c -Wall $^ -o $@

activation.o: cdnn_activation.c
	${CC} -fPIC -c -Wall cdnn.c -o cdnn.o

cdnn.so: cdnn.o tensor.o convolution.o pooling.o activation.o
	${CC} --shared -fPIC -Wl,-soname,libcdnn.so -o libcdnn.so cdnn.o -lc

clean:
	rm *.o
	rm *.so
