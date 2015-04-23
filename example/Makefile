CC=g++
INC=-I ../src/
LIB=-L ../lib/

all: tensor_example conv_layer_example pool_layer_example act_layer_example convnet_example

tensor_example: tensor_example.c
	${CC} ${INC} ${LIB} -g $^ -lcdnn -o $@

conv_layer_example: conv_layer_example.c
	${CC} ${INC} $^ -o $@

pool_layer_example: pool_layer_example.c
	${CC} ${INC} $^ -o $@

act_layer_example: act_layer_example.c
	${CC} ${INC} $^ -o $@

convnet_example: convnet_example.c
	${CC} ${INC} $^ -o $@

clean:
	rm conv_layer_example
	rm pool_layer_example
	rm act_layer_example
	rm convnet_example
