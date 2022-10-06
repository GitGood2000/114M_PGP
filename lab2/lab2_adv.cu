#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <random>

using namespace std;

// Утилиты проверки cuda-команд
#define checkCudaErrors(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)

// Проверка на наличие элементов массива
#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}

// Определим размер массива
#define SIZE (100*1024*1024) // 100 MB

// модуль для создания большого массива случайных элементов (0-255)
void* big_random_block(int size) {
	unsigned char* data = (unsigned char*)malloc(size);
	HANDLE_NULL(data);
	for (int i = 0; i < size; i++)
		data[i] = rand();

	return data;
}

// __global__  - выполняется на gpu, вызывается с cpu
__global__ void histo_kernel_optimized(unsigned char* buffer, long size,
	unsigned int* histo)
{
	__shared__ unsigned int temp[1024];

	// помещаем 4 256-массива в один массив размером 1024
	temp[threadIdx.x + 0] = 0;
	temp[threadIdx.x + 256] = 0;
	temp[threadIdx.x + 512] = 0;
	temp[threadIdx.x + 768] = 0;
	__syncthreads();

	// координаты прохода по сетке
	int i = threadIdx.x + blockIdx.x * blockDim.x; // Абсолютный номер потока 
	//blockDim (кол-во потоков в блоке) и blockIdx (номер блока в thread) - глобальные константы в CUDA, по осям x,y,z и т.д
	//threadIds (номер thread-a) 
	int offset = blockDim.x * gridDim.x; // Шаг
	while (i < size)
	{
		/*atomicAdd - Блокирует адрес, в который записывает; прибавляет туда значения (в нашем случае — 1); и затем разблокирует.
		все атомарные операции принимают указатели потому мы прибавляем к local_hist значение элемента. Ссылки (&) неприемлемы.

		Особенность этих функций – они выполняются за один такт, поэтому обеспечивается безопасность данных.
		*/
		atomicAdd(&temp[buffer[i]], 1);
		i += offset;
	}
	__syncthreads();


	// собираем 256-массивы обратно
	atomicAdd(&(histo[threadIdx.x + 0]), temp[threadIdx.x + 0]);
	atomicAdd(&(histo[threadIdx.x + 256]), temp[threadIdx.x + 256]);
	atomicAdd(&(histo[threadIdx.x + 512]), temp[threadIdx.x + 512]);
	atomicAdd(&(histo[threadIdx.x + 768]), temp[threadIdx.x + 768]);

}

int main(void) {
	// в буфере мы будет хранить 100 MB случайных символов
	unsigned char* buffer = (unsigned char*)big_random_block(SIZE);
	
	//Замер времени
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start, 0));


	// выделим память буфера и самой гистрограммы в device
	unsigned char* dev_buffer;
	unsigned int* dev_histo;
	checkCudaErrors(cudaMalloc((void**)&dev_buffer, SIZE));
	checkCudaErrors(cudaMemcpy(dev_buffer, buffer, SIZE,
		cudaMemcpyHostToDevice));
	// выделяем память для гистрограммы (диапазон char [0-255])
	checkCudaErrors(cudaMalloc((void**)&dev_histo, 256 * sizeof(long)));
	checkCudaErrors(cudaMemset(dev_histo, 0, 256 * sizeof(int)));

	//Получаем информацию об устройстве
	cudaDeviceProp prop;
	checkCudaErrors(cudaGetDeviceProperties(&prop, 0));
	// Определяем размер блоков в зависимости от данной информации
	// Поскольку в char диапазон 0-255, размер блока 256 нам наиболее удобный
	int blocks = prop.multiProcessorCount;
	
	//Собственно, само ядро гистрограммы
	histo_kernel_optimized << <blocks * 2, 256 >> > (dev_buffer, SIZE, dev_histo);

	// Гистограмма - распределение массива элементов по массиву ячеек, 
	// где каждая ячейка может содержать только элементы с определенными свойствами.
	// Например, количество букв в предложении.
	unsigned int histo[256];
	checkCudaErrors(cudaMemcpy(histo, dev_histo, 256 * sizeof(int),
		cudaMemcpyDeviceToHost));

	// Остановка замера времени и вывод времени работы
	// Среднее время работы гистрограммы через массив размером 100 MB на процессоре 
	// Intel Core 2 Duo равна 418 мс
	// Наш результат в среднем получается 20 мс (Быстрее в 21 раз!)
	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	float elapsedTime;
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
	printf("Elapsed Time: %3.1f ms\n", elapsedTime);

	// Выведем количество каждого встреченного элемента гистрограммы:
	/*printf("Histogram Result (Times certain number appeared):\n");
	for (int i = 0; i < 256; i++) {
		printf("Element %d: %ld\n", i, histo[i]);
		//printf("%ld\n", histo[i]);
	}*/


	// Выведем полученное количество элементов всего массива (104,857,600)
	long histoCount = 0;
	for (int i = 0; i < 256; i++) {
		histoCount += histo[i];
	}
	printf("Histogram Sum: %ld\n", histoCount);

	// Проверим, правильно ли посчитала программа, сравнив подсчёт с элементами массива
	for (int i = 0; i < SIZE; i++)
		histo[buffer[i]]--;
	for (int i = 0; i < 256; i++) {
		if (histo[i] != 0)
			printf("Failure at %d!\n", i);
	}

	//очистка памяти
	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));
	checkCudaErrors(cudaFree(dev_histo));
	checkCudaErrors(cudaFree(dev_buffer));
	free(buffer);
	return 0;
}
