#include <cstdio>

void merge(int array[], int start, int middle, int end) {

	int length1 = middle-start+1;
	int length2 = end-middle;
	int firstHalf[length1];
	int secondHalf[length2];

	for (int i = 0; i < length1; i++) {
		firstHalf[i] = array[start+i];
	}
	for (int i = 0; i < length2; i++) {
		secondHalf = array[middle+1+i];
	}

	int i = 0;
	int j = 0;
	int k = start;

	while (i < length1 && j < length2) {
		if (firstHalf[i] > secondHalf[j]) {
			array[k] = secondHalf[j];
			j++;
		}
		else {
			array[k] = firstHalf[i];
			i++;
		}
		k++;
	}

	while (i < length1) {
		array[k] = firstHalf[i];
		i++;
		k++;
	}

	while (j < length2) {
		array[k] = secondHalf[j];
		j++;
		k++;
	}
}

void mergeSort(int array[], int start, int end) {

	if (start < end) {
		int middle = (start+end)/2;

		mergeSort(array, start, middle);
		mergeSort(array, middle+1, end);

		merge(array, start, middle, end);
	}
}

void printArray(int A[], int size)
{
    int i;
    for (i=0; i < size; i++)
        printf("%d ", A[i]);
    printf("\n");
}
 
/* Driver program to test above functions */
int main()
{
    int arr[] = {12, 11, 13, 5, 6, 7};
    int arr_size = sizeof(arr)/sizeof(arr[0]);
 
    printf("Given array is \n");
    printArray(arr, arr_size);
 
    mergeSort(arr, 0, arr_size - 1);
 
    printf("\nSorted array is \n");
    printArray(arr, arr_size);
    return 0;
}