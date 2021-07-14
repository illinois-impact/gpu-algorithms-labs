
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <sys/stat.h>

void generate_csrData(int dim, int **csrRowPtr,
    int **csrColIdx, int **csrData, int * nnz) {

    const int MAX_NNZ_PER_ROW = dim/10 + 1;

    *csrRowPtr = (int*) malloc( sizeof(int)*(dim + 1) );
    (*csrRowPtr)[0] = 0;
    for(int rowIdx = 0; rowIdx < dim; ++rowIdx) {
        int rowNNZ = rand()%(MAX_NNZ_PER_ROW + 1);
        (*csrRowPtr)[rowIdx + 1] = (*csrRowPtr)[rowIdx] + rowNNZ;
    }

    const int NNZ = (*csrRowPtr)[dim];
    *csrColIdx = (int*) malloc( sizeof(int)*NNZ );
    *csrData = (int*) malloc( sizeof(int)*NNZ );
    for (int i = 0; i < NNZ; ++i) {
        (*csrColIdx)[i] = rand()%dim;
        (*csrData)[i] = rand()%10;
    }
    *nnz = NNZ;

}

int * generate_vecData(int len) {
    int * res = (int *) malloc(sizeof(int)*len);
    for (int ii = 0; ii < len; ii++) {
        res[ii] = rand()%10;
    }
    return res;
}

static char *strjoin(const char *s1, const char *s2) {
  char *result = ( char * )malloc(strlen(s1) + strlen(s2) + 1);
  strcpy(result, s1);
  strcat(result, s2);
  return result;
}

static char base_dir[] = "./data";

static void write_data(char *file_name, int flag) {
  FILE *handle = fopen(file_name, "w");
  fprintf(handle, "%d\n", flag);
  fflush(handle);
  fclose(handle);
}

static void write_data(char *file_name, int *data, int len) {
  FILE *handle = fopen(file_name, "w");
  fprintf(handle, "%d\n", len);
  for (int jj = 0; jj < len; jj++) {
    fprintf(handle, "%d", *data++);
    if (jj != len - 1) {
      fprintf(handle, "\n");
    }
  }
  fflush(handle);
  fclose(handle);
}

int * compute(int * rows, int * cols, int * data, int * vec, int len) {
    int * res = (int *) malloc(sizeof(int)*len);
    for (int row = 0; row < len; row++) {
        res[row] = 0;
        int accum = 0;
        int start = rows[row];
        int end = rows[row+1];
        for (int ii = start; ii < end; ii++) {
            int col = cols[ii];
            accum += data[ii]*vec[col];
        }
        res[row] = accum;
    }
    return res;
}

static void create_dataset(int num, int flag, int dim) {
  char dir_name[1024];
  int * cols, * rows, * data, * vec, nnz;

  sprintf(dir_name, "%s/%d", base_dir, num);

  mkdir(dir_name, 0777);

  char *flag_file_name = strjoin(dir_name, "/mode.flag");
  char *csr_col_file_name = strjoin(dir_name, "/col.raw");
  char *csr_row_file_name = strjoin(dir_name, "/row.raw");
  char *csr_data_file_name = strjoin(dir_name, "/data.raw");
  char *vec_file_name = strjoin(dir_name, "/vec.raw");
  char *out_file_name = strjoin(dir_name, "/output.raw");

  generate_csrData(dim, &rows, &cols, &data, &nnz);

  vec = generate_vecData(dim);

  int * res = compute(rows, cols, data, vec, dim);

  write_data(flag_file_name, flag);
  write_data(csr_col_file_name, cols, nnz);
  write_data(csr_row_file_name, rows, dim + 1);
  write_data(csr_data_file_name, data, nnz);
  write_data(vec_file_name, vec, dim);
  write_data(out_file_name, res, dim);

  free(cols);
  free(rows);
  free(data);
  free(vec);
  free(res);
}

int main() {
  create_dataset(0, 0, 16);
  create_dataset(1, 0, 256);
  create_dataset(2, 0, 255);
  create_dataset(3, 0, 912);
  create_dataset(4, 0, 892);
  create_dataset(5, 1, 16);
  create_dataset(6, 1, 256);
  create_dataset(7, 1, 255);
  create_dataset(8, 1, 912);
  create_dataset(9, 1, 892);
  return 0;
}
