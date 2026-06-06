#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <cuda_runtime.h>

typedef struct {
    int ancho, altura, maxcolor;
    int *R, *G, *B;
} Imagen;

typedef struct {
    int kx, ky;
    float *data;
} Kernel;

/* Wall-clock timer used for CPU-side phases and total execution time */
static double wall_time() {
    struct timeval tim;
    gettimeofday(&tim, NULL);
    return tim.tv_sec + tim.tv_usec / 1000000.0;
}

/* Wrapper to stop execution immediately if any CUDA runtime call fails */
static void checkCuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

Imagen *allocImage(int w, int h) {
    Imagen *img = (Imagen *)malloc(sizeof(Imagen));
    int size;

    if (!img) return NULL;

    size = w * h;
    img->ancho = w;
    img->altura = h;
    img->maxcolor = 255;

    img->R = (int *)malloc(size * sizeof(int));
    img->G = (int *)malloc(size * sizeof(int));
    img->B = (int *)malloc(size * sizeof(int));

    if (!img->R || !img->G || !img->B) {
        free(img->R);
        free(img->G);
        free(img->B);
        free(img);
        return NULL;
    }

    return img;
}

void freeImage(Imagen *img) {
    if (!img) return;
    free(img->R);
    free(img->G);
    free(img->B);
    free(img);
}

void freeKernel(Kernel *k) {
    if (!k) return;
    free(k->data);
    free(k);
}

Imagen *readPPM(char *filename) {
    FILE *fp;
    char format[3];
    int c;
    int w, h, max;
    int size, i;
    Imagen *img;

    fp = fopen(filename, "r");
    if (!fp) {
        perror("Error opening image");
        return NULL;
    }

    fscanf(fp, "%2s", format);

    if (strcmp(format, "P3") != 0) {
        printf("Only P3 format supported\n");
        fclose(fp);
        return NULL;
    }

    c = fgetc(fp);
    while (c == '\n' || c == ' ' || c == '\t') {
        c = fgetc(fp);
    }

    while (c == '#') {
        while (c != '\n' && c != EOF) {
            c = fgetc(fp);
        }
        c = fgetc(fp);
        while (c == '\n' || c == ' ' || c == '\t') {
            c = fgetc(fp);
        }
    }

    ungetc(c, fp);

    fscanf(fp, "%d %d", &w, &h);
    fscanf(fp, "%d", &max);

    img = allocImage(w, h);
    if (!img) {
        fclose(fp);
        return NULL;
    }

    img->maxcolor = max;
    size = w * h;

    for (i = 0; i < size; i++) {
        fscanf(fp, "%d %d %d", &img->R[i], &img->G[i], &img->B[i]);
    }

    fclose(fp);
    return img;
}

void writePPM(char *filename, Imagen *img) {
    FILE *fp;
    int size, i;

    fp = fopen(filename, "w");
    if (!fp) {
        perror("Error writing image");
        return;
    }

    fprintf(fp, "P3\n");
    fprintf(fp, "%d %d\n", img->ancho, img->altura);
    fprintf(fp, "%d\n", img->maxcolor);

    size = img->ancho * img->altura;

    for (i = 0; i < size; i++) {
        fprintf(fp, "%d %d %d ", img->R[i], img->G[i], img->B[i]);
        if ((i + 1) % img->ancho == 0) {
            fprintf(fp, "\n");
        }
    }

    fclose(fp);
}

Kernel *readKernel(char *filename) {
    FILE *fp;
    Kernel *k;
    int size, i;

    fp = fopen(filename, "r");
    if (!fp) {
        perror("Kernel error");
        return NULL;
    }

    k = (Kernel *)malloc(sizeof(Kernel));
    if (!k) {
        fclose(fp);
        return NULL;
    }

    fscanf(fp, "%d,%d,", &k->kx, &k->ky);

    size = k->kx * k->ky;
    k->data = (float *)malloc(size * sizeof(float));

    if (!k->data) {
        free(k);
        fclose(fp);
        return NULL;
    }

    for (i = 0; i < size - 1; i++) {
        fscanf(fp, "%f,", &k->data[i]);
    }

    fscanf(fp, "%f", &k->data[size - 1]);

    fclose(fp);
    return k;
}

/* One GPU thread computes one output pixel of one color channel */
__global__ void convolve2D_cuda(const int *in, int *out,
                                int W, int H,
                                const float *kernel,
                                int kx, int ky,
                                int maxcolor) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int cx, cy;
    int m, n;
    float sum;
    int value;

    if (x >= W || y >= H) return;

    cx = kx / 2;
    cy = ky / 2;
    sum = 0.0f;

    for (m = 0; m < ky; m++) {
        int yy = y + m - cy;
        if (yy < 0 || yy >= H) continue;

        for (n = 0; n < kx; n++) {
            int xx = x + n - cx;
            if (xx < 0 || xx >= W) continue;

            /* Kernel indices are flipped to implement standard convolution */
            sum += in[yy * W + xx] * kernel[(ky - 1 - m) * kx + (kx - 1 - n)];
        }
    }

    if (sum >= 0.0f)
        value = (int)(sum + 0.5f);
    else
        value = (int)(sum - 0.5f);

    if (value < 0) value = 0;
    if (value > maxcolor) value = maxcolor;

    out[y * W + x] = value;
}

int main(int argc, char **argv) {
    Imagen *img = NULL;
    Kernel *kern = NULL;

    /* Device buffers for the three input channels, three output channels, and the convolution kernel */
    int *d_Rin = NULL, *d_Gin = NULL, *d_Bin = NULL;
    int *d_Rout = NULL, *d_Gout = NULL, *d_Bout = NULL;
    float *d_kernel = NULL;

    int sizeImg, sizeKernel;
    int partitions;
    int blockX, blockY;

    /* CUDA launch configuration selected at runtime through command-line arguments */
    dim3 block, grid;

    double tstart_total, tend_total;
    double t_read_img = 0.0, t_read_kernel = 0.0, t_write = 0.0;

    /* CUDA events are used to time GPU transfers and kernel execution accurately on the device side */
    float t_h2d = 0.0f, t_kernel = 0.0f, t_d2h = 0.0f;
    cudaEvent_t ev_start, ev_stop;

    if (argc != 7) {
        printf("Usage: %s <image-file> <kernel-file> <result-file> <partitions> <blockX> <blockY>\n", argv[0]);
        printf("- image-file : source image path (*.ppm)\n");
        printf("- kernel-file: kernel path\n");
        printf("- result-file: result image path (*.ppm)\n");
        printf("- partitions : kept for compatibility with previous versions\n");
        printf("- blockX blockY: CUDA block dimensions\n");
        return -1;
    }

    /* The partitions argument is preserved only to keep the same external interface as previous versions */
    partitions = atoi(argv[4]);
    (void)partitions;

    /* Block dimensions are passed as arguments so different CUDA configurations can be tested without recompiling */
    blockX = atoi(argv[5]);
    blockY = atoi(argv[6]);

    if (blockX <= 0 || blockY <= 0) {
        fprintf(stderr, "Invalid CUDA block dimensions\n");
        return -1;
    }

    tstart_total = wall_time();

    t_read_img = wall_time();
    img = readPPM(argv[1]);
    if (!img) {
        return -1;
    }
    t_read_img = wall_time() - t_read_img;

    t_read_kernel = wall_time();
    kern = readKernel(argv[2]);
    if (!kern) {
        freeImage(img);
        return -1;
    }
    t_read_kernel = wall_time() - t_read_kernel;

    sizeImg = img->ancho * img->altura;
    sizeKernel = kern->kx * kern->ky;

    /* Allocate device memory for full image channels and the kernel coefficients */
    checkCuda(cudaMalloc((void **)&d_Rin, sizeImg * sizeof(int)), "cudaMalloc d_Rin");
    checkCuda(cudaMalloc((void **)&d_Gin, sizeImg * sizeof(int)), "cudaMalloc d_Gin");
    checkCuda(cudaMalloc((void **)&d_Bin, sizeImg * sizeof(int)), "cudaMalloc d_Bin");

    checkCuda(cudaMalloc((void **)&d_Rout, sizeImg * sizeof(int)), "cudaMalloc d_Rout");
    checkCuda(cudaMalloc((void **)&d_Gout, sizeImg * sizeof(int)), "cudaMalloc d_Gout");
    checkCuda(cudaMalloc((void **)&d_Bout, sizeImg * sizeof(int)), "cudaMalloc d_Bout");

    checkCuda(cudaMalloc((void **)&d_kernel, sizeKernel * sizeof(float)), "cudaMalloc d_kernel");

    checkCuda(cudaEventCreate(&ev_start), "cudaEventCreate start");
    checkCuda(cudaEventCreate(&ev_stop), "cudaEventCreate stop");

    /* Measure host-to-device copies, including the three image channels and the convolution kernel */
    checkCuda(cudaEventRecord(ev_start), "record H2D start");

    checkCuda(cudaMemcpy(d_Rin, img->R, sizeImg * sizeof(int), cudaMemcpyHostToDevice), "Memcpy R H2D");
    checkCuda(cudaMemcpy(d_Gin, img->G, sizeImg * sizeof(int), cudaMemcpyHostToDevice), "Memcpy G H2D");
    checkCuda(cudaMemcpy(d_Bin, img->B, sizeImg * sizeof(int), cudaMemcpyHostToDevice), "Memcpy B H2D");
    checkCuda(cudaMemcpy(d_kernel, kern->data, sizeKernel * sizeof(float), cudaMemcpyHostToDevice), "Memcpy kernel H2D");

    checkCuda(cudaEventRecord(ev_stop), "record H2D stop");
    checkCuda(cudaEventSynchronize(ev_stop), "sync H2D stop");
    checkCuda(cudaEventElapsedTime(&t_h2d, ev_start, ev_stop), "elapsed H2D");

    /* The grid is computed so that all pixels of the image are covered by the CUDA launch */
    block = dim3(blockX, blockY);
    grid = dim3((img->ancho + block.x - 1) / block.x,
                (img->altura + block.y - 1) / block.y);

    /* Launch the same CUDA kernel three times, one for each independent color channel */
    checkCuda(cudaEventRecord(ev_start), "record kernel start");

    convolve2D_cuda<<<grid, block>>>(d_Rin, d_Rout, img->ancho, img->altura,
                                     d_kernel, kern->kx, kern->ky, img->maxcolor);
    convolve2D_cuda<<<grid, block>>>(d_Gin, d_Gout, img->ancho, img->altura,
                                     d_kernel, kern->kx, kern->ky, img->maxcolor);
    convolve2D_cuda<<<grid, block>>>(d_Bin, d_Bout, img->ancho, img->altura,
                                     d_kernel, kern->kx, kern->ky, img->maxcolor);

    /* Check launch errors and wait until all GPU work is finished before reading the elapsed time */
    checkCuda(cudaGetLastError(), "kernel launch");
    checkCuda(cudaEventRecord(ev_stop), "record kernel stop");
    checkCuda(cudaEventSynchronize(ev_stop), "sync kernel stop");
    checkCuda(cudaEventElapsedTime(&t_kernel, ev_start, ev_stop), "elapsed kernel");

    /* Measure the time to copy the three output channels back from the device to the host */
    checkCuda(cudaEventRecord(ev_start), "record D2H start");

    checkCuda(cudaMemcpy(img->R, d_Rout, sizeImg * sizeof(int), cudaMemcpyDeviceToHost), "Memcpy R D2H");
    checkCuda(cudaMemcpy(img->G, d_Gout, sizeImg * sizeof(int), cudaMemcpyDeviceToHost), "Memcpy G D2H");
    checkCuda(cudaMemcpy(img->B, d_Bout, sizeImg * sizeof(int), cudaMemcpyDeviceToHost), "Memcpy B D2H");

    checkCuda(cudaEventRecord(ev_stop), "record D2H stop");
    checkCuda(cudaEventSynchronize(ev_stop), "sync D2H stop");
    checkCuda(cudaEventElapsedTime(&t_d2h, ev_start, ev_stop), "elapsed D2H");

    t_write = wall_time();
    writePPM(argv[3], img);
    t_write = wall_time() - t_write;

    tend_total = wall_time();

    printf("Image: %s\n", argv[1]);
    printf("Kernel: %s\n", argv[2]);
    printf("Output: %s\n", argv[3]);
    printf("Image size: %d x %d\n", img->ancho, img->altura);
    printf("Kernel size: %d x %d\n", kern->kx, kern->ky);
    printf("CUDA block size: %d x %d\n", blockX, blockY);
    printf("CUDA grid size: %d x %d\n", grid.x, grid.y);
    printf("%.6f seconds elapsed for reading image file.\n", t_read_img);
    printf("%.6f seconds elapsed for reading kernel matrix.\n", t_read_kernel);
    printf("%.6f seconds elapsed for host to device transfers.\n", t_h2d / 1000.0);
    printf("%.6f seconds elapsed for CUDA convolution.\n", t_kernel / 1000.0);
    printf("%.6f seconds elapsed for device to host transfers.\n", t_d2h / 1000.0);
    printf("%.6f seconds elapsed for writing the resulting image.\n", t_write);
    printf("%.6f seconds elapsed in total.\n", tend_total - tstart_total);

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    cudaFree(d_Rin);
    cudaFree(d_Gin);
    cudaFree(d_Bin);
    cudaFree(d_Rout);
    cudaFree(d_Gout);
    cudaFree(d_Bout);
    cudaFree(d_kernel);

    freeImage(img);
    freeKernel(kern);

    return 0;
}