#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>

/* Enable or disable MPI debug traces */
#define DEBUG_MPI 1

/* Image stored as three independent color channels */
typedef struct {
    int ancho, altura, maxcolor;
    int *R, *G, *B;
} Imagen;

/* Kernel stored in flattened 1D form */
typedef struct {
    int kx, ky;
    float *data;
} Kernel;

Imagen *allocImage(int w, int h) {
    Imagen *img = malloc(sizeof(Imagen));
    int size;

    if (!img) return NULL;

    size = w * h;

    img->ancho = w;
    img->altura = h;
    img->maxcolor = 255;

    img->R = malloc(size * sizeof(int));
    img->G = malloc(size * sizeof(int));
    img->B = malloc(size * sizeof(int));

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

/* Read an ASCII PPM image in P3 format */
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

    /* Skip blank characters before possible comment lines */
    c = fgetc(fp);
    while (c == '\n' || c == ' ' || c == '\t') {
        c = fgetc(fp);
    }

    /* Skip comment lines beginning with '#' */
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

/* Write the resulting image in ASCII PPM format */
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

        /* Insert a newline after each image row for readability */
        if ((i + 1) % img->ancho == 0) {
            fprintf(fp, "\n");
        }
    }

    fclose(fp);
}

/* Read kernel dimensions and coefficients from the text file */
Kernel *readKernel(char *filename) {
    FILE *fp;
    Kernel *k;
    int size, i;

    fp = fopen(filename, "r");
    if (!fp) {
        perror("Kernel error");
        return NULL;
    }

    k = malloc(sizeof(Kernel));
    if (!k) {
        fclose(fp);
        return NULL;
    }

    fscanf(fp, "%d,%d,", &k->kx, &k->ky);

    size = k->kx * k->ky;
    k->data = malloc(size * sizeof(float));

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

/*
 * Split the image into static row blocks.
 * Each rank receives:
 * - local_rows: rows that belong to its final result
 * - read_rows: local_rows plus halo rows needed for boundary accesses
 * - offset_row: position of the first useful row inside the received block
 */
void compute_static_block(int height, int nprocs, int rank, int halo,
                          int *start_row, int *local_rows,
                          int *read_start, int *read_rows, int *offset_row) {
    int base_rows, extra, end_row, read_end;

    base_rows = height / nprocs;
    extra = height % nprocs;

    *local_rows = base_rows + (rank < extra ? 1 : 0);
    *start_row = rank * base_rows + (rank < extra ? rank : extra);

    end_row = *start_row + *local_rows - 1;

    *read_start = (*start_row - halo > 0) ? (*start_row - halo) : 0;
    read_end = (end_row + halo < height - 1) ? (end_row + halo) : (height - 1);

    *read_rows = read_end - *read_start + 1;
    *offset_row = *start_row - *read_start;
}

/*
 * Apply convolution only on the useful local rows.
 * Halo rows are included in the input buffer only to provide neighbor values.
 */
void convolveRowsOMP(int *in, int *out,
                     int W, int localH,
                     float *kernel,
                     int kx, int ky,
                     int maxcolor,
                     int valid_start_row,
                     int valid_rows) {
    int i, j, m, n;
    int cx, cy;
    int out_row;
    float sum;
    int value;

    cx = kx / 2;
    cy = ky / 2;

    /* Parallelize by rows: each thread writes to different output rows */
    #pragma omp parallel for private(j,m,n,sum,out_row,value) schedule(static)
    for (i = valid_start_row; i < valid_start_row + valid_rows; i++) {
        out_row = i - valid_start_row;

        for (j = 0; j < W; j++) {
            sum = 0.0f;

            for (m = 0; m < ky; m++) {
                int ii = i + m - cy;
                if (ii < 0 || ii >= localH) continue;

                for (n = 0; n < kx; n++) {
                    int jj = j + n - cx;
                    if (jj < 0 || jj >= W) continue;

                    /* Flipped kernel indices implement standard convolution */
                    sum += in[ii * W + jj] *
                           kernel[(ky - 1 - m) * kx + (kx - 1 - n)];
                }
            }

            /* Round and clamp the result to the valid pixel range */
            if (sum >= 0.0f)
                value = (int)(sum + 0.5f);
            else
                value = (int)(sum - 0.5f);

            if (value < 0) value = 0;
            if (value > maxcolor) value = maxcolor;

            out[out_row * W + j] = value;
        }
    }
}

int main(int argc, char **argv) {
    int rank, nprocs;
    Imagen *img = NULL;
    Kernel *kern = NULL;

    int W = 0, H = 0, maxcolor = 0;
    int kx = 0, ky = 0, ksize = 0;
    int halo = 0;
    int partitions = 1;

    int start_row = 0, local_rows = 0, read_start = 0, read_rows = 0, offset_row = 0;
    int input_pixels = 0, output_pixels = 0;

    int *localRin = NULL, *localGin = NULL, *localBin = NULL;
    int *localRout = NULL, *localGout = NULL, *localBout = NULL;
    float *kernel = NULL;

    double start_total = 0.0, end_total = 0.0;
    double start_phase = 0.0;
    double t_read_img = 0.0, t_read_kernel = 0.0, t_dist = 0.0, t_conv = 0.0, t_gather = 0.0, t_write = 0.0;
    double local_time = 0.0, max_time = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    /* Keep a fixed OpenMP configuration inside each MPI process */
    omp_set_dynamic(0);

    if (argc != 5) {
        if (rank == 0) {
            printf("Usage: %s <image.ppm> <kernel.txt> <output.ppm> <partitions>\n", argv[0]);
        }
        MPI_Finalize();
        return -1;
    }

    /*
     * Kept only for compatibility with the original interface.
     * In the hybrid version, the real decomposition is controlled by MPI ranks.
     */
    partitions = atoi(argv[4]);
    (void)partitions;

    MPI_Barrier(MPI_COMM_WORLD);
    start_total = MPI_Wtime();

#if DEBUG_MPI
    /* Check how many OpenMP threads each rank is actually using */
    MPI_Barrier(MPI_COMM_WORLD);
    #pragma omp parallel
    {
        #pragma omp single
        {
            printf("[DEBUG] Rank %d -> OpenMP threads actually used: %d\n", rank, omp_get_num_threads());
            fflush(stdout);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    /* Rank 0 loads the full image and the kernel from disk */
    if (rank == 0) {
        start_phase = MPI_Wtime();

        img = readPPM(argv[1]);
        if (!img) {
            fprintf(stderr, "Error reading image\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        t_read_img = MPI_Wtime() - start_phase;

        start_phase = MPI_Wtime();

        kern = readKernel(argv[2]);
        if (!kern) {
            fprintf(stderr, "Error reading kernel\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        t_read_kernel = MPI_Wtime() - start_phase;

        W = img->ancho;
        H = img->altura;
        maxcolor = img->maxcolor;

        kx = kern->kx;
        ky = kern->ky;
        ksize = kx * ky;
        halo = ky / 2;
    }

    /* Broadcast global metadata so all ranks can compute their local block */
    MPI_Bcast(&W, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&H, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxcolor, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&kx, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ky, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&halo, 1, MPI_INT, 0, MPI_COMM_WORLD);

    ksize = kx * ky;

    kernel = malloc(ksize * sizeof(float));
    if (!kernel) {
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    if (rank == 0) {
        memcpy(kernel, kern->data, ksize * sizeof(float));
    }

    /* Every rank needs the full kernel for the local convolution */
    MPI_Bcast(kernel, ksize, MPI_FLOAT, 0, MPI_COMM_WORLD);

    /* Compute which useful rows and halo rows belong to this rank */
    compute_static_block(H, nprocs, rank, halo,
                         &start_row, &local_rows, &read_start, &read_rows, &offset_row);

    if (local_rows <= 0 || read_rows <= 0 || W <= 0) {
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    input_pixels = read_rows * W;
    output_pixels = local_rows * W;

    /* Input buffers include halo rows; output buffers contain only useful rows */
    localRin  = malloc(input_pixels * sizeof(int));
    localGin  = malloc(input_pixels * sizeof(int));
    localBin  = malloc(input_pixels * sizeof(int));

    localRout = malloc(output_pixels * sizeof(int));
    localGout = malloc(output_pixels * sizeof(int));
    localBout = malloc(output_pixels * sizeof(int));

    if (!localRin || !localGin || !localBin || !localRout || !localGout || !localBout) {
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

#if DEBUG_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    printf("[DEBUG] Rank %d BEFORE distribution\n", rank);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    start_phase = MPI_Wtime();

    if (rank == 0) {
        int dest;

        /*
         * Rank 0 distributes to each process the exact row block it needs,
         * including halo rows for boundary accesses.
         */
        for (dest = 0; dest < nprocs; dest++) {
            int d_start_row, d_local_rows, d_read_start, d_read_rows, d_offset_row;
            int src_offset, send_count;

            compute_static_block(H, nprocs, dest, halo,
                                 &d_start_row, &d_local_rows, &d_read_start, &d_read_rows, &d_offset_row);

            src_offset = d_read_start * W;
            send_count = d_read_rows * W;

#if DEBUG_MPI
            printf("[DEBUG] Master preparing block for rank %d -> start=%d local=%d read_start=%d read_rows=%d send_count=%d\n",
                   dest, d_start_row, d_local_rows, d_read_start, d_read_rows, send_count);
            fflush(stdout);
#endif

            if (dest == 0) {
                memcpy(localRin, img->R + src_offset, send_count * sizeof(int));
                memcpy(localGin, img->G + src_offset, send_count * sizeof(int));
                memcpy(localBin, img->B + src_offset, send_count * sizeof(int));
            } else {
#if DEBUG_MPI
                printf("[DEBUG] Master BEFORE send to rank %d\n", dest);
                fflush(stdout);
#endif

                MPI_Send(img->R + src_offset, send_count, MPI_INT, dest, 10, MPI_COMM_WORLD);
                MPI_Send(img->G + src_offset, send_count, MPI_INT, dest, 11, MPI_COMM_WORLD);
                MPI_Send(img->B + src_offset, send_count, MPI_INT, dest, 12, MPI_COMM_WORLD);

#if DEBUG_MPI
                printf("[DEBUG] Master AFTER send to rank %d\n", dest);
                fflush(stdout);
#endif
            }
        }
    } else {
#if DEBUG_MPI
        printf("[DEBUG] Rank %d BEFORE recv distribution\n", rank);
        fflush(stdout);
#endif

        MPI_Recv(localRin, input_pixels, MPI_INT, 0, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(localGin, input_pixels, MPI_INT, 0, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(localBin, input_pixels, MPI_INT, 0, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

#if DEBUG_MPI
        printf("[DEBUG] Rank %d AFTER recv distribution\n", rank);
        fflush(stdout);
#endif
    }

    MPI_Barrier(MPI_COMM_WORLD);
    local_time = MPI_Wtime() - start_phase;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) t_dist = max_time;

#if DEBUG_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    printf("[DEBUG] Rank %d AFTER distribution\n", rank);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
#endif

#if DEBUG_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    printf("[DEBUG] Rank %d BEFORE convolution\n", rank);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    start_phase = MPI_Wtime();

    /* Each rank convolves its three local channels using OpenMP */
    convolveRowsOMP(localRin, localRout, W, read_rows, kernel, kx, ky, maxcolor, offset_row, local_rows);
    convolveRowsOMP(localGin, localGout, W, read_rows, kernel, kx, ky, maxcolor, offset_row, local_rows);
    convolveRowsOMP(localBin, localBout, W, read_rows, kernel, kx, ky, maxcolor, offset_row, local_rows);

    MPI_Barrier(MPI_COMM_WORLD);
    local_time = MPI_Wtime() - start_phase;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) t_conv = max_time;

#if DEBUG_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    printf("[DEBUG] Rank %d AFTER convolution\n", rank);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
#endif

#if DEBUG_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    printf("[DEBUG] Rank %d BEFORE gather\n", rank);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    start_phase = MPI_Wtime();

    if (rank == 0) {
        int src;

        /* Rank 0 copies its own useful rows directly into the final image */
        memcpy(img->R + start_row * W, localRout, output_pixels * sizeof(int));
        memcpy(img->G + start_row * W, localGout, output_pixels * sizeof(int));
        memcpy(img->B + start_row * W, localBout, output_pixels * sizeof(int));

#if DEBUG_MPI
        printf("[DEBUG] Master copied own block start_row=%d rows=%d\n", start_row, local_rows);
        fflush(stdout);
#endif

        /* Gather useful rows from the remaining ranks */
        for (src = 1; src < nprocs; src++) {
            int s_start_row, s_local_rows, s_read_start, s_read_rows, s_offset_row;
            int recv_offset, recv_count;

            compute_static_block(H, nprocs, src, halo,
                                 &s_start_row, &s_local_rows, &s_read_start, &s_read_rows, &s_offset_row);

            recv_offset = s_start_row * W;
            recv_count = s_local_rows * W;

#if DEBUG_MPI
            printf("[DEBUG] Master BEFORE recv from rank %d -> recv_offset=%d recv_count=%d\n",
                   src, recv_offset, recv_count);
            fflush(stdout);
#endif

            MPI_Recv(img->R + recv_offset, recv_count, MPI_INT, src, 20, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(img->G + recv_offset, recv_count, MPI_INT, src, 21, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(img->B + recv_offset, recv_count, MPI_INT, src, 22, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

#if DEBUG_MPI
            printf("[DEBUG] Master AFTER recv from rank %d\n", src);
            fflush(stdout);
#endif
        }
    } else {
#if DEBUG_MPI
        printf("[DEBUG] Rank %d BEFORE send gather rows=%d\n", rank, local_rows);
        fflush(stdout);
#endif

        MPI_Send(localRout, output_pixels, MPI_INT, 0, 20, MPI_COMM_WORLD);
        MPI_Send(localGout, output_pixels, MPI_INT, 0, 21, MPI_COMM_WORLD);
        MPI_Send(localBout, output_pixels, MPI_INT, 0, 22, MPI_COMM_WORLD);

#if DEBUG_MPI
        printf("[DEBUG] Rank %d AFTER send gather\n", rank);
        fflush(stdout);
#endif
    }

    MPI_Barrier(MPI_COMM_WORLD);
    local_time = MPI_Wtime() - start_phase;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) t_gather = max_time;

#if DEBUG_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    printf("[DEBUG] Rank %d AFTER gather\n", rank);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    /* Only rank 0 writes the reconstructed output image */
    if (rank == 0) {
        start_phase = MPI_Wtime();
        writePPM(argv[3], img);
        t_write = MPI_Wtime() - start_phase;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    end_total = MPI_Wtime();

    if (rank == 0) {
        printf("Image: %s\n", argv[1]);
        printf("Kernel: %s\n", argv[2]);
        printf("Output: %s\n", argv[3]);
        printf("Image size: %d x %d\n", W, H);
        printf("Kernel size: %d x %d\n", kx, ky);
        printf("MPI processes: %d\n", nprocs);
        printf("OpenMP max threads per process: %d\n", omp_get_max_threads());
        printf("%.6f seconds elapsed for reading image file.\n", t_read_img);
        printf("%.6f seconds elapsed for reading kernel matrix.\n", t_read_kernel);
        printf("%.6f seconds elapsed for distributing row blocks.\n", t_dist);
        printf("%.6f seconds elapsed for local convolution.\n", t_conv);
        printf("%.6f seconds elapsed for gathering useful rows.\n", t_gather);
        printf("%.6f seconds elapsed for writing the resulting image.\n", t_write);
        printf("%.6f seconds elapsed in total.\n", end_total - start_total);
    }

    free(localRin);
    free(localGin);
    free(localBin);
    free(localRout);
    free(localGout);
    free(localBout);
    free(kernel);

    if (rank == 0) {
        freeImage(img);
        freeKernel(kern);
    }

    MPI_Finalize();
    return 0;
}