#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// DEFINE RELEVANT STRUCTS AND CONSTRUCTORS/DESTRUCTORS

typedef struct level_ {
    int incoming_ct, outgoing_ct, batch_sz;
    float *weights, *biases;
    float *outputs, *activations, *delta_biases, *delta_weights;
} level;

typedef struct neural_network_ {
    int input_node_ct, output_node_ct, batch_sz;
    int levels_ct;
    int *level_num_nodes;
    level **levels;
} neural_network;

level * construct_level(int incoming_ct, int outgoing_ct, int batch_sz) {
    level *l = (level *)malloc(sizeof(level));
    
    l->weights = (float *)malloc(outgoing_ct * incoming_ct * sizeof(float));
    l->biases = (float *)malloc(outgoing_ct * batch_sz * sizeof(float));
    l->outputs = (float *)malloc(outgoing_ct * batch_sz * sizeof(float));
    l->activations = (float *)malloc(outgoing_ct * batch_sz * sizeof(float));
    l->delta_biases = (float *)malloc(outgoing_ct * batch_sz * sizeof(float));
    l->delta_weights = (float *)malloc(outgoing_ct * incoming_ct * sizeof(float));
    l->incoming_ct = incoming_ct;
    l->outgoing_ct = outgoing_ct;
    l->batch_sz = batch_sz;

    return l;
}

neural_network * construct_net(int input, int output, int size, int levels_ct, int *level_num_nodes) {
    neural_network *model = (neural_network *)malloc(sizeof(neural_network));
    model->input_node_ct = input;
    model->output_node_ct = output;
    model->batch_sz = size;
    model->levels_ct = levels_ct;
    model->level_num_nodes = level_num_nodes;
    model->levels = (level **)malloc(levels_ct * sizeof(level *));

    for (int l = 1; l < levels_ct; ++l) {
        model->levels[l] = construct_level(level_num_nodes[l - 1], level_num_nodes[l], size);
    }

    return model;
}

void initialize_params_kaiming(neural_network *net) {
    int layer_idx, node_idx, weight_idx;
    float std_dev, rand1, rand2, gaussian, angle, scale;

    for (layer_idx = 1; layer_idx < net->levels_ct; layer_idx++) {
        level *current_layer = net->levels[layer_idx];
        std_dev = sqrtf(2.0f / (float)current_layer->incoming_ct);

        for (node_idx = 0; node_idx < current_layer->outgoing_ct; node_idx++) {
            for (weight_idx = 0; weight_idx < current_layer->incoming_ct; weight_idx++) {
                rand1 = (float)rand() / RAND_MAX;
                rand2 = (float)rand() / RAND_MAX;
                scale = sqrtf(-2 * logf(rand1));
                angle = 2 * M_PI * rand2;
                gaussian = scale * cosf(angle) * std_dev;
                int weight_position = node_idx * current_layer->incoming_ct + weight_idx;
                current_layer->weights[weight_position] = gaussian;
            }
        }

        int bias_count = current_layer->outgoing_ct * current_layer->batch_sz;
        for (int bias_idx = 0; bias_idx < bias_count; bias_idx++) {
            current_layer->biases[bias_idx] = 0.0f;
        }
    }
}



void free_nn(neural_network *model) {
    for(int l = 1; l < model->levels_ct; l++) {
        level *lr = model->levels[l];
        free(lr->biases);
        free(lr->weights);
        free(lr->outputs);
        free(lr->delta_biases);
        free(lr->delta_weights);
        free(lr->activations);
        free(lr);
    }
    free(model->levels);
    free(model);
}

// DEFINE RELEVANT MATHEMATICAL HELPER FUNCTIONS
void mat_mul(int M, int N, int K, float alpha, float *A, int lda, float *B, int ldb, float beta, float *C, int ldc) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float temp = 0.0f;
            for (int k = 0; k < K; k++) {
                temp += A[m * lda + k] * B[k * ldb + n];
            }
            C[m * ldc + n] = alpha * temp + beta * C[m * ldc + n];
        }
    }
}


void t_mat_mul(int M, int N, int K, float alpha, float *A, int lda, float *B, int ldb, float beta, float *C, int ldc) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float temp = 0.0f;
            for (int k = 0; k < K; k++) {
                temp += A[k * lda + m] * B[k * ldb + n];
            }
            C[m * ldc + n] = alpha * temp + beta * C[m * ldc + n];
        }
    }
}


void mat_mul_t(int M, int N, int K, float alpha, float *A, int lda, float *B, int ldb, float beta, float *C, int ldc) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float temp = 0.0f;
            for (int k = 0; k < K; k++) {
                temp += A[m * lda + k] * B[n * ldb + k];
            }
            C[m * ldc + n] = alpha * temp + beta * C[m * ldc + n];
        }
    }
}



void softmax(float *output_matrix, float *input_matrix, int total_rows, int total_cols) {
    for (int col = 0; col < total_cols; col++) {
        float exp_sum = 0.0f;
        for (int row = 0; row < total_rows; row++) {
            exp_sum += expf(input_matrix[row * total_cols + col]);
        }
        for (int row = 0; row < total_rows; row++) {
            int index = row * total_cols + col;
            output_matrix[index] = expf(input_matrix[index]) / exp_sum;
        }
    }
}


void scale(float *result, float *input, float factor, int total_rows, int total_cols) {
    for (int row = 0; row < total_rows; row++) {
        for (int col = 0; col < total_cols; col++) {
            int idx = row * total_cols + col;
            result[idx] = input[idx] * factor;
        }
    }
}


void sub_bias(float *result_biases, float *current_biases, int total_rows, int batch_sz, int total_cols) {
    for (int row = 0; row < total_rows; row++) {
        float bias_sum = 0.0f;
        for (int batch_idx = 0; batch_idx < batch_sz; batch_idx++) {
            bias_sum += current_biases[row * batch_sz + batch_idx];
        }
        for (int col = 0; col < total_cols; col++) {
            result_biases[row * total_cols + col] -= bias_sum;
        }
    }
}


void sub_matrices(float *diff_matrix, float *matrix_first, float *matrix_second, int total_rows, int total_cols) {
    for (int idx = 0; idx < total_rows * total_cols; idx++) {
        diff_matrix[idx] = matrix_first[idx] - matrix_second[idx];
    }
}



// NEURAL NETWORK FUNCTIONALITY
void bce_cost(neural_network *n, float *labels, int batch_size) {
    int last_layer_idx = n->levels_ct - 1;
    level *last_layer = n->levels[last_layer_idx];
    sub_matrices(last_layer->outputs, last_layer->outputs, labels, last_layer->outgoing_ct, batch_size);
}


void proc_forward(neural_network *n, int l, float *input_activations) {
    int output_neurons = n->levels[l]->outgoing_ct;
    int input_neurons = n->levels[l]->incoming_ct;
    int batch_size = n->levels[l]->batch_sz;
    float *weights = n->levels[l]->weights;
    float *biases = n->levels[l]->biases;
    float *z_values = n->levels[l]->activations;
    float *a_values = n->levels[l]->outputs;

    //cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, output_neurons, batch_size, input_neurons, 1.0f, weights, input_neurons, input_activations, batch_size, 0.0f, z_values, batch_size);
    mat_mul(output_neurons, batch_size, input_neurons, 1.0f, weights, input_neurons, input_activations, batch_size, 0.0f, z_values, batch_size);

    for (int i = 0; i < output_neurons; i++) {
        for (int j = 0; j < batch_size; j++) {
            z_values[i * batch_size + j] += biases[i];
        }
    }

    if (l < n->levels_ct - 1) {
        for (int i = 0; i < output_neurons; i++) {
            for (int j = 0; j < batch_size; j++) {
                a_values[i * batch_size + j] = z_values[i * batch_size + j] > 0 ? z_values[i * batch_size + j] : 0.0f;
            }
        }
    } else {
        softmax(a_values, z_values, output_neurons, batch_size);
    }
}

void pass_forward(neural_network *n, float *inp_a) {
    proc_forward(n, 1, inp_a);
    for (int l = 2; l < n->levels_ct; l++) {
        proc_forward(n, l, n->levels[l-1]->outputs);
    }
}


void proc_backward(float *out_del, float *inp_a, neural_network *n, int l, int batch_size) {
    int num_outputs = n->levels[l]->outgoing_ct;
    int num_inputs = n->levels[l]->incoming_ct;
    float *delta_biases = n->levels[l]->delta_biases;
    float *activations = n->levels[l]->activations;
    float *delta_weights = n->levels[l]->delta_weights;

    if (l < n->levels_ct - 1) {
        for (int i = 0; i < num_outputs; i++) {
            for (int j = 0; j < batch_size; j++) {
                int idx = i * batch_size + j;
                float relu_derivative = activations[idx] > 0 ? 1.0f : 0.0f;
                delta_biases[idx] = out_del[idx] * relu_derivative;
            }
        }
    } else {
        size_t size = num_outputs * batch_size * sizeof(float);
        memcpy(delta_biases, out_del, size);
    }

    //cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, num_outputs, num_inputs, batch_size, 1.0f, delta_biases, batch_size, inp_a, batch_size, 0.0f, delta_weights, num_inputs); //multiply
    mat_mul_t(num_outputs, num_inputs, batch_size, 1.0f, delta_biases, batch_size, inp_a, batch_size, 0.0f, delta_weights, num_inputs);
}


void pass_backward(neural_network *n, float *inp_a, int batch_size) {
    int last_layer = n->levels_ct - 1;

    proc_backward(n->levels[last_layer]->outputs, inp_a, n, last_layer, batch_size);

    for (int l = last_layer - 1; l >= 1; l--) {
        float *prev_a = l > 1 ? n->levels[l - 1]->outputs : inp_a;

        float *outputs = n->levels[l]->outputs;
        float *next_layer_weights = n->levels[l + 1]->weights;
        float *next_layer_deltas = n->levels[l + 1]->delta_biases;
        int current_layer_neurons = n->levels[l]->outgoing_ct;
        int next_layer_neurons = n->levels[l + 1]->outgoing_ct;
        int current_batch_size = n->levels[l]->batch_sz;

        //cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, current_layer_neurons, current_batch_size, next_layer_neurons, 1.0f, next_layer_weights, current_layer_neurons, next_layer_deltas, current_batch_size, 0.0f, outputs, current_batch_size);
        t_mat_mul(current_layer_neurons, current_batch_size, next_layer_neurons, 1.0f, next_layer_weights, current_layer_neurons, next_layer_deltas, current_batch_size, 0.0f, outputs, current_batch_size);

        proc_backward(outputs, prev_a, n, l, batch_size);
    }
}


void update_params(neural_network *model, float learning_rate, int batch_size) {
    for (int layer_idx = 1; layer_idx < model->levels_ct; layer_idx++) {
        level *current_layer = model->levels[layer_idx];
        float *delta_biases = current_layer->delta_biases;
        float *delta_weights = current_layer->delta_weights;
        float *weights = current_layer->weights;
        float *biases = current_layer->biases;
        int outgoing_count = current_layer->outgoing_ct;
        int incoming_count = current_layer->incoming_ct;
        int layer_batch_size = current_layer->batch_sz;

        scale(delta_biases, delta_biases, (learning_rate / (float)batch_size), outgoing_count, layer_batch_size);
        scale(delta_weights, delta_weights, (learning_rate / (float)batch_size), outgoing_count, incoming_count);
        sub_matrices(weights, weights, delta_weights, outgoing_count, incoming_count);
        sub_bias(biases, delta_biases, outgoing_count, batch_size, layer_batch_size);
    }
}
