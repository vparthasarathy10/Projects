#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h> 
#include <time.h>
#include "supp_cpu_native.h"

#define input_node_count 784
#define output_node_count 10

#include <stdio.h>
#include <stdlib.h>

float* read_labels_from_file(const char* file_path, int* label_count) {
    FILE* label_file = fopen(file_path, "rb");
    if (!label_file) {
        printf("Error opening label file\n");
        return NULL;
    }

    int data_signature = 0, num_labels = 0;
    fread(&data_signature, sizeof(data_signature), 1, label_file);
    data_signature = ((data_signature & 255) << 24) + ((data_signature >> 8 & 255) << 16) + ((data_signature >> 16 & 255) << 8) + (data_signature >> 24 & 255);

    fread(&num_labels, sizeof(num_labels), 1, label_file);
    num_labels = ((num_labels & 255) << 24) + ((num_labels >> 8 & 255) << 16) + ((num_labels >> 16 & 255) << 8) + (num_labels >> 24 & 255);

    *label_count = num_labels;
    float* labels = (float*)calloc(num_labels * output_node_count, sizeof(float));

    for (int i = 0; i < num_labels; ++i) {
        unsigned char label = 0;
        fread(&label, sizeof(label), 1, label_file);
        labels[i * 10 + label] = 1.0f;
    }
    fclose(label_file);
    return labels;
}


float* read_images_from_file(const char* file_path, int* image_count, int* image_size) {
    FILE* image_file = fopen(file_path, "rb");
    if (!image_file) {
        printf("Error opening image file\n");
        return NULL;
    }

    int data_signature = 0, num_images = 0, height = 0, width = 0;
    fread(&data_signature, sizeof(data_signature), 1, image_file);
    data_signature = ((data_signature & 255) << 24) + ((data_signature >> 8 & 255) << 16) + ((data_signature >> 16 & 255) << 8) + (data_signature >> 24 & 255);

    fread(&num_images, sizeof(num_images), 1, image_file);
    num_images = ((num_images & 255) << 24) + ((num_images >> 8 & 255) << 16) + ((num_images >> 16 & 255) << 8) + (num_images >> 24 & 255);

    fread(&height, sizeof(height), 1, image_file);
    height = ((height & 255) << 24) + ((height >> 8 & 255) << 16) + ((height >> 16 & 255) << 8) + (height >> 24 & 255);

    fread(&width, sizeof(width), 1, image_file);
    width = ((width & 255) << 24) + ((width >> 8 & 255) << 16) + ((width >> 16 & 255) << 8) + (width >> 24 & 255);

    *image_count = num_images;
    *image_size = height * width;

    float* images = (float*)malloc(num_images * height * width * sizeof(float));

    for (int i = 0; i < num_images; ++i) {
        for (int j = 0; j < height * width; ++j) {
            unsigned char pixel = 0;
            fread(&pixel, sizeof(pixel), 1, image_file);
            images[i * height * width + j] = pixel / 255.0f;
        }
    }
    fclose(image_file);
    return images;
}



float* batch_images(float *images, int num_images, int size_input, int b, int batch_sz, int curr_batch_size) {
    float *outputImages = (float*)malloc(curr_batch_size * size_input * sizeof(float));
    int offset = b * batch_sz * size_input;

    for (int i = 0; i < curr_batch_size; ++i) {
        for (int j = 0; j < size_input; ++j) {
            outputImages[j * curr_batch_size + i] = images[offset + i * size_input + j];
        }
    }

    return outputImages;
}


float* batch_labels(float *labels, int num_labs, int b, int batch_sz, int curr_batch_size) {
    float *outputLabels = (float*)malloc(curr_batch_size * 10 * sizeof(float));
    int offset = b * batch_sz * 10;

    for (int i = 0; i < curr_batch_size; ++i) {
        for (int j = 0; j < 10; ++j) {
            outputLabels[j * curr_batch_size + i] = labels[offset + i * 10 + j];
        }
    }

    return outputLabels;
}



int find_actual_label(float* batched_labels, int output_node_ct, int img, int batch_sz) {
    for (int i = 0; i < output_node_ct; i++) {
        if (batched_labels[i * batch_sz + img] == 1.0) {
            return i;
        }
    }
    return -1;
}

int batch_infer(neural_network* model, float* batched_data, float* batched_labels, int batch_sz) {
    int correct = 0;
    
    pass_forward(model, batched_data);

    for (int img = 0; img < batch_sz; img++) {
        int prediction = 0;
        float max_val = model->levels[model->levels_ct - 1]->outputs[img];

        for (int i = 1; i < model->output_node_ct; i++) {
            float val = model->levels[model->levels_ct - 1]->outputs[i * batch_sz + img];
            if (val > max_val) {
                max_val = val;
                prediction = i;
            }
        }

        int actual = find_actual_label(batched_labels, model->output_node_ct, img, batch_sz);

        if (prediction == actual) {
            correct++;
        }
    }

    return correct;
}


float inference(neural_network* model, float* testImages, float* testLabels, int num_test_images, int size_input, int nb) {
    int correct = 0;
    int num_batches = (num_test_images + nb - 1) / nb;

    for (int b = 0; b < num_batches; b++) {
        int batch_sz = (b < num_batches - 1) ? nb : (num_test_images - b * nb);
        float* batched_data = batch_images(testImages, num_test_images, size_input, b, nb, batch_sz);
        float* batched_labels = batch_labels(testLabels, num_test_images, b, nb, batch_sz);
        
        correct += batch_infer(model, batched_data, batched_labels, batch_sz);

        free(batched_data);
        free(batched_labels);
    }

    return (float)correct / num_test_images * 100.0f;
}

float compute_loss(neural_network *n, float *data, float *labels, int num_samples, int size_input) {
    float total_loss = 0.0f;
    int batch_size = n->batch_sz; // Assuming batch size is set in the network
    int num_batches = (num_samples + batch_size - 1) / batch_size;

    for (int b = 0; b < num_batches; b++) {
        int current_batch_size = (b < num_batches - 1) ? batch_size : (num_samples - b * batch_size);
        float *batch_data = batch_images(data, num_samples, size_input, b, batch_size, current_batch_size);
        float *batch_labs = batch_labels(labels, num_samples, b, batch_size, current_batch_size);

        pass_forward(n, batch_data);

        // Compute the loss for the current batch
        level *last_layer = n->levels[n->levels_ct - 1];
        for (int i = 0; i < current_batch_size; i++) {
            for (int j = 0; j < last_layer->outgoing_ct; j++) {
                int idx = i * last_layer->outgoing_ct + j;
                // Adding a small epsilon to avoid log(0)
                total_loss += -batch_labs[idx] * logf(last_layer->outputs[idx] + 1e-10f);
            }
        }

        free(batch_data);
        free(batch_labs);
    }

    return total_loss / num_samples; // Returning the average loss
}





int main(int argc, char *argv[]) {
    srand(time(NULL));

    int settings[5]; // nl, nh, ne, nb, alpha
    for (int i = 0; i < 5; i++) {
        settings[i] = atoi(argv[i + 1]);
    }
    float alpha = atof(argv[5]);

    int levels_count = settings[0] + 2;
    int *level_num_nodes = (int *)malloc(levels_count * sizeof(int));
    level_num_nodes[0] = input_node_count;
    level_num_nodes[settings[0] + 1] = output_node_count;
    for (int i = 1; i <= settings[0]; i++) {
        level_num_nodes[i] = settings[1];
    }

    neural_network *model = construct_net(input_node_count, output_node_count, settings[3], levels_count, level_num_nodes);
    initialize_params_kaiming(model);

    int num_images, size_input, num_labs;
    float *all_images = read_images_from_file("train-images-idx3-ubyte", &num_images, &size_input);
    float *all_labels = read_labels_from_file("train-labels-idx1-ubyte", &num_labs);

    int num_train_images = 50000;
    int num_val_images = num_images - num_train_images;

    float *train_images = all_images;
    float *val_images = all_images + num_train_images * size_input;

    float *train_labels = all_labels;
    float *val_labels = all_labels + num_train_images * output_node_count;

    int num_batches = (num_images + settings[3] - 1) / settings[3];

    float *validation_loss = (float *)malloc(settings[2] * sizeof(float));

    clock_t start_time = clock();
    for (int epoch = 0; epoch < settings[2]; epoch++) {
        for (int b = 0; b < num_batches; b++) {
            int curr_batch_size = b == num_batches - 1 ? num_images - b * settings[3] : settings[3];

            float *batched_data = batch_images(train_images, num_images, size_input, b, settings[3], curr_batch_size);
            float *batched_labels = batch_labels(train_labels, num_labs, b, settings[3], curr_batch_size);

            for (int l = 1; l < model->levels_ct; l++) {
                level *layer = model->levels[l];
                memset(layer->delta_biases, 0, layer->outgoing_ct * layer->batch_sz * sizeof(float));
                memset(layer->delta_weights, 0, layer->outgoing_ct * layer->incoming_ct * sizeof(float));
            }

            proc_forward(model, 1, batched_data);
            for (int l = 2; l < model->levels_ct; l++) {
                proc_forward(model, l, model->levels[l - 1]->outputs);
            }

            bce_cost(model, batched_labels, curr_batch_size);
            pass_backward(model, batched_data, curr_batch_size);
            update_params(model, alpha, curr_batch_size);
            free(batched_labels);
            free(batched_data);
        }
        float val_loss = compute_loss(model, val_images, val_labels, num_val_images, size_input);
        printf("Epoch %d Complete - Validation Loss: %f\n", epoch + 1, val_loss);
        validation_loss[epoch] = val_loss;
    }
    clock_t end_time = clock();
    float training_time = (float)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Training time: %.2f seconds\n", training_time);
    float training_grind_rate = 50000 * settings[2] / training_time;
    printf("Training grind rate: %.2f\n", training_grind_rate);

    int num_test_images, image_size, num_test_labs;
    float *testImages = read_images_from_file("t10k-images-idx3", &num_test_images, &image_size);
    float *testLabels = read_labels_from_file("t10k-labels", &num_test_labs);

    start_time = clock();
    float accuracy = inference(model, testImages, testLabels, num_test_images, size_input, settings[3]);
    end_time = clock();

    float inference_time = (float)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Inference time: %.2f seconds\n", inference_time);
    float inference_grind_rate = 10000 / inference_time;
    printf("Inference grind rate: %.2f\n", inference_grind_rate);
    printf("Test Set Accuracy: %.2f%%\n", accuracy);

    FILE *file = fopen("validation_loss.txt", "w");
    if (file == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }

    for (int i = 0; i < settings[2]; i++) {
        fprintf(file, "%d %f\n", i, validation_loss[i]);
    }
    fclose(file);

    free(validation_loss);
    free(train_images);
    free(train_labels);
    free(testImages);
    free(testLabels);
    free_nn(model);
    free(level_num_nodes);
}
