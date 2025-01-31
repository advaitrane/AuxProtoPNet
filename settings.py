base_architecture = 'vgg19'
img_size = 224
patch_size = 56
prototype_shape = (112, 128, 1, 1)
num_classes = 200
prototype_activation_function = 'log'
add_on_layers_type = 'regular'
patch_encoder_type = 'same'
update_grads_for_prototypes = True
prototype_update_iter_step = 1

experiment_run = '006'

data_path = './datasets/cub200_cropped/'
train_dir = data_path + 'train_cropped_augmented/'
test_dir = data_path + 'test_cropped/'
aux_concept_dir = data_path + 'CUB_ConceptCrops/'
cub_data_dir = data_path + 'CUB_200_2011/'
# train_push_dir = data_path + 'train_cropped/'
train_batch_size = 80
test_batch_size = 100
# train_push_batch_size = 75
aux_batch_size = 112

joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       # 'prototype_vectors': 3e-3
                       }
joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      # 'prototype_vectors': 3e-3
                      }

last_layer_optimizer_lr = 1e-4

coefs = {
    'crs_ent': 1,
    'clst': 0.0008,
    'sep': -0.008,
    'l1': 1e-3,
}

num_train_epochs = 50
num_warm_epochs = 5

push_start = 5
push_epochs = [i for i in range(num_train_epochs) if i % 5 == 0]