# This is branch master
import os
import shutil

import torch
import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse
import re

from helpers import makedir
import appnet_model
# import push
# import prune
import train_and_test as tnt
import appnet_train_and_test as atnt
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function

from aux_loaders import load_aux_metadata

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0') # python3 main.py -gpuid=0,1,2,3
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
print(os.environ['CUDA_VISIBLE_DEVICES'])

# book keeping namings and code
from settings import base_architecture, img_size, patch_size, prototype_shape, num_classes, \
                     prototype_activation_function, add_on_layers_type, experiment_run, \
                     patch_encoder_type, update_grads_for_prototypes

base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)

model_dir = './saved_models_appnet/' + base_architecture + '/' + experiment_run + '/'
makedir(model_dir)
shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'settings.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), base_architecture_type + '_features.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'model.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test.py'), dst=model_dir)

log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
img_dir = os.path.join(model_dir, 'img')
makedir(img_dir)
weight_matrix_filename = 'outputL_weights'
prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'

# load the data
from settings import train_dir, test_dir, aux_concept_dir, cub_data_dir, \
                     train_batch_size, test_batch_size, aux_batch_size

normalize = transforms.Normalize(mean=mean,
                                 std=std)

# all datasets
# train set
train_dataset = datasets.ImageFolder(
    train_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ]))
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=train_batch_size, shuffle=True,
    num_workers=4, pin_memory=False)
# test set
test_dataset = datasets.ImageFolder(
    test_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ]))
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=test_batch_size, shuffle=True,
    num_workers=4, pin_memory=False)
# Aux set
concept_dataset = datasets.ImageFolder(
    aux_concept_dir,
    transforms.Compose([
        transforms.Resize(size=(patch_size, patch_size)),
        transforms.ToTensor(),
        normalize,
    ]))
concept_loader = torch.utils.data.DataLoader(
    concept_dataset, batch_size=aux_batch_size, shuffle=True,
    num_workers=4, pin_memory=False)

# we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
log('training set size: {0}'.format(len(train_loader.dataset)))
log('aux set size: {0}'.format(len(concept_loader.dataset)))
log('test set size: {0}'.format(len(test_loader.dataset)))
log('batch size: {0}'.format(train_batch_size))

# Getting class attribute labels
attr_proc, class_attr_proc, attributes_dict, class_dict = load_aux_metadata(
    cub_data_dir
    )

# Cnverting to prototype class identity
attr_in_concepts = [attributes_dict[i].replace('::', '_') for i in attr_proc]
attr_label_order = [
    f for f in os.listdir(aux_concept_dir) if f.startswith('has_')
    ]
attr_label_order.sort()

_num_classes, num_prototypes = class_attr_proc.shape
assert(num_classes == _num_classes)
prototype_class_identity = torch.zeros((num_prototypes, num_classes))

for idx_attr, attr in enumerate(attr_in_concepts):
    attr_label = attr_label_order.index(attr)
    prototype_class_identity[attr_label] = torch.from_numpy(
        class_attr_proc[:, idx_attr]
        )

# construct the model
appnet = appnet_model.construct_APPNet(
    base_architecture, 
    prototype_class_identity, 
    pretrained=True, 
    img_size=img_size,
    prototype_shape=prototype_shape, 
    num_classes=num_classes,
    patch_encoder_type=patch_encoder_type, 
    update_grads_for_prototypes=update_grads_for_prototypes,
    prototype_activation_function=prototype_activation_function,
    add_on_layers_type=add_on_layers_type
)

#if prototype_activation_function == 'linear':
#    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)
appnet = appnet.cuda()
appnet.prototype_to_device('cuda')
appnet_multi = torch.nn.DataParallel(appnet)
class_specific = True

# define optimizer
from settings import joint_optimizer_lrs, joint_lr_step_size
joint_optimizer_specs = \
[{'params': appnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
 {'params': appnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3}
]
joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)

from settings import warm_optimizer_lrs
warm_optimizer_specs = \
[{'params': appnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
]
warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

from settings import last_layer_optimizer_lr
last_layer_optimizer_specs = [{'params': appnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

# weighting of different training losses
from settings import coefs

# number of training epochs, number of warm epochs, push start epoch, push epochs
from settings import num_train_epochs, num_warm_epochs, push_start, push_epochs
from settings import prototype_update_iter_step

# train the model
log('start training')
import copy
for epoch in range(num_train_epochs):
    log('epoch: \t{0}'.format(epoch))

    if epoch < num_warm_epochs:
        atnt.warm_only(model=appnet_multi, log=log)
        _ = atnt.train(
            model=appnet_multi, 
            dataloader=train_loader, 
            aux_dataloader=concept_loader,
            prototype_update_iter_step=prototype_update_iter_step,
            optimizer=warm_optimizer,
            class_specific=class_specific, 
            coefs=coefs, 
            log=log
            )
    else:
        atnt.joint(model=appnet_multi, log=log)
        joint_lr_scheduler.step()
        _ = atnt.train(
            model=appnet_multi, 
            dataloader=train_loader, 
            aux_dataloader=concept_loader,
            prototype_update_iter_step=prototype_update_iter_step,
            optimizer=joint_optimizer,
            class_specific=class_specific, 
            coefs=coefs, 
            log=log
            )

    accu = atnt.test(
        model=appnet_multi, 
        dataloader=test_loader,
        aux_dataloader=concept_loader,
        class_specific=class_specific, 
        log=log
        )
    save.save_model_w_condition(
        model=appnet, 
        model_dir=model_dir, 
        model_name=str(epoch) + 'nopush', 
        accu=accu,
        target_accu=0.20, 
        log=log
        )

    
    if epoch >= push_start and epoch in push_epochs:
        """
        push.push_prototypes(
            train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
            prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
            class_specific=class_specific,
            preprocess_input_function=preprocess_input_function, # normalize if needed
            prototype_layer_stride=1,
            root_dir_for_saving_prototypes=img_dir, # if not None, prototypes will be saved here
            epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
            prototype_img_filename_prefix=prototype_img_filename_prefix,
            prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
            proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
            save_prototype_class_identity=True,
            log=log)
        accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                        class_specific=class_specific, log=log)
        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', accu=accu,
                                    target_accu=0.70, log=log)
        """
        if prototype_activation_function != 'linear':
            atnt.last_only(model=appnet_multi, log=log)
            for i in range(20):
                log('iteration: \t{0}'.format(i))
                _ = atnt.train(
                    model=appnet_multi, 
                    dataloader=train_loader, 
                    aux_dataloader=concept_loader,
                    prototype_update_iter_step=prototype_update_iter_step,
                    optimizer=last_layer_optimizer,
                    class_specific=class_specific, 
                    coefs=coefs, 
                    log=log
                    )
                accu = atnt.test(
                    model=appnet_multi, 
                    dataloader=test_loader,
                    aux_dataloader=concept_loader,
                    class_specific=class_specific, 
                    log=log
                    )
                save.save_model_w_condition(
                    model=appnet, 
                    model_dir=model_dir, 
                    model_name=str(epoch) + '_' + str(i) + 'push', 
                    accu=accu,
                    target_accu=0.20, 
                    log=log)
    
   
logclose()

