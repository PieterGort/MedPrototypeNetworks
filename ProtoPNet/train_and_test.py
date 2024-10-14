import time
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score, precision_recall_curve, auc, roc_auc_score
from utils.helpers import list_of_distances, make_one_hot_sklearn
from utils.metrics import get_sample_weights, binary_metrics, multiclass_metrics
from torch.cuda.amp import GradScaler, autocast

def _train_or_test(model, dataloader, optimizer=None, class_specific=True, use_l1_mask=True,
                   coefs=None, log=print):
    '''
    model: the multi-gpu model
    dataloader:

    optimizer: if None, will be test evaluation
    '''
    is_train = optimizer is not None
    start_epoch = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0

    accumulated_loss = {
        'cross_entropy': torch.tensor(0., device='cuda'),
        'cluster_cost': torch.tensor(0., device='cuda'),
        'separation_cost': torch.tensor(0., device='cuda'),
        'avg_separation_cost': torch.tensor(0., device='cuda'),
        'l1_regularization': torch.tensor(0., device='cuda')
    }
    # Automatic Mixed Precision
    use_amp=True
    scaler = GradScaler(enabled=use_amp)

    # initialize the list of targets and predicted
    all_targets = torch.tensor([], device='cuda')
    all_predicted = torch.tensor([], device='cuda')
    all_probabilities = torch.tensor([], device='cuda')

    for i, (image, label) in enumerate(dataloader):
        input = image.to('cuda')
        target = label.to('cuda')
        
        with torch.set_grad_enabled(is_train):

            with autocast(dtype=torch.float32, enabled=use_amp):
                output, min_distances = model(input)
                cross_entropy = torch.nn.functional.cross_entropy(output, target)

                if class_specific:
                    max_dist = (model.module.prototype_shape[1]
                                * model.module.prototype_shape[2]
                                * model.module.prototype_shape[3])

                    # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
                    # calculate cluster cost
                    prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:,label]).to('cuda')
                    inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
                    cluster_cost = torch.mean(max_dist - inverted_distances)

                    # calculate separation cost
                    prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                    inverted_distances_to_nontarget_prototypes, _ = \
                        torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
                    separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

                    # calculate avg cluster cost
                    avg_separation_cost = \
                        torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
                    avg_separation_cost = torch.mean(avg_separation_cost)
                    
                    if use_l1_mask:
                        l1_mask = 1 - torch.t(model.module.prototype_class_identity).to('cuda')
                        l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
                    else:
                        l1 = model.module.last_layer.weight.norm(p=1) 

                else:
                    min_distance, _ = torch.min(min_distances, dim=1)
                    cluster_cost = torch.mean(min_distance)
                    l1 = model.module.last_layer.weight.norm(p=1)

                # evaluation statistics
                _, predicted = torch.max(output.data, 1)
                n_examples += target.size(0)
                n_correct += (predicted == target).sum().item()

                n_batches += 1
                accumulated_loss['cross_entropy'] += cross_entropy
                accumulated_loss['cluster_cost'] += cluster_cost
                if class_specific:
                    accumulated_loss['separation_cost'] += separation_cost
                    accumulated_loss['avg_separation_cost'] += avg_separation_cost
                accumulated_loss['l1_regularization'] += l1

                probabilities = torch.nn.functional.softmax(output, dim=1)

            # compute gradient and do SGD step
            if is_train:
                if class_specific:
                    if coefs is not None:
                        loss = (coefs['crs_ent'] * cross_entropy
                            + coefs['clst'] * cluster_cost
                            + coefs['sep'] * separation_cost
                            + coefs['l1'] * l1)
                    else:
                        loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1
                else:
                    if coefs is not None:
                        loss = (coefs['crs_ent'] * cross_entropy
                            + coefs['clst'] * cluster_cost
                            + coefs['l1'] * l1)
                    else:
                        loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                for p in model.parameters():
                    p.grad = None
            
            all_targets = torch.cat((all_targets, target), 0)
            all_predicted = torch.cat((all_predicted, predicted), 0)
            all_probabilities = torch.cat((all_probabilities, probabilities), 0)

        del input
        del target
        del output
        del predicted
        del probabilities
        del min_distances

        

    all_probabilities_cpu = all_probabilities.detach().cpu().numpy()
    all_targets_cpu = all_targets.cpu().numpy().astype(np.int64)
    all_predicted_cpu = all_predicted.cpu().numpy().astype(np.int64)

    sample_weights = get_sample_weights(all_targets_cpu)

    # Multiclass Metrics:
    n_classes = all_probabilities_cpu.shape[1]

    if n_classes > 2:
        metrics = multiclass_metrics(all_targets_cpu, all_predicted_cpu, all_probabilities_cpu, n_classes, sample_weights)

    # Binary Metrics:
    else:
        metrics = binary_metrics(all_targets_cpu, all_predicted_cpu, all_probabilities_cpu, pos_label=1, sample_weights=sample_weights)
        metrics['Confusion Matrix'] = None # Not applicable for multiclass


    # Compute average loss
    avg_losses = {k: v.item() / n_batches for k, v in accumulated_loss.items()}
    avg_loss = avg_losses['cross_entropy']*coefs['crs_ent'] \
                + avg_losses['cluster_cost']*coefs['clst'] \
                + avg_losses['separation_cost']*coefs['sep'] \
                + avg_losses['l1_regularization']*coefs['l1'] \
    
    end_epoch = time.time()

    log('\ttime: \t{:.4f}'.format(end_epoch -  start_epoch))
    log('\tcross ent: \t{:.4f}'.format(accumulated_loss['cross_entropy'] / n_batches))
    log('\tcluster: \t{:.4f}'.format(accumulated_loss['cluster_cost'] / n_batches))
    if class_specific:
        log('\tseparation:\t{:.4f}'.format(accumulated_loss['separation_cost'] / n_batches))
        log('\tavg separation:\t{:.4f}'.format(accumulated_loss['avg_separation_cost'] / n_batches))
    log('\taccu: \t\t{:.4f}%'.format(n_correct / n_examples * 100))
    log('\tl1: \t\t{:.4f}'.format(model.module.last_layer.weight.norm(p=1).item()))
    log('\trecall: \t{:.4f}'.format(metrics['Recall']))
    log('\tprecision: \t{:.4f}'.format(metrics['Precision']))
    log('\tf1score: \t{:.4f}'.format(metrics['F1 Score']))
    log('\tbalanced accuracy: \t{:.4f}'.format(metrics['Balanced Accuracy']))
    log('\tauroc: \t{:.4f}'.format(metrics['AUC-ROC']))
    p = model.module.prototype_vectors.view(model.module.num_prototypes, -1).cpu()
    with torch.no_grad():
        p_avg_pair_dist = torch.mean(list_of_distances(p, p))
    log('\tp dist pair: \t{:.4f}'.format(p_avg_pair_dist.item()))
    log('\tloss: \t{:.4f}'.format(avg_loss))

    return {'accuracy': n_correct / n_examples, 
            'cross_entropy': accumulated_loss['cross_entropy'] / n_batches,
            'cluster_cost': accumulated_loss['cluster_cost'] / n_batches,
            'separation_cost': accumulated_loss['separation_cost'] / n_batches,
            'avg_separation_cost': accumulated_loss['avg_separation_cost'] / n_batches,
            'l1': model.module.last_layer.weight.norm(p=1).item(),
            'recall': metrics['Recall'],
            'precision': metrics['Precision'],
            'f1score': metrics['F1 Score'],
            'balanced_accuracy': metrics['Balanced Accuracy'],
            'AUROC': metrics['AUC-ROC'],
            'Confusion Matrix': metrics['Confusion Matrix'],
            'p_avg_pair_dist': p_avg_pair_dist.item(),
            'loss': avg_loss}

def train(model, dataloader, optimizer, class_specific=False, coefs=None, log=print):
    assert(optimizer is not None)
    log('\ttrain')
    model.train()
    scores = _train_or_test(model=model, dataloader=dataloader, optimizer=optimizer,
                            class_specific=class_specific, coefs=coefs, log=log)
    return scores

def val(model, dataloader, class_specific=False, coefs=None, log=print):
    log('\tvalidate')
    model.eval()
    scores = _train_or_test(model=model, dataloader=dataloader, optimizer=None,
                          class_specific=class_specific, coefs=coefs, log=log)
    return scores

def test(model, dataloader, class_specific=False, coefs=None, log=print):
    log('\tPERFORMANCE ON TEST SET FOR FINAL MODEL')
    model.eval()
    scores = _train_or_test(model=model, dataloader=dataloader, optimizer=None,
                          class_specific=class_specific, coefs=coefs, log=log)
    return scores

def last_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False
    model.module.prototype_vectors.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\tlast layer')

def warm_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\twarm')

def joint(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\tjoint')
