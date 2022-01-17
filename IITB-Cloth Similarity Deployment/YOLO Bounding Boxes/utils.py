import torch
import torchvision
import torch.nn as nn
from pytorch_pretrained_vit import ViT

def accuracy(positives, negatives):
    # _, preds = torch.max(outputs, dim=1)
    difference_distance = torch.abs(negatives) - torch.abs(positives) 
    result = torch.abs(positives) - torch.abs(negatives)
    # result = torch.tensor([1 if distance > 0 else 0 for distance in result]) 
    result[result >= 0] = 0
    result[result < 0] = 1
    return torch.tensor(torch.sum(result).item() / len(positives)), difference_distance.mean()
    
class TripletLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin=margin
    
    def forward(self, anchor, postive, negative):
        distance_postive=torch.nn.functional.pairwise_distance(anchor, postive)
        distance_negative=torch.nn.functional.pairwise_distance(anchor, negative)
        loss=torch.nn.functional.relu(distance_postive - distance_negative + self.margin)
        # extra postive -- negative pair distance
        return distance_postive.detach(), distance_negative.detach(), loss.mean()

criterion=TripletLoss()       

class SiameseBase(nn.Module):
    def cal_loss(self, batch, ones=None, zeros=None):
        # anchor_img, positive_img = batch[0], batch[1]
        anc_emb, anc_pred = self(batch[0]) 
        pos_emb, pos_pred = self(batch[1])
        batch[0], batch[1] = batch[0].cpu(), batch[1].cpu()
        
        for neg_item in range(len(batch) - 2):
            # negative_img = batch[2 + neg_item]
            neg_emb, neg_pred = self(batch[2 + neg_item])
            batch[2 + neg_item] = batch[2 + neg_item].cpu()
            if neg_item == 0:
                distance_postive, distance_negative, loss = criterion(anc_emb, pos_emb, neg_emb)
            else:
                _, n, l = criterion(anc_emb, pos_emb, neg_emb)
                loss+=l
        return loss, distance_postive, distance_negative, anc_emb, anc_pred, pos_emb, pos_pred, neg_emb, neg_pred

    def training_step(self, batch):
        # ones = torch.ones(batch[0].shape[0], dtype=torch.int64)
        # zeros = torch.zeros(batch[0].shape[0], dtype=torch.int64)
        loss, distance_postive, distance_negative, anc_emb, anc_pred, pos_emb, pos_pred, neg_emb, neg_pred = self.cal_loss(batch)
        train_acc, difference_distance = accuracy(distance_postive.detach(), distance_negative.detach())
        return loss, train_acc.detach(), difference_distance
    
    def validation_step(self, batch):
        # ones = torch.ones(batch[0].shape[0], dtype=torch.int64)
        # zeros = torch.zeros(batch[0].shape[0], dtype=torch.int64)
        loss, distance_postive, distance_negative, anc_emb, anc_pred, pos_emb, pos_pred, neg_emb, neg_pred = self.cal_loss(batch)
        acc, difference_distance = accuracy(distance_postive.detach(), distance_negative.detach())
        return {'val_loss': loss.detach(), 'distance_postive':torch.abs(distance_postive).mean(), 'distance_negative':torch.abs(distance_negative).mean()
                ,'val_acc': acc.detach(), 'difference_distance': difference_distance
                }
                # 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        # print(batch_losses, batch_accs)
        epoch_acc = torch.stack(batch_accs).mean()   # Combine losses
        batch_distance_postive= [x['distance_postive'] for x in outputs]
        epoch_distance_postive = torch.abs(torch.stack(batch_distance_postive)).mean()
        batch_distance_negative = [x['distance_negative'] for x in outputs]
        epoch_distance_negative = torch.abs(torch.stack(batch_distance_negative)).mean()
        batch_difference_distance = [x['difference_distance'] for x in outputs]
        epoch_difference_distance = torch.abs(torch.stack(batch_difference_distance)).mean()
        return {'val_loss': epoch_loss.item(), 'distance_postive':epoch_distance_postive.item(), 'distance_negative':epoch_distance_negative.item()
                ,'val_acc': epoch_acc.item(), 'difference_distance': epoch_difference_distance.item()
                }
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, train_acc: {:.4f}, val_acc: {:.4f}, distance_postive: {:.4f}, distance_negative: {:.4f}, difference_distance: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['train_acc'], result['val_acc'], result['distance_postive'], result['distance_negative'], result['difference_distance']))

class pre_trained_model(SiameseBase):
    def __init__(self, num_classes):
        super().__init__()
        
        self.model = ViT('B_32_imagenet1k', pretrained=True)
        self.model.fc = nn.Sequential()

    def forward(self, xb):
        embedding_out = self.model(xb)
        return embedding_out, 0