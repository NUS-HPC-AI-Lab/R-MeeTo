"""
Implements the knowledge distillation loss
"""
import torch
from torch.nn import functional as F


class DistillDiffPruningLoss_dynamic(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """

    def __init__(self, teacher_model, base_criterion: torch.nn.Module, dynamic=False, clf_weight=0, mse_token=False,
                 print_mode=True, distill_weight=1):
        super().__init__()
        self.teacher_model = teacher_model
        self.base_criterion = base_criterion
        self.clf_weight = clf_weight
        self.count = 0
        self.print_mode = print_mode
        self.cls_loss = 0
        self.cls_distill_loss = 0
        self.mse_token = mse_token
        self.dynamic = dynamic
        self.distill_weight = distill_weight
        if dynamic:
            print('using dynamic loss')

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """

        pred = outputs
        # pred, token_pred, mask, out_pred_score = outputs

        cls_loss = self.base_criterion(pred, labels)

        with torch.no_grad():

            cls_t, token_t = self.teacher_model(inputs)

        cls_kl_loss = F.kl_div(
            F.log_softmax(pred, dim=-1),
            F.log_softmax(cls_t, dim=-1),
            reduction='batchmean',
            log_target=True
        )

        loss_part = []
        loss = self.clf_weight * cls_loss + self.distill_weight * cls_kl_loss

        if torch.isnan(loss):
            if self.print_mode:
                print('loss is nan')
                print('cls_loss', cls_loss)
                print('cls_kl_loss', cls_kl_loss)

        if self.print_mode:
            self.cls_loss += cls_loss.item()

            self.cls_distill_loss += cls_kl_loss.item()

            loss_part.append(cls_loss)

            loss_part.append(cls_kl_loss)

            self.count += 1
            if self.count == 100:
                print('loss info: cls_loss=%.4f, cls_kl=%.4f' % (self.cls_loss / 100, self.cls_distill_loss / 100))
                self.count = 0
                self.cls_loss = 0
                self.cls_distill_loss = 0
        return loss
