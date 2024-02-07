from PIL import ImageDraw
import torch
import torchvision
from tqdm import tqdm
import wandb

from horizon.base.trainer import BaseTrainer
import horizon.data_loader as module_data
from horizon.utils.util import MetricTracker


class BaseEmotic(BaseTrainer):

    def __init__(self, config, model, device):
        super().__init__(config, model, device)
        self.lambda_cat = 0.5
        self.lambda_cont = 1 - self.lambda_cat
        self.train_metrics = MetricTracker('loss')

    def _add_data_loaders(self):
        train_data_loader = self.config.init_obj('dataset',
                                                 module_data,
                                                 phase="train",
                                                 shuffle=True)
        test_data_loader = self.config.init_obj('dataset',
                                                module_data,
                                                phase="test")
        val_data_loader = self.config.init_obj('dataset',
                                               module_data,
                                               phase="val")

        return train_data_loader, val_data_loader, test_data_loader

    def _train(self):
        self.model.train()
        self.train_metrics.reset()
        for _, (data, target) in tqdm(enumerate(self.train_data),
                                      total=len(self.train_data),
                                      leave=True):
            self.optimizer.zero_grad()
            body, image = data['body'].to(self.device), data['image'].to(
                self.device)
            cat_label, cont_label = target['cat'].to(
                self.device), target['cont'].to(self.device)
            output = self.model(body, image)
            cat_loss = self.criterion['weighted_euclidean_loss'](output['cat'],
                                                                 cat_label)
            cont_loss = self.criterion['smooth_L1'](output['cont'], cont_label)
            loss = (self.lambda_cat * cat_loss) + (self.lambda_cont *
                                                   cont_loss)
            self.train_metrics.update('loss', loss.item())
            loss.backward()
            self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return self.train_metrics.result()

    def _valid(self, model):
        model.eval()
        self.val_metrics.reset()
        with torch.no_grad():
            outputs_cat = []
            outputs_cont = []
            cat_labels = []
            cont_labels = []
            for _, (data, target) in enumerate(self.val_data):
                body, image = data['body'].to(self.device), data['image'].to(
                    self.device)
                cat_label, cont_label = target['cat'].to(
                    self.device), target['cont'].to(self.device)

                output = model(body, image)
                cat_loss = self.criterion['weighted_euclidean_loss'](
                    output['cat'], cat_label)
                cont_loss = self.criterion['smooth_L1'](output['cont'],
                                                        cont_label)
                loss = (self.lambda_cat * cat_loss) + (self.lambda_cont *
                                                       cont_loss)

                self.val_metrics.update('loss', loss.item())
                outputs_cat.append(output['cat'])
                outputs_cont.append(output['cont'])
                cat_labels.append(cat_label)
                cont_labels.append(cont_label)
            outputs_cat = torch.cat(outputs_cat, dim=0)
            outputs_cont = torch.cat(outputs_cont, dim=0)
            cat_labels = torch.cat(cat_labels, dim=0)
            cont_labels = torch.cat(cont_labels, dim=0)
            for met in self.metric_ftns:
                if met.__name__ == 'mean_ap':
                    self.val_metrics.update(met.__name__,
                                            met(outputs_cat, cat_labels))
                if met.__name__ == 'mean_aae':
                    self.val_metrics.update(met.__name__,
                                            met(outputs_cont, cont_labels))

        return self.val_metrics.result()

    def _test(self, model):
        model.eval()
        self.test_metrics.reset()
        with torch.no_grad():
            outputs_cat = []
            outputs_cont = []
            cat_labels = []
            cont_labels = []
            for batch_idx, (data, target) in enumerate(tqdm(self.test_data)):
                body, image = data['body'].to(self.device), data['image'].to(
                    self.device)
                cat_label, cont_label = target['cat'].to(
                    self.device), target['cont'].to(self.device)

                output = model(body, image)
                cat_loss = self.criterion['weighted_euclidean_loss'](
                    output['cat'], cat_label)
                cont_loss = self.criterion['smooth_L1'](output['cont'],
                                                        cont_label)
                loss = (self.lambda_cat * cat_loss) + (self.lambda_cont *
                                                       cont_loss)
                self.test_metrics.update('loss', loss.item())
                outputs_cat.append(output['cat'])
                outputs_cont.append(output['cont'])
                cat_labels.append(cat_label)
                cont_labels.append(cont_label)
                if batch_idx == 0 and self.config.wandb:
                    self._log_image_table(data, target, output)
            outputs_cat = torch.cat(outputs_cat, dim=0)
            outputs_cont = torch.cat(outputs_cont, dim=0)
            cat_labels = torch.cat(cat_labels, dim=0)
            cont_labels = torch.cat(cont_labels, dim=0)
            for met in self.metric_ftns:
                if met.__name__ == 'mean_ap':
                    self.test_metrics.update(met.__name__,
                                             met(outputs_cat, cat_labels))
                if met.__name__ == 'mean_aae':
                    self.test_metrics.update(met.__name__,
                                             met(outputs_cont, cont_labels))

        return self.test_metrics.result()

    def _log_image_table(self, data, target, output):
        label_map = ['Affection', 'Anger', 'Annoyance', 'Anticipation', \
                'Aversion', 'Confidence', 'Disapproval', 'Disconnection', \
                'Disquietment', 'Doubt/Confusion', 'Embarrassment', \
                'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear',\
                'Happiness', 'Pain', 'Peace', 'Pleasure', 'Sadness', \
                'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning']
        mean = (0.4690646, 0.4407227, 0.40508908)
        std = (0.2514227, 0.24312855, 0.24266963)
        table = wandb.Table(columns=[
            "image", "focus", "body", "context", "VAD", "pred", "categories"
        ] + [label_map[i] for i in range(26)])
        for img, bbox, body, context, vad, pred, cats, prob in zip(
                data['image'].to("cpu"), data['bbox'].to("cpu"),
                data['body'].to("cpu"), data['context'].to("cpu"),
                target['cont'].to("cpu"), output['cont'].to("cpu"),
                target['cat'].to("cpu"), output['cat'].to("cpu")):
            _, H, W = img.shape
            t_mean = torch.FloatTensor(mean).view(3, 1, 1).expand(3, H, W)
            t_std = torch.FloatTensor(std).view(3, 1, 1).expand(3, H, W)
            focus = img * t_std + t_mean
            focus = torchvision.transforms.functional.to_pil_image(focus)
            draw = ImageDraw.Draw(focus)
            draw.rectangle(bbox.tolist(), outline='red', width=2)
            vad = ",".join([str(int(i)) for i in (vad * 10).tolist()])
            pred = ",".join([str(i) for i in (pred * 10).tolist()])
            targ_cats = [
                label_map[i] for i, cat in enumerate(cats) if cat == 1
            ]
            targ_cats = ", ".join(targ_cats)

            table.add_data(wandb.Image(img), wandb.Image(focus),
                           wandb.Image(body), wandb.Image(context), vad, pred,
                           targ_cats, *prob.numpy())
        wandb.log({"predictions_table": table}, commit=False)
