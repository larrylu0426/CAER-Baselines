from PIL import ImageDraw
import torch
import torchvision
from tqdm import tqdm
import wandb

from horizon.base.trainer import BaseTrainer
import horizon.data_loader as module_data
from horizon.utils.visualizer import get_masked_image


class BaseCAERS(BaseTrainer):

    def __init__(self, config, model, device):
        super().__init__(config, model, device)

    def _add_data_loaders(self):
        train_data_loader = self.config.init_obj('dataset',
                                                 module_data,
                                                 phase="train",
                                                 shuffle=True)
        test_data_loader = self.config.init_obj('dataset',
                                                module_data,
                                                phase="test",
                                                val_split=1.0 / 3)
        val_data_loader = test_data_loader.split_validation()

        return train_data_loader, val_data_loader, test_data_loader

    def _train(self):
        self.model.train()
        self.train_metrics.reset()
        for _, (data, target) in tqdm(enumerate(self.train_data),
                                      total=len(self.train_data),
                                      leave=True):
            self.optimizer.zero_grad()
            face, context = data['face'].to(self.device), data['context'].to(
                self.device)
            target = target.to(self.device)
            output, _ = self.model(face, context)
            loss = self.criterion(output, target)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))
            loss.backward()
            self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return self.train_metrics.result()

    def _valid(self, model):
        model.eval()
        self.val_metrics.reset()
        with torch.no_grad():
            for _, (data, target) in enumerate(self.val_data):
                face, context = data['face'].to(
                    self.device), data['context'].to(self.device)
                target = target.to(self.device)
                output, _ = model(face, context)
                loss = self.criterion(output, target)
                self.val_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.val_metrics.update(met.__name__, met(output, target))

        return self.val_metrics.result()

    def _test(self, model):
        model.eval()
        self.test_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(self.test_data)):
                face, context = data['face'].to(
                    self.device), data['context'].to(self.device)
                target = target.to(self.device)
                output, options = model(face, context)
                loss = self.criterion(output, target)
                self.test_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.test_metrics.update(met.__name__, met(output, target))
                if batch_idx == 0 and self.config.wandb:
                    pred = torch.argmax(output, dim=1)
                    self._log_image_table(data, pred, target, output, options)

        return self.test_metrics.result()

    def _log_image_table(self, data, predicted, labels, probs, options):
        label_map = [
            'Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'
        ]
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        table = wandb.Table(columns=[
            "image", "focus", "face", "context", "context_attn", "target",
            "prediction"
        ] + [label_map[i] for i in range(7)])
        attn = options
        for img, bbox, face, context, attn, pred, targ, prob in zip(
                data['image'].to("cpu"), data['bbox'].to("cpu"),
                data['face'].to("cpu"), data['context'].to("cpu"),
                attn.to("cpu"), predicted.to("cpu"), labels.to("cpu"),
                probs.to("cpu")):
            _, H, W = img.shape
            t_mean = torch.FloatTensor(mean).view(3, 1, 1).expand(3, H, W)
            t_std = torch.FloatTensor(std).view(3, 1, 1).expand(3, H, W)
            focus = img * t_std + t_mean
            focus = torchvision.transforms.functional.to_pil_image(focus)
            draw = ImageDraw.Draw(focus)
            draw.rectangle(bbox.tolist(), outline='red', width=2)
            table.add_data(
                wandb.Image(img), wandb.Image(focus), wandb.Image(face),
                wandb.Image(context),
                wandb.Image(
                    get_masked_image(image=context,
                                     mean=mean,
                                     std=std,
                                     attn=attn)), label_map[targ],
                label_map[pred], *prob.numpy())
        wandb.log({"predictions_table": table}, commit=False)
