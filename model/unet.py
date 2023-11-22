import matplotlib.pyplot as plt
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch

from model.dataloader import URLS

class UnetModel(pl.LightningModule):

    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1)) #from 1 3 1 1

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.training_step_tp, self.training_step_fp, self.training_step_tn, self.training_step_fn = [], [], [], []
        self.validation_step_tp, self.validation_step_fp, self.validation_step_tn, self.validation_step_fn = [], [], [], []
        self.test_step_tp, self.test_step_fp, self.test_step_tn, self.test_step_fn = [], [], [], []


    def forward(self, image):
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):

        image = batch["image"]

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32,
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch["mask"]

        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, stage):
        # aggregate step metics
        if stage == "train":
            tp = torch.cat([x for x in self.training_step_tp])
            fp = torch.cat([x for x in self.training_step_fp])
            fn = torch.cat([x for x in self.training_step_fn])
            tn = torch.cat([x for x in self.training_step_tn])
        if stage == "valid":
            tp = torch.cat([x for x in self.validation_step_tp])
            fp = torch.cat([x for x in self.validation_step_fp])
            fn = torch.cat([x for x in self.validation_step_fn])
            tn = torch.cat([x for x in self.validation_step_tn])
        if stage == "test":
            tp = torch.cat([x for x in self.test_step_tp])
            fp = torch.cat([x for x in self.test_step_fp])
            fn = torch.cat([x for x in self.test_step_fn])
            tn = torch.cat([x for x in self.test_step_tn])


        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")

        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "train")["loss"]
        self.training_step_outputs.append(loss)

        tp = self.shared_step(batch, "train")["tp"]
        self.training_step_tp.append(tp)
        tn = self.shared_step(batch, "train")["tn"]
        self.training_step_tn.append(tn)
        fp = self.shared_step(batch, "train")["fp"]
        self.training_step_fp.append(fp)
        fn = self.shared_step(batch, "train")["fn"]
        self.training_step_fn.append(fn)
        return loss

    def on_training_epoch_end(self):
        epoch_average = torch.stack(self.training_step_outputs).mean()
        self.log("training_epoch_average", epoch_average)
        self.shared_epoch_end("train")
        self.training_step_outputs.clear()
        self.training_step_tp.clear()
        self.training_step_fp.clear()
        self.training_step_tn.clear()
        self.training_step_fn.clear()

    def validation_step(self, batch, batch_idx):
        loss =  self.shared_step(batch, "valid")["loss"]
        self.validation_step_outputs.append(loss)

        tp = self.shared_step(batch, "valid")["tp"]
        self.validation_step_tp.append(tp)
        tn = self.shared_step(batch, "valid")["tn"]
        self.validation_step_tn.append(tn)
        fp = self.shared_step(batch, "valid")["fp"]
        self.validation_step_fp.append(fp)
        fn = self.shared_step(batch, "valid")["fn"]
        self.validation_step_fn.append(fn)
        return loss

    def on_validation_epoch_end(self):
        epoch_average = torch.stack(self.validation_step_outputs).mean()
        self.log("validation_epoch_average", epoch_average)
        self.shared_epoch_end("valid")
        self.validation_step_outputs.clear()
        self.validation_step_tp.clear()
        self.validation_step_fp.clear()
        self.validation_step_tn.clear()
        self.validation_step_fn.clear()

    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "test")["loss"]
        self.test_step_outputs.append(loss)
        tp = self.shared_step(batch, "test")["tp"]
        self.test_step_tp.append(tp)
        tn = self.shared_step(batch, "test")["tn"]
        self.test_step_tn.append(tn)
        fp = self.shared_step(batch, "test")["fp"]
        self.test_step_fp.append(fp)
        fn = self.shared_step(batch, "test")["fn"]
        self.test_step_fn.append(fn)
        return loss

    def on_test_epoch_end(self):
        epoch_average = torch.stack(self.test_step_outputs).mean()
        self.log("test_epoch_average", epoch_average)
        self.shared_epoch_end("test")
        self.test_step_outputs.clear()  # free memory
        self.test_step_tp.clear()
        self.test_step_fp.clear()
        self.test_step_tn.clear()
        self.test_step_fn.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

class DairyUnet:
    def __init__(self):
        self.model = UnetModel("Unet", "resnet50", in_channels=3, out_classes=3)
        self.model.load_state_dict(torch.load('unet_weights/unet_best2.pth'))
        self.model.eval()
        self.model.to('cuda')

    def infer(self, input):
        num_chunks = 9
        chunk_size = 224
        output_tensor = torch.empty(3, 2016, 2016)

        for i in range(num_chunks):
            for j in range(num_chunks):
                start_i, start_j = i * chunk_size, j * chunk_size
                end_i, end_j = start_i + chunk_size, start_j + chunk_size
                chunk = input[:, start_i:end_i, start_j:end_j]

                out = self.model(chunk.unsqueeze(0)).squeeze(0)
                output_tensor[:, start_i:end_i, start_j:end_j] = out.detach()
        return output_tensor
        # return model(input.unsqueeze(0)).squeeze(0).detach().cpu()

    def make_mask(self, img):
        return (img.permute(1,2,0) >= 0.5).to(torch.float32)

    def amount_white(self, mask):
        return torch.sum(mask).item() // 3

    def print_mask(self, input_img):
        plt.figure(figsize=(10,8))
        plt.tight_layout()
        
        plt.subplot(1,2,1)
        plt.axis('off')
        plt.imshow(input_img.permute(1,2,0))

        img = self.infer(input_img.to('cuda'))
        input_img.detach()
        mask = self.make_mask(img)
        num_white = self.amount_white(mask)
        print(num_white)

        plt.subplot(1,2,2)
        plt.axis('off')
        plt.imshow(mask)
        plt.title(f'num pixels: {int(num_white)}')
        plt.show()

class PoultryUnet:
    def __init__(self):
        self.model = smp.Unet(encoder_name="resnet18", encoder_depth=3,
                              encoder_weights=None, decoder_channels=(128, 64, 64),
                              in_channels=4, classes=2)
        self.model.load_state_dict(torch.load('unet_weights/train-all_unet_0.5_0.01_rotation_best-checkpoint.pt')['model_checkpoint'])
        self.model.to('cuda')
        self.model.eval()

    def infer(self, input):
        num_chunks = 8
        chunk_size = 256
        output_tensor = torch.empty(2, 2048, 2048)

        for i in range(num_chunks):
            for j in range(num_chunks):
                start_i, start_j = i * chunk_size, j * chunk_size
                end_i, end_j = start_i + chunk_size, start_j + chunk_size
                chunk = input[:, start_i:end_i, start_j:end_j]

                out = self.model(chunk.unsqueeze(0)).squeeze(0)
                output_tensor[:, start_i:end_i, start_j:end_j] = out.detach()
        return output_tensor

    def make_mask(self, img):
        soft = torch.nn.functional.softmax(img, dim=0)
        return (soft[1] > 0.2).to(torch.float32)

    def amount_white(self, mask):
        return torch.sum(mask).item()

    def print_mask(self, input_img):
        plt.figure(figsize=(10,8))
        plt.tight_layout()
        
        plt.subplot(1,2,1)
        plt.axis('off')
        plt.imshow(input_img.permute(1,2,0))

        img = self.infer(input_img.to('cuda'))
        input_img.detach()
        mask = self.make_mask(img)
        num_white = self.amount_white(mask)
        print(num_white)

        plt.subplot(1,2,2)
        plt.axis('off')
        plt.imshow(mask)
        plt.title(f'num pixels: {int(num_white)}')
        plt.show()
