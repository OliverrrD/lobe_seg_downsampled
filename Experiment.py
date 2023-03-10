"""Class for setting up loading experiment configs_old and visualizing/evaluating results quickly"""
import torch
import os
import glob
from pathlib import Path
from monai.utils import set_determinism
from monai.inferers import sliding_window_inference
from dataloader import val_dataloader
import paral_clip_overlay_mask as overlay

CONFIG_DIR = "/local_directory/lobe_seg/configs"

class Experiment():
    def __init__(self, config, config_id, model):
        self.config = config
        self.config_id = config_id
        self.device = torch.device("cuda:0")
        self.model = model(6).to(self.device)
        set_determinism(seed=config["random_seed"])
        images = sorted(glob.glob(os.path.join(config["test_dir"], config["image_type"])))
        self.val_loader = val_dataloader(self.config, images)
        self.vis_dir = os.path.join(config["tmp_dir"], "clips", config_id)
        Path(self.vis_dir).mkdir(parents=True, exist_ok=True)

    def vis_checkpoint(self, checkpoint_name, nvals=1):
        """Visualize single checkpoint"""
        checkpoint_path = os.path.join(self.config['checkpoint_dir'], self.config_id, checkpoint_name)
        checkpoint = torch.load(checkpoint_path)
        epoch = checkpoint["epoch"]
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.vis(epoch, nvals)

    def vis_model(self, model_name, nvals=1):
        """Visualize best model"""
        model_path = os.path.join(self.config['model_dir'], self.config_id, model_name)
        epoch = "best"
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.vis(epoch, nvals)

    def vis(self, epoch, nvals):
        with torch.no_grad():
            loader = iter(self.val_loader)
            for i in range(nvals):
                val_data = next(loader)
                roi_size = (160, 160, 160)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(val_data["image"].to(self.device), roi_size, sw_batch_size, self.model)
                np_img = val_data["image"][0, 0, :, :, :].detach().cpu().numpy()
                np_label = val_data["label"][0, 0, :, :, :].detach().cpu().numpy()
                output = torch.argmax(val_outputs, dim=1)[0,:,:,:].detach().cpu().numpy()

                overlay.multiple_clip_overlay_with_mask_from_npy(np_img, np_label, os.path.join(self.vis_dir, f"label_{i}_{self.config_id}_{epoch}_coronal.png"), img_vrange=(0,1), clip_plane='coronal')
                overlay.multiple_clip_overlay_with_mask_from_npy(np_img, np_label, os.path.join(self.vis_dir, f"label_{i}_{self.config_id}_{epoch}_sagittal.png"), img_vrange=(0,1), clip_plane='sagittal')
                overlay.multiple_clip_overlay_with_mask_from_npy(np_img, np_label, os.path.join(self.vis_dir, f"label_{i}_{self.config_id}_{epoch}_axial.png"), img_vrange=(0,1), clip_plane='axial')
                overlay.multiple_clip_overlay_with_mask_from_npy(np_img, output, os.path.join(self.vis_dir, f"val_{i}_{self.config_id}_{epoch}_coronal.png"), img_vrange=(0,1), clip_plane='coronal')
                overlay.multiple_clip_overlay_with_mask_from_npy(np_img, output, os.path.join(self.vis_dir, f"val_{i}_{self.config_id}_{epoch}_sagittal.png"), img_vrange=(0,1), clip_plane='sagittal')
                overlay.multiple_clip_overlay_with_mask_from_npy(np_img, output, os.path.join(self.vis_dir, f"val_{i}_{self.config_id}_{epoch}_axial.png"), img_vrange=(0,1), clip_plane='axial')

    def vis_checkpoints(self, checkpoint_names, nvals=1):
        """visualize multiple checkponts to observe change over epochs"""
        for checkpoint_name in checkpoint_names:
            checkpoint_path = os.path.join(self.config['checkpoint_dir'], self.config_id, checkpoint_name)
            checkpoint = torch.load(checkpoint_path)
            epoch = checkpoint["epoch"]
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            with torch.no_grad():
                loader = iter(self.val_loader)
                for i in range(nvals):
                    val_data = next(loader)
                    roi_size = (160, 160, 160)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_data["image"].to(self.device), roi_size, sw_batch_size,
                                                           self.model)
                    np_img = val_data["image"][0, 0, :, :, :].detach().cpu().numpy()
                    np_label = val_data["label"][0, 0, :, :, :].detach().cpu().numpy()
                    output = torch.argmax(val_outputs, dim=1)[0, :, :, :].detach().cpu().numpy()

                    overlay.multiple_clip_overlay_with_mask_from_npy(np_img, np_label, os.path.join(self.vis_dir,
                                                                                                f"label_{self.config_id}_{epoch}_coronal.png"), img_vrange=(0,1),
                                                                 clip_plane='coronal')
                    overlay.multiple_clip_overlay_with_mask_from_npy(np_img, np_label, os.path.join(self.vis_dir,
                                                                                                f"label_{self.config_id}_{epoch}_sagittal.png"), img_vrange=(0,1),
                                                                 clip_plane='sagittal')
                    overlay.multiple_clip_overlay_with_mask_from_npy(np_img, np_label, os.path.join(self.vis_dir,
                                                                                                f"label_{self.config_id}_{epoch}_axial.png"), img_vrange=(0,1),
                                                                 clip_plane='axial')
                    overlay.multiple_clip_overlay_with_mask_from_npy(np_img, output, os.path.join(self.vis_dir,
                                                                                              f"val_{self.config_id}_{epoch}_coronal.png"), img_vrange=(0,1),
                                                                 clip_plane='coronal')
                    overlay.multiple_clip_overlay_with_mask_from_npy(np_img, output, os.path.join(self.vis_dir,
                                                                                              f"val_{self.config_id}_{epoch}_sagittal.png"), img_vrange=(0,1),
                                                                 clip_plane='sagittal')
                    overlay.multiple_clip_overlay_with_mask_from_npy(np_img, output, os.path.join(self.vis_dir,
                                                                                              f"val_{self.config_id}_{epoch}_axial.png"), img_vrange=(0,1),
                                                                 clip_plane='axial')