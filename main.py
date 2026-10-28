import os
import argparse
import torch
import warnings
from trainer import cross_domain_trainer
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

parser = argparse.ArgumentParser()


# ========  Experiments Name ================
parser.add_argument('--save_dir',               default='experiments_logs',         type=str, help='Directory containing all experiments')
parser.add_argument('--experiment_description', default='collect_results',          type=str, help='Name of your experiment')
parser.add_argument('--run_description',        default='cotmix_dinov3_uci_har',         type=str, help='name of your runs')
parser.add_argument('--sweep_project',          default='cotmix_hparams_sweep',     type=str, help='name of your sweep project')

# ========= Select the DA methods ============
parser.add_argument('--da_method',              default='CoTMix',      type=str, help='Our method - but can include any method implemented in algorithms file')

# ========= Select the DATASET ==============
parser.add_argument('--dataset',                default='HAR',        type=str, help='Dataset of choice: WISDM, EEG, HAR, HHAR')

# ========= Select the BACKBONE ==============
parser.add_argument('--backbone',               default='DINOv3ViTS16', type=str,
                    choices=['CNN', 'DINOv3ViTS16'], help='Feature encoder / backbone')
parser.add_argument('--dinov3_model_name', default='facebook/dinov3-vits16-pretrain-lvd1689m', type=str,
                    help='Hugging Face model ID for DINOv3 ViT-S/16')
parser.add_argument('--dinov3_image_size', default=224, type=int, help='DINOv3 image input size')
parser.add_argument('--dinov3_lr', default=1e-5, type=float, help='Learning rate for unfrozen DINOv3')
parser.add_argument('--classifier_lr', default=1e-3, type=float, help='Learning rate for classifier')
parser.add_argument('--unfreeze_dinov3', action='store_true',
                    help='Fine-tune DINOv3 instead of freezing the pretrained backbone')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size (DINOv3 default: 16)')
parser.add_argument('--epochs', default=50, type=int, help='Number of epochs')


# ========= Experiment settings ===============
parser.add_argument('--data_path',              default=r'./data/',    type=str,   help='Path containing dataset')
parser.add_argument('--num_runs',               default=3,             type=int,   help='Number of consecutive run with different seeds')
parser.add_argument('--device',                 default='cuda:0',      type=str,   help='cpu or cuda')
parser.add_argument('--is_sweep', action='store_true', help='Run a Weights & Biases hyperparameter sweep')
parser.add_argument('--num_sweeps',             default=100,           type=int,   help='Number of sweep runs')

args = parser.parse_args()

if __name__ == "__main__":
    trainer = cross_domain_trainer(args)

    if args.is_sweep:
        trainer.sweep()
    else:
        trainer.train()

save_path = "cotmix_uci_har.pkl"
torch.save(trainer.algorithm, save_path)
print(f"Đã xuất mô hình thành công tại: {save_path}")