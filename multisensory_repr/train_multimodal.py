import torch
torch.cuda.empty_cache()
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import h5py
import numpy as np
import cv2

import os
import argparse
import init_path
from models.model_utils import safe_cuda, Modality_input, SensorFusion

class MultiModalDataset(Dataset):
    def __init__(self, hdf5_file):
        f = h5py.File(hdf5_file, "r")

        self.total_len = f["data"].attrs["len"]

        self.f = f
        
        self.ft_states = safe_cuda(torch.from_numpy(self.f["data/ft_states"][()]))
        self.contact_states = safe_cuda(torch.from_numpy(self.f["data/contact_states"][()]))

        self.proprio_states = safe_cuda(torch.from_numpy(self.f["data/proprio_states"][()]))

        self.agentview_images = safe_cuda(torch.from_numpy((self.f["data/agentview_images"][()])))
        self.eye_in_hand_images = safe_cuda(torch.from_numpy(self.f["data/eye_in_hand_images"][()]))

        f.close()
        print("Finish loading")
    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):

        ft_state = self.ft_states[idx].float()
        contact_state = self.contact_states[idx].squeeze().float()
        proprio_state = self.proprio_states[idx].float()

        agentview_image = self.agentview_images[idx].float()
        eye_in_hand_image = self.eye_in_hand_images[idx].float()
        
        return ft_state, contact_state, proprio_state, agentview_image, eye_in_hand_image

def kl_normal(qm, qv, pm, pv):
    element_wise = 0.5 * (
        torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1
    )
    kl = element_wise.sum(-1)
    return kl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset-name',
        type=str,
        default="training_set",
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        '--alpha-kl',
        type=float,
        default=0.05,
    )
    parser.add_argument(
        '--alpha-force',
        type=float,
        default=1.0,
    )
    parser.add_argument(
            '--z-dim',
            type=int,
            default=128
    )

    parser.add_argument(
        '--num-workers',
        type=int,
        default=0,
    )

    parser.add_argument(
        '--num-epochs',
        type=int,
        default=100,
    )

    parser.add_argument(
        "--modalities",
        type=str,
        nargs="+",
        default=["agentview", "eye_in_hand", "force"]
    )

    parser.add_argument(
        "--no-skip",
        action="store_true"
    )

    parser.add_argument(
        '--use-checkpoint',
        action="store_true"
    )

    args = parser.parse_args()
    
    dataset = MultiModalDataset(f"datasets/{args.dataset_name}/demo.hdf5")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    data = []
    modalities = args.modalities
    modality_str = modalities[0]
    for modality in modalities[1:]:
        modality_str += f"_{modality}"
    modality_str += f"_{args.alpha_kl}"

    if args.no_skip:
        modality_str += "_no_skip"
    sensor_fusion = safe_cuda(SensorFusion(z_dim=args.z_dim, use_skip_connection=not args.no_skip, modalities=modalities))

    if args.use_checkpoint:
        sensor_fusion.load_state_dict(torch.load(f"results/{args.dataset_name}/Fusion_{modality_str}_{args.z_dim}_checkpoint.pth"))
        print("loaded checkpoint")
    optimizer = torch.optim.Adam(sensor_fusion.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=50)

    if args.no_skip:
        reduction = 'sum'
    else:
        reduction = 'mean'
    
    reconstruction_loss = torch.nn.MSELoss(reduction=reduction)
    bce_loss = torch.nn.BCELoss(reduction=reduction)
    n_epochs = args.num_epochs

    os.makedirs(f"results/{args.dataset_name}/imgs", exist_ok=True)
    last_loss = None
    for epoch in range(n_epochs):
        sensor_fusion.train()
        training_loss = 0
        for (ft, contact_state, proprio, img_1, img_2) in dataloader:

            ft = safe_cuda(ft)
            proprio = safe_cuda(proprio)
            img_1 = safe_cuda(img_1 / 255.)
            img_2 = safe_cuda(img_2 / 255.)

            x = Modality_input(frontview=None, agentview=img_1, eye_in_hand=img_2, force=ft, proprio=proprio)

            output, mu_z, var_z, mu_prior, var_prior = sensor_fusion(x)

            k = mu_z.size()[1]
            if args.no_skip:
                
                loss = args.alpha_kl * torch.sum(kl_normal(mu_z, var_z, mu_prior.squeeze(0), var_prior.squeeze(0)))
            else:
                loss = args.alpha_kl * torch.mean(kl_normal(mu_z, var_z, mu_prior.squeeze(0), var_prior.squeeze(0)))

            if 'agentview' in modalities:
                loss = loss + reconstruction_loss(img_1, output.agentview_recon)
            if 'eye_in_hand' in modalities:
                loss = loss + reconstruction_loss(img_2, output.eye_in_hand_recon)
            if 'force' in modalities:
                loss = loss + args.alpha_force * bce_loss(output.contact.squeeze(1), contact_state)
            if 'proprio' in modalities:
                loss = loss + reconstruction_loss(proprio, output.proprio) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

        print(f"Training loss: {training_loss}")
            
        if epoch % 10 == 0:
            sensor_fusion.eval()
            ft = safe_cuda(ft)
            proprio = safe_cuda(proprio)

            x = Modality_input(frontview=None, agentview=img_1, eye_in_hand=img_2, force=ft, proprio=proprio)

            output, _, _, _, _ = sensor_fusion(x)
            
            if 'agentview' in modalities and 'eye_in_hand' in modalities:
                output_agentview = (output.agentview_recon * 255.).detach().cpu().numpy().astype(np.uint8).transpose(0, 2, 3, 1)
                output_eye_in_hand = (output.eye_in_hand_recon * 255.).detach().cpu().numpy().astype(np.uint8).transpose(0, 2, 3, 1)

                example_agentview_img = []
                example_eye_in_hand_img = []

                for i in range(output_agentview.shape[0]):
                    example_agentview_img.append(output_agentview[i])
                    example_eye_in_hand_img.append(output_eye_in_hand[i])

                final_img_agentview = np.concatenate(example_agentview_img, axis=1)
                final_img_eye_in_hand = np.concatenate(example_eye_in_hand_img, axis=1)

                cv2.imwrite(f"results/{args.dataset_name}/imgs/test_agentview_img_{epoch}.png", final_img_agentview)
                cv2.imwrite(f"results/{args.dataset_name}/imgs/test_eye_in_hand_img_{epoch}.png", final_img_eye_in_hand)

            if last_loss is None:
                last_loss = training_loss
            elif last_loss > training_loss:
                print("Saving checkpoint")
                torch.save(sensor_fusion.state_dict(), f"results/{args.dataset_name}/Fusion_{modality_str}_{args.z_dim}_checkpoint.pth")
                last_loss = training_loss
        scheduler.step(training_loss)
        if optimizer.param_groups[0]['lr'] < 1e-4:
            print("Learning rate became too low, stop training")
            break
    torch.save(sensor_fusion.state_dict(), f"results/{args.dataset_name}/Fusion_{modality_str}_{args.z_dim}.pth")
    print(f"Final saved loss: {training_loss}")

if __name__ == "__main__":
    main()


