import os
import tqdm
import torch
import config
import datasets
import tools.utils as utils
import torch.nn.functional as F
from tools.logging import Logger
from tools.loss import LossOperator
from models.model_gmm import CAFWM
from models.model_gen import ResUnetGenerator

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.args.stage = 'GEN'

        self.args = config.MetricsInit(self.args)
        self.logger = Logger(self.args, self.args.SaveFolder_GEN)
        self.lossOp = LossOperator(self.args)
        
        self.visualizer = utils.Visualizer(self.args, self.args.SaveFolder_GEN)
        self._build_model()
        self._get_dataloader()
        self._set_optimizer()
    
    def _set_optimizer(self):
        self.optimizer = torch.optim.AdamW([
            {"params": self.net_warp.parameters(), 'lr':self.args.lr*0.2}, 
            {"params": self.net_gen.parameters()}], 
            lr=self.args.lr, betas=[self.args.beta1, 0.999]
        )
    
    def _get_dataloader(self):
        train_dataset = datasets.ImagesDataset(self.args, phase='train')
        self.train_dataLoader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.args.BatchSize, shuffle=True, persistent_workers=True, pin_memory=True, num_workers=self.args.NumWorkers)
        self.args.len_dataset = len(self.train_dataLoader)

    def _build_model(self):
        self.net_warp = CAFWM(self.args).to(self.args.device)
        self.net_gen  = ResUnetGenerator(32, 4, 5, ngf=64).to(self.args.device)

        self.logger.print_network(self.net_gen, 'net')
        utils.weights_initialization(self.net_gen, 'xavier', gain=1.0)
        weight = torch.load(os.path.join(self.args.RootCheckpoint_GMM, 'checkpoint.best.pth.tar'))
        self.net_warp.load_state_dict(weight['state_dict'])

    def _set_mode(self, is_train=True):
        self.net_warp.train() if is_train else self.net_warp.eval()
        self.net_gen.train() if is_train else self.net_gen.eval()

    def _visualize(self, vis_list, order):
        de_norm = lambda x: (x + 1) / 2 * 255.
        vis_list = [de_norm(image).detach().cpu() for image in vis_list]
        self.visualizer.vis_train(vis_list, order)
                           
    def _train_one_epoch(self):
        self._set_mode()
        num_layers = 4
        with tqdm.tqdm(self.train_dataLoader, desc="training") as pbar:
            for idx, sample in enumerate(pbar):
                image       = sample['image'].to(self.args.device)
                cloth       = sample['cloth'].to(self.args.device)
                cloth_mask  = sample['cloth_mask'].to(self.args.device)
                agnostic    = sample['agnostic'].to(self.args.device)
                person_shape= sample['person_shape'].to(self.args.device)
                person_clothes = sample['person_clothes'].to(self.args.device)
                person_clothes_mask = sample['person_clothes_mask'].to(self.args.device)
                image_grid  = sample['image_grid'].to(self.args.device)

                loss_warp = 0
                loss_mask = 0
                loss_styles = 0
                loss_smooth = 0
                loss_contents = 0
                loss_laplacian = 0
                with torch.set_grad_enabled(True):
                    # Warping Network
                    output = self.net_warp(cloth, cloth_mask, person_shape)
                    for i in range(num_layers):
                        warped_mask    = output['warping_masks'][i]
                        warped_cloth   = output['warping_cloths'][i]
                        shape_last_flow = output['shape_last_flows'][i]
                        cloth_last_flow = output['cloth_last_flows'][i]
                        shape_delta_flow = output['shape_delta_flows'][i]
                        cloth_delta_flow = output['cloth_delta_flows'][i]

                        _, _, h, w = warped_mask.shape
                        person_clothes_ = F.interpolate(person_clothes, size=(h, w))
                        person_clothes_mask_ = F.interpolate(person_clothes_mask, size=(h, w))

                        loss_content, loss_style = self.lossOp.calc_vgg_loss(warped_cloth, person_clothes_)
                        loss_warp     += self.lossOp.criterion_L1(warped_cloth, person_clothes_) * (i+1)
                        loss_mask     += self.lossOp.criterion_L1(warped_mask, person_clothes_mask_) * (i+1)
                        loss_laplacian+= (self.lossOp.calc_laplacian_loss(shape_last_flow) + self.lossOp.calc_laplacian_loss(cloth_last_flow)) / 2 * (i+1) * 6
                        loss_smooth   += (self.lossOp.calc_total_variation_loss(shape_delta_flow) + self.lossOp.calc_total_variation_loss(cloth_delta_flow)) / 2 * 0.01
                        loss_contents += loss_content * (i+1) * 0.2
                        loss_styles   += loss_style * (i+1) * 10
                    warp_loss = loss_contents + loss_styles + loss_laplacian + loss_smooth + loss_mask + loss_warp
                    
                    # Generative Network
                    warped_mask = output['warping_masks'][-1]
                    warped_cloth = output['warping_cloths'][-1]
                    gen_inputs = torch.cat([agnostic, warped_cloth, warped_mask], dim=1)
                    gen_outputs = self.net_gen(gen_inputs)

                    p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
                    p_rendered = torch.tanh(p_rendered)
                    m_composite = torch.sigmoid(m_composite)
                    m_composite1 = m_composite * warped_mask
                    m_composite =  person_clothes_mask * m_composite1
                    p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)

                    loss_mask_l1 = torch.mean(torch.abs(1 - m_composite))
                    loss_l1 = self.lossOp.criterion_L1(p_tryon, image)
                    loss_vgg, loss_style = self.lossOp.calc_vgg_loss(p_tryon, image)

                    bg_loss_l1 = self.lossOp.criterion_L1(p_rendered, image)
                    bg_loss_vgg, bg_loss_style = self.lossOp.calc_vgg_loss(p_rendered, image)
                    gen_loss = (loss_l1 * 5 + loss_vgg + bg_loss_l1 * 5 + bg_loss_vgg + loss_mask_l1)

                    loss = 0.5 * warp_loss + 1.0 * gen_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()


                if idx % 1000 == 0:
                    warped_cloth = output['warping_cloths'][-1]
                    cloth_last_flows = output['cloth_last_flows_'][-1]
                    warped_grid = self.net_warp.dec_tryon.spatial_transform(image_grid, cloth_last_flows, padding_mode='zeros')
                    self._visualize([warped_grid, warped_cloth, (warped_cloth + image)*0.5, image, p_tryon], order=1)
                self.logger.loss_tmp['wloss-content'] += loss_contents.item()
                self.logger.loss_tmp['wloss-mask']    += loss_mask.item()
                self.logger.loss_tmp['wloss-style']   += loss_styles.item()
                self.logger.loss_tmp['wloss-lapla']   += loss_laplacian.item()
                self.logger.loss_tmp['wloss-smooth']  += loss_smooth.item()
                self.logger.loss_tmp['wloss-warp']    += loss_warp.item()

                self.logger.loss_tmp['gloss-vgg-bg'] += bg_loss_l1.item()
                self.logger.loss_tmp['gloss-con-bg'] += bg_loss_vgg.item()
                self.logger.loss_tmp['gloss-con']    += loss_l1.item()
                self.logger.loss_tmp['gloss-vgg']    += loss_vgg.item()
                self.logger.loss_tmp['gloss-mask']   += loss_mask_l1.item()
                self.logger.loss_tmp['gloss-gen']    += gen_loss.item()
                
                self.logger.loss_tmp['loss-total']   += loss.item()
                pbar.set_description(f"Epoch: {self.args.epoch}, Loss: {self.logger.loss_tmp['loss-total'] / (idx + 1):.4f}")
            self.logger.loss_tmp = {k : v / self.args.len_dataset for k, v in self.logger.loss_tmp.items()}
        return self.logger.loss_tmp['loss-total']


    def train(self):
        print(self.args)
        for epoch in range(0, self.args.epochs):
            utils.flush()
            self.args.epoch = epoch + 1

            loss = self._train_one_epoch()
            
            self.logger.Log_PerEpochLoss()
            self.logger.Reset_PerEpochLoss()
            self.visualizer.plot_loss(self.logger.loss_history)
            
            utils.save_checkpoint(self.args.RootCheckpoint_GEN, {
                'args': self.args, 
                'GEN_state_dict': self.net_gen.state_dict(),
                'GMM_state_dict': self.net_warp.state_dict(),
            }, loss < self.args.best_loss)
            self.args.best_loss = min(loss, self.args.best_loss)


if __name__ == '__main__':
    trainer = Trainer(config.GetConfig())
    trainer.train()

