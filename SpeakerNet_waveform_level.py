import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy, math, pdb, sys, random
import time, os, itertools, shutil, importlib
import argparse
import numpy as np
#import tensorboard

from tuneThreshold import tuneThresholdfromScore
from DatasetLoader import test_dataset_loader
from torch.cuda.amp import autocast, GradScaler
#from torch.utils.tensorboard import SummaryWriter


class WrappedModel(nn.Module):

    ## The purpose of this wrapper is to make the model structure consistent between single and multi-GPU

    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.module = model

    def forward(self, mixed_inputs, label=None, permuted_label = None, lam = 1):
        return self.module(mixed_inputs, label, permuted_label, lam)


class SpeakerNet(nn.Module):

    def __init__(self, model, optimizer, trainfunc, nPerSpeaker, **kwargs):
        super(SpeakerNet, self).__init__()

        SpeakerNetModel = importlib.import_module('models.' + model).__getattribute__('MainModel')
        self.__S__ = SpeakerNetModel(**kwargs)

        LossFunction = importlib.import_module('loss.' + trainfunc).__getattribute__('LossFunction')
        self.__L__ = LossFunction(**kwargs)

        self.nPerSpeaker = nPerSpeaker

    def forward(self, mixed_inputs, original_label=None, permuted_label = None, lam = 1):
        mixed_inputs = mixed_inputs.reshape(-1, mixed_inputs.size()[-1]).cuda()
        mixed_outp = self.__S__.forward(mixed_inputs)

        if original_label is None:
            return mixed_outp
        else:
            mixed_outp = mixed_outp.reshape(self.nPerSpeaker, -1, mixed_outp.size()[-1]).transpose(1, 0).squeeze(1)
            nloss, prec1 = self.__L__.forward(mixed_outp, original_label, permuted_label, lam)
            return nloss, prec1


class ModelTrainer(object):

    def __init__(self, speaker_model, optimizer, scheduler, gpu, mixedprec, alpha, save_path, **kwargs):

        self.__model__ = speaker_model

        Optimizer = importlib.import_module('optimizer.' + optimizer).__getattribute__('Optimizer')
        self.__optimizer__ = Optimizer(self.__model__.parameters(), **kwargs)

        Scheduler = importlib.import_module('scheduler.' + scheduler).__getattribute__('Scheduler')
        self.__scheduler__, self.lr_step = Scheduler(self.__optimizer__, **kwargs)

        self.scaler = GradScaler()

        self.gpu = gpu

        self.mixedprec = mixedprec

        self.alpha = alpha

        self.save_path = save_path

        assert self.lr_step in ['epoch', 'iteration']

    # ## ===== ===== ===== ===== ===== ===== ===== =====
    # ## Train network
    # ## ===== ===== ===== ===== ===== ===== ===== =====

    def train_network(self, loader, verbose, epoch = None):

        self.__model__.train()

        stepsize = loader.batch_size

        counter = 0;
        index = 0;
        loss = 0;
        top1 = 0;  # EER or accuracy

        tstart = time.time()
        alpha = self.alpha

        for data, data_label in loader:

            original_inputs = data
            original_inputs = original_inputs.transpose(1, 0)

            mixed_inputs, original_labels, permuted_labels, lam = self.mixup_data(data, data_label, alpha, use_cuda=True, epoch = epoch)
            mixed_inputs = mixed_inputs.transpose(1, 0)

            self.__model__.zero_grad()

            original_labels = torch.LongTensor(original_labels).cuda()
            permuted_labels = torch.LongTensor(permuted_labels).cuda()

            if self.mixedprec:
                with autocast():
                    nloss, prec1 = self.__model__(data, label)
                self.scaler.scale(nloss).backward();
                self.scaler.step(self.__optimizer__);
                self.scaler.update();
            else:
                nloss, prec1 = self.__model__(mixed_inputs, original_labels, permuted_labels, lam)
                nloss.backward()
                self.__optimizer__.step()

            loss += nloss.detach().cpu().item();
            top1 += prec1.detach().cpu().item();
            counter += 1;
            index += stepsize;

            #self.writer.add_scalar('Loss/train', float(loss), counter)
            #self.writer.add_scalar('Acc/train', float(top1), counter)

            telapsed = time.time() - tstart
            tstart = time.time()

            if verbose:
                sys.stdout.write("\rProcessing {:d} of {:d}:".format(index, loader.__len__() * loader.batch_size));
                sys.stdout.write("Loss {:f} TEER/TAcc {:2.3f}% - {:.2f} Hz ".format(loss / counter, top1 / counter,
                                                                                    stepsize / telapsed));
                sys.stdout.flush();

            if self.lr_step == 'iteration': self.__scheduler__.step()

        if self.lr_step == 'epoch': self.__scheduler__.step()

        return (loss / counter, top1 / counter);

    def mixup_data(self, inputs, labels = None, alpha=0, use_cuda=True, epoch = None):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        if alpha == -1 and epoch is not None:
            high = min(1.0, max(0, epoch-1)/600.0)
            lam = 1.0-np.random.uniform(high=high)

        '''
        0715 audio normalization
        '''
        inputs_scale = ((inputs ** 2).sum(axis=-1) / inputs.shape[-1]) ** 0.5 + 0.001  # [B, N, T] for 0 volume case.
        inputs_scale = inputs_scale.mean() / inputs_scale
        inputs = inputs * inputs_scale.unsqueeze(-1)

        batch_size = inputs.size()[0]
        if use_cuda:
            index_memo = torch.randperm(batch_size)
            index = index_memo.cuda()
        else:
            index = torch.randperm(batch_size)

        mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :, :]
        mixed_inputs [:, 1:, :] = inputs[:, 1:, :]

        original_labels = torch.from_numpy(numpy.asarray(range(0, batch_size)))
        permuted_labels = index_memo

        return mixed_inputs.detach(), original_labels.detach(), permuted_labels.detach(), lam

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Evaluate from list
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def evaluateFromList(self, test_list, test_path, nDataLoaderThread, distributed, print_interval=100, num_eval=10,
                         **kwargs):

        if distributed:
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        self.__model__.eval()

        lines = []
        files = []
        feats = {}
        tstart = time.time()

        ## Read all lines
        with open(test_list) as f:
            lines = f.readlines()

        ## Get a list of unique file names
        files = list(itertools.chain(*[x.strip().split()[-2:] for x in lines]))
        setfiles = list(set(files))
        setfiles.sort()

        ## Define test data loader
        test_dataset = test_dataset_loader(setfiles, test_path, num_eval=num_eval, **kwargs)

        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
        else:
            sampler = None

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=nDataLoaderThread,
            drop_last=False,
            sampler=sampler
        )

        ## Extract features for every image
        for idx, data in enumerate(test_loader):
            inp1 = data[0][0].cuda()
            with torch.no_grad():
                ref_feat = self.__model__(inp1).detach().cpu()
            feats[data[1][0]] = ref_feat
            telapsed = time.time() - tstart

            if idx % print_interval == 0 and rank == 0:
                sys.stdout.write(
                    "\rReading {:d} of {:d}: {:.2f} Hz, embedding size {:d}".format(idx, test_loader.__len__(),
                                                                                    idx / telapsed,
                                                                                    ref_feat.size()[1]))

        all_scores = [];
        all_labels = [];
        all_trials = [];

        if distributed:
            ## Gather features from all GPUs
            feats_all = [None for _ in range(0, torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(feats_all, feats)

        if rank == 0:

            tstart = time.time()
            print('')

            ## Combine gathered features
            if distributed:
                feats = feats_all[0]
                for feats_batch in feats_all[1:]:
                    feats.update(feats_batch)

            ## Read files and compute all scores
            for idx, line in enumerate(lines):

                data = line.split()

                ## Append random label if missing
                if len(data) == 2: data = [random.randint(0, 1)] + data

                ref_feat = feats[data[1]].cuda()
                com_feat = feats[data[2]].cuda()

                if self.__model__.module.__L__.test_normalize:
                    ref_feat = F.normalize(ref_feat, p=2, dim=1)
                    com_feat = F.normalize(com_feat, p=2, dim=1)

                dist = F.pairwise_distance(ref_feat.unsqueeze(-1),
                                           com_feat.unsqueeze(-1).transpose(0, 2)).detach().cpu().numpy()

                score = -1 * numpy.mean(dist)

                all_scores.append(score)
                all_labels.append(int(data[0]))
                all_trials.append(data[1] + " " + data[2])

                if idx % print_interval == 0:
                    telapsed = time.time() - tstart
                    sys.stdout.write("\rComputing {:d} of {:d}: {:.2f} Hz".format(idx, len(lines), idx / telapsed));
                    sys.stdout.flush();

        return (all_scores, all_labels, all_trials);

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Save parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def saveParameters(self, path):

        torch.save(self.__model__.module.state_dict(), path);

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Load parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def loadParameters(self, path):

        self_state = self.__model__.module.state_dict();
        loaded_state = torch.load(path, map_location="cuda:%d" % self.gpu);
        for name, param in loaded_state.items():
            origname = name;
            if name not in self_state:
                name = name.replace("module.", "");

                if name not in self_state:
                    print("{} is not in the model.".format(origname));
                    continue;

            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: {}, model: {}, loaded: {}".format(origname, self_state[name].size(),
                                                                                 loaded_state[origname].size()));
                continue;

            self_state[name].copy_(param);