import time
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from timeit import default_timer as timer
from OppModeling.utils import mlp


class CPC(nn.Module):
    def __init__(self, timestep, obs_dim, hidden_sizes,z_dim, c_dim):
        super(CPC, self).__init__()
        self.timestep = timestep
        self.c_dim = c_dim
        self.z_dim = z_dim
        self.encoder = mlp([obs_dim] + list(hidden_sizes) + [z_dim], nn.ReLU, nn.Identity)
        self.gru = nn.GRU(z_dim, c_dim, num_layers=1, bidirectional=False, batch_first=True)
        self.Wk = nn.ModuleList([nn.Linear(c_dim, z_dim) for i in range(timestep)])
        self.softmax = F.softmax
        self.lsoftmax = F.log_softmax

        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # initialize gru
        for layer_p in self.gru._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(_weights_init)

    def init_hidden(self, batch_size, feature_dim, use_gpu=True):
        if use_gpu:
            return torch.zeros(1, batch_size, feature_dim).cuda()
        else:
            return torch.zeros(1, batch_size, feature_dim)

    def get_z(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        if len(x.shape) == 1:
            x = x.unsqueeze(dim=0)
        if len(x.shape) == 2:
            x = x.unsqueeze(dim=0)
        batch = x.size()[0]
        seq_len = x.size()[1]
        obs_dim = x.size()[2]
        z = torch.empty((batch, seq_len, self.z_dim)).float()  # e.g. size 12*8*512
        for i in range(seq_len):
            z[:, i, :] = self.encoder(x[:, i, :])
        return z, batch, seq_len, obs_dim

    def forward(self, x, c_hidden):
        # input sequence is N*C*L, e.g. 8*1*20480
        z, batch, seq_len, obs_dim = self.get_z(x)
        # no Down sampling in RL
        assert self.timestep < seq_len
        t_samples = torch.randint(seq_len - self.timestep, size=(1,)).long()  # randomly pick time stamps
        # Do not need transpose as the input shape is N*L*C
        # z = z.transpose(1, 2)
        nce = 0  # average over timestep and batch
        encode_samples = torch.empty((self.timestep, batch, self.z_dim)).float()  # e.g. size 12*8*512
        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = z[:, (t_samples + i).long(), :].view(batch, self.z_dim)  # z_tk e.g. size 8*512
        forward_seq = z[:, :t_samples + 1, :]  # e.g. size 8*100*512

        # calculate the full trace latent for the buffer update, so freeze the gradient in this step
        with torch.no_grad():
            latents, _ = self.gru(z, c_hidden)  # output size e.g. 8*100*256
        output, _ = self.gru(forward_seq, c_hidden)  # output size e.g. 8*100*256
        c_t = output[:, t_samples, :].view(batch, self.c_dim)  # c_t e.g. size 8*256
        pred = torch.empty((self.timestep, batch, self.z_dim)).float()  # e.g. size 12*8*512
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t)  # Wk*c_t e.g. size 8*512
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))  # e.g. size 8*8
            correct = torch.sum(
                torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, batch)))  # correct is a tensor
            nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
        nce /= -1. * batch * self.timestep
        accuracy = 1. * correct.item() / batch

        return accuracy, nce, latents # return output to update the latent value of the buffer

    @torch.no_grad()
    def predict(self, x, c_hidden):
        z, _, _, _ = self.get_z(x)
        output, c_hidden = self.gru(z, c_hidden)# output size e.g. 8*128*256
        return output, c_hidden # return every frame


def train(args, model, device, train_loader, optimizer, epoch, batch_size):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.float().unsqueeze(1).to(device)  # add channel dimension
        optimizer.zero_grad()
        c_hidden = model.init_hidden(len(data), args.c_dim, use_gpu=True)
        acc, loss, hidden = model(data, c_hidden)

        loss.backward()
        # add gradient clipping
        nn.utils.clip_grad_norm(model.parameters(), 20)
        optimizer.step()
        lr = optimizer.update_learning_rate()


def validation(args, model, device, data_loader, batch_size):
    logging.info("Starting Validation")
    model.eval()
    total_loss = 0
    total_acc = 0

    with torch.no_grad():
        for data in data_loader:
            data = data.float().unsqueeze(1).to(device)  # add channel dimension
            hidden = model.init_hidden(len(data), args.c_dim,use_gpu=True)
            acc, loss, hidden = model(data, hidden)
            total_loss += len(data) * loss
            total_acc += len(data) * acc

    total_loss /= len(data_loader.dataset)  # average loss
    total_acc /= len(data_loader.dataset)  # average acc

    logging.info('===> Validation set: Average loss: {:.4f}\tAccuracy: {:.4f}\n'.format(
        total_loss, total_acc))

    return total_acc, total_loss


class ScheduledOptim(object):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, n_warmup_steps):
        self.optimizer = optimizer
        self.d_model = 128
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.delta = 1

    def state_dict(self):
        self.optimizer.state_dict()

    def step(self):
        """Step by the inner optimizer"""
        self.optimizer.step()

    def zero_grad(self):
        """Zero out the gradients by the inner optimizer"""
        self.optimizer.zero_grad()

    def increase_delta(self):
        self.delta *= 2

    def update_learning_rate(self):
        """Learning rate scheduling per step"""

        self.n_current_steps += self.delta
        new_lr = np.power(self.d_model, -0.5) * np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr


def main():
    # Settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--train-raw', required=True)
    parser.add_argument('--validation-raw', required=True)
    parser.add_argument('--eval-raw')
    parser.add_argument('--train-list', required=True)
    parser.add_argument('--validation-list', required=True)
    parser.add_argument('--eval-list')
    parser.add_argument('--logging-dir', required=True,
                        help='model save directory')
    parser.add_argument('--epochs', type=int, default=60, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--n-warmup-steps', type=int, default=50)
    parser.add_argument('--timestep', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--z_dim', type=int, default=64,
                        help='batch size')
    parser.add_argument('--c_dim', type=int, default=32,
                        help='batch size')
    parser.add_argument('--audio-window', type=int, default=20480,
                        help='window length to sample from each utterance')

    parser.add_argument('--masked-frames', type=int, default=20)
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print('use_cuda is', use_cuda)
    global_timer = timer()  # global timer
    device = torch.device("cuda" if use_cuda else "cpu")
    model = CPC(args.timestep, args.batch_size, args.audio_window).to(device)
    ## Loading the dataset
    params = {'num_workers': 0,
              'pin_memory': False} if use_cuda else {}

    logging.info('===> loading train, validation and eval dataset')

    # nanxin optimizer
    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True),
        args.n_warmup_steps)

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info('### Model summary below###\n {}\n'.format(str(model)))
    logging.info('===> Model total parameter: {}\n'.format(model_params))
    ## Start training
    best_acc = 0
    best_loss = np.inf
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        epoch_timer = timer()

        # Train and validate
        # trainXXreverse(args, model, device, train_loader, optimizer, epoch, args.batch_size)
        # val_acc, val_loss = validationXXreverse(args, model, device, validation_loader, args.batch_size)
        train(args, model, device, train_loader, optimizer, epoch, args.batch_size)
        val_acc, val_loss = validation(args, model, device, validation_loader, args.batch_size)

        # Save
        if val_acc > best_acc:
            best_acc = max(val_acc, best_acc)
            # TODO add save functions
            best_epoch = epoch + 1
        elif epoch - best_epoch > 2:
            optimizer.increase_delta()
            best_epoch = epoch + 1

        end_epoch_timer = timer()
        logging.info("#### End epoch {}/{}, elapsed time: {}".format(epoch, args.epochs, end_epoch_timer - epoch_timer))

    ## end
    end_global_timer = timer()
    logging.info("################## Success #########################")
    logging.info("Total elapsed time: %s" % (end_global_timer - global_timer))


if __name__ == '__main__':
    ############ Control Center and Hyperparameter ###############
    # TODO add the unit test for CPC only, try on the mnist data set
    run_name = "cdc" + time.strftime("-%Y-%m-%d_%H_%M_%S")
    print(run_name)
    main()
