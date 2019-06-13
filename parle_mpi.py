from mpi4py import MPI

import pickle

import torch as th
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import pyplot

import torch.nn as nn
import torchnet as tnt #TNT is a library providing powerful dataloading, logging and visualization utilities for Python.
from torchvision.datasets.mnist import MNIST, FashionMNIST
from torch.autograd import Variable  # Variables wrap a Tensor and it stores the gradient
import torch.backends.cudnn as cudnn # This flag allows you to enable \
                                     #the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.
cudnn.benchmark = True

import argparse, random, pdb
from copy import deepcopy

p = argparse.ArgumentParser('Parle',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# p.add_argument('--data', type=str, default='/local2/pra
# tikac/mnist', help='dataset')
p.add_argument('--data', type=str, default='Fashion MNIST', help='dataset')
p.add_argument('--lr', type=float, default=0.1, help='learning rate')
p.add_argument('-b', type=int, default=128, help='batch size')
p.add_argument('-L', type=int, default=25, help='prox. eval steps')
p.add_argument('--gamma', type=float, default=0.01, help='gamma')
p.add_argument('--rho', type=float, default=0.01, help='rho')
p.add_argument('-n', type=int, default=1, help='replicas')
opt = vars(p.parse_args())
print('the type of opt is: ', type(opt)) # First print statement ---------------------------------------------
comm = MPI.COMM_WORLD

opt['B'], opt['l2'] = 5, -1.0
opt['r'] = comm.Get_rank() ## The id of the cpu (rank)
opt['n'] = comm.Get_size() ## The number of cpu that we have which is different from the no. of gpus
print('No. of CPUs: ', opt['n'])
opt['rho'] = opt['rho']*opt['L']*opt['n'] # Momentum

opt['s'] = 42 + opt['r'] # Dont know why they did that
random.seed(opt['s']) # seeding a different no. for each id
np.random.seed(opt['s'])# seeding a different no. for each id
th.manual_seed(opt['s'])# seeding a different no. for each id

opt['cuda'] = th.cuda.is_available() ## checking for cuda environment


if opt['cuda']: 
    ngpus = th.cuda.device_count() ## Finding the number of gpus on a process
    print('No. of gpus in the environment are: ', ngpus) ## I am getting the number of gpus to be 4 if I use the command
    opt['g'] = int(opt['r'] % ngpus) # To provide one gpu to each of the rank
    print('Made the cpu of rank: {} assigned to GPU no.: {}'.format(opt['r'],opt['g']))
    th.cuda.set_device(opt['g']) # To provide one gpu to each of the rank
    th.cuda.manual_seed(opt['s']) # Dont know why they did that
print(opt)


def get_iterator(mode): ## Will be used in the training set, it is kind of a data loader
    ds = FashionMNIST(root=opt['data'], download=True, train=mode) ##Changed loader to 
    data = getattr(ds, 'train_data' if mode else 'test_data')      ##Fashion MNIST Loader such 
    labels = getattr(ds, 'train_labels' if mode else 'test_labels')##that it directly downloads the data
    tds = tnt.dataset.TensorDataset([data, labels])
    return tds.parallel(batch_size=opt['b'],
            num_workers=0, shuffle=mode, pin_memory=True)

class View(nn.Module):
    def __init__(self,o):
        super(View, self).__init__()
        self.o = o
    def forward(self,x):
        return x.view(-1, self.o)

def convbn(ci,co,ksz,psz,p):
    return nn.Sequential(
        nn.Conv2d(ci,co,ksz),
        nn.BatchNorm2d(co),
        nn.ReLU(True),
        nn.MaxPool2d(psz,stride=psz),
        nn.Dropout(p))

model = nn.Sequential(              ##To create a model
    convbn(1,20,5,3,0.25),
    convbn(20,50,5,2,0.25),
    View(50*2*2),
    nn.Linear(50*2*2, 500),
    nn.BatchNorm1d(500),
    nn.ReLU(True),
    nn.Dropout(0.25),
    nn.Linear(500,10),
    )
criterion = nn.CrossEntropyLoss()
#for p in model.parameters():
#    print(p.shape)

## Below are the parameters shape
#torch.Size([20, 1, 5, 5])
#torch.Size([20])
#torch.Size([20])
#torch.Size([20])
#torch.Size([50, 20, 5, 5])
#torch.Size([50])
#torch.Size([50])
#torch.Size([50])
#torch.Size([500, 200])
#torch.Size([500])
#torch.Size([500])
#torch.Size([500])
#torch.Size([10, 500])
#torch.Size([10])





if opt['cuda']:
    model = model.cuda() # Basically putting the net on Cuda
    criterion = criterion.cuda()

def parle_step(sync=False):
    eps = 1e-3

    mom, alpha = 0.9, 0.75  #momentum, acceleration parameters
    lr = opt['lr'] #other known parameters learning rate
    r = opt['r'] # rank
    nb = opt['nb'] # nb = 469 

    if not 'state' in opt:  ##defines and sets some parameters that it
        s = {} ##Initialized S as an empty dictionary
        t=0    ##Initialized a variable t
        s['t'] = t  ## s has a key 't' and its value is 0
        
        
        
        for k in ['za', 'muy', 'mux', 'xa', 'x', 'cache']: # now more key with values as dictionary
            s[k] = {}

        for p in model.parameters():
            for k in ['za', 'muy', 'mux', 'xa']:
                s[k][p] = p.data.clone() ##Clone is better than copy, kinda
            s['muy'][p].zero_()   ##like a shared memory location,the gradient
            s['mux'][p].zero_()   ##will propagate to the orig and the clone
                                  ##zero_ fills with zeros
            s['x'][p] = p.data.cpu().numpy()     ##So.. Just numpy 
            s['cache'][p] = p.data.cpu().numpy() ##versions of the same thing?
        opt['state'] = s
    else:
        s = opt['state'] ## This should be opposite -----------------------------------------------------------------
        t = s['t']

    za, muy, mux, xa, x, cache = s['za'], s['muy'], s['mux'], \
        s['xa'], s['x'], s['cache']

    gamma = opt['gamma']*(1 + 0.5/nb)**(t // opt['L'])
    rho = opt['rho']*(1 + 0.5/nb)**(t // opt['L'])
    gamma, rho = min(gamma, 1), min(rho, 10)

    def sync_with_master(xa, x):
        for p in model.parameters():
            xa[p] = xa[p].cpu().numpy()
            comm.Reduce(xa[p], s['cache'][p], op=MPI.SUM, root=0)
            xa[p] = th.from_numpy(xa[p])
            if opt['cuda']:
                xa[p] = xa[p].cuda()

        comm.Barrier()

        if r == 0:
            for p in model.parameters():
                x[p] = s['cache'][p]/float(opt['n'])

        for p in model.parameters():
            comm.Bcast(x[p], root=0)
        comm.Barrier()

    if sync: ## This thing is right this is parle algorithm
        # add another sync, helps with large L
        sync_with_master(za, x)

        for p in model.parameters():
            tmp = th.from_numpy(x[p])
            if opt['cuda']:
                tmp = tmp.cuda()

            # elastic-sgd term
            p.grad.data.zero_()
            p.grad.data.add_(1, xa[p] - za[p]).add_(rho, xa[p] - tmp) ### 8(c)

            mux[p].mul_(mom).add_(p.grad.data) ##(eta is momentum)
            p.grad.data.add_(mux[p])
            p.data.add_(-lr, p.grad.data)

            xa[p].copy_(p.data)
        sync_with_master(xa, x)
    else:
        # entropy-sgd iterations
        for p in model.parameters():
            p.grad.data.add_(gamma, p.data - xa[p])

            muy[p].mul_(mom).add_(p.grad.data)
            p.grad.data.add_(muy[p])
            p.data.add_(-lr, p.grad.data)

            za[p].mul_(alpha).add_(1-alpha, p.data)
            t=t+1

if(opt['r']==0):
    acc_list_train0 = []
    loss_list_train0 = []
if(opt['r']==1):
    acc_list_train1 = []
    loss_list_train1 = []
if(opt['r']==2):
    acc_list_train2 = []
    loss_list_train2 = []
if(opt['r']==3):
    acc_list_train3 = []
    loss_list_train3 = []


loss_list_val = []
acc_list_val = []
            
def train(e): ### e will range from 0 to 5 as we are using for e in range(opt['B']):
    model.train() ## to set it in the train state
    
    train_ds = get_iterator(True)
    train_iter = train_ds.__iter__()
    opt['nb'] = len(train_iter)   ## Number of batches which is 469
    print('No. of batch is: ', opt['nb'])

    loss = tnt.meter.AverageValueMeter()        ##Average Loss + std of loss
    top1 = tnt.meter.ClassErrorMeter()          ##Classification Error (Zero/One Loss)
                                                ##Top 1 just means its considering only the best prediction
    for b in range(opt['nb']): ## kind of mini batch around 469 opt['nb'] = 496
        retloss, reterror = [],[] ## two list
        for l in range(opt['L']): ## opt['L'] = 25 ------------------------------------------------------- Why it is 25
            try:
                x,y = next(train_iter)
            
            except StopIteration: ## To handel the exception "I dont know why is there any"
                train_iter = train_ds.__iter__()
                x,y = next(train_iter)

            x = Variable(x.view(-1,1,28,28).float() / 255.0)            ##Normalises the data I guess
            y = Variable(th.LongTensor(y))
            if opt['cuda']:
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

            model.zero_grad()             ##Empty gradient
            yh = model(x)                 ##Forward Pass
            f = criterion(yh, y)          ##Calculate Loss
            f.backward()                  ##Backpropagate

            if opt['l2'] > 0:
                for p in model.parameters():
                    p.grad.data.add_(opt['l2'], p.data)

            if l == 0:
                top1.add(yh.data, y.data)
                loss.add(f.item())
                
                if(opt['r']==0):
                    acc_list_train0.append(100-top1.value()[0])                   
                    loss_list_train0.append(loss.value()[0])
                    
                if(opt['r']==1):
                    acc_list_train1.append(100-top1.value()[0])
                    loss_list_train1.append(loss.value()[0])
                    
                if(opt['r']==2):
                    acc_list_train2.append(100-top1.value()[0])
                    loss_list_train2.append(loss.value()[0])
                    
                if(opt['r']==3):
                    acc_list_train3.append(100-top1.value()[0])
                    loss_list_train3.append(loss.value()[0])
            
            

                if b % 100 == 0 and b > 0:
                    print('Epoch:[%03d], BatchID/Mini Batches: [%03d/%03d], Avg Loss: %.3f, Classification Error: %.3f%%'%(e, b, opt['nb'], \
                            loss.value()[0], top1.value()[0]))

            parle_step()        ##Entropy SGD

        # setup value for sync
        #opt['state']['f'][0] = loss.value()[0]
        parle_step(sync=True)   ##Parle

    r = dict(f=loss.value()[0], top1=top1.value()[0])
    print('+[%02d] %.3f %.3f%%'%(e, r['f'], r['top1']))
    return r

def dry_feed(m):
    def set_dropout(cache = None, p=0):
        if cache is None:
            cache = []
            for l in m.modules():
                if 'Dropout' in str(type(l)):
                    cache.append(l.p)
                    l.p = p
            return cache
        else:
            for l in m.modules():
                if 'Dropout' in str(type(l)):
                    assert len(cache) > 0, 'cache is empty'
                    l.p = cache.pop(0)

    m.train()
    cache = set_dropout()
    train_iter = get_iterator(True)
    with th.no_grad():
        for _, (x,y) in enumerate(train_iter):
            x = Variable(x.view(-1,1,28,28).float() / 255.0)
            if opt['cuda']:
                x = x.cuda(non_blocking=True)
            m(x)
    set_dropout(cache)

def validate(e):
    m = deepcopy(model)
    for p,q in zip(m.parameters(), model.parameters()):
        tmp = th.from_numpy(opt['state']['x'][q])
        if opt['cuda']:
            tmp = tmp.cuda()
        p.data.copy_(tmp)

    dry_feed(m)
    m.eval()

    val_iter = get_iterator(False)

    loss = tnt.meter.AverageValueMeter()
    top1 = tnt.meter.ClassErrorMeter()

    for b, (x,y) in enumerate(val_iter):
        x = Variable(x.view(-1,1,28,28).float() / 255.0)
        y = Variable(th.LongTensor(y))
        if opt['cuda']:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

        yh = m(x)
        f = criterion(yh, y)

        top1.add(yh.data, y.data)
        loss.add(f.item())

    r = dict(f=loss.value()[0], top1=top1.value()[0])
    print('*[%02d] %.3f %.3f%%'%(e, r['f'], r['top1']))
    return r


training_loss, training_error = [],[] ##Both are list
for e in range(opt['B']): #opt['B'] = 5 may be these are the epocs
    if opt['r'] == 0: # I can only find opt['r'] = 0 # for the master thread
        print("Hey I am the master thread")
    
    
    if(opt['r']!=0): ## This is how I figured out. There is some problem with this
        print('Hey I am the thread No.: ', opt['r'])

    r= train(e) ## returning the dictionary, loss and error
    comm.Barrier() ## MPI stops but I dont know when it started

    if opt['r'] == 0:
        validate(e)     ##Master Thread assesses testing accuracy
    comm.Barrier() ## Why there are two barriers

    opt['lr'] /= 10.0
    

plt.figure()

if(opt['r']==0):
    #plt.plot(acc_list_train0,label='Train Accuracy on thread 0')
    #plt.savefig('Accuracy'+str(opt['r'])+'.png')
    with open('train0.pkl', 'wb') as f:
        pickle.dump(acc_list_train0, f)
        
    with open('loss0.pkl', 'wb') as f:
        pickle.dump(loss_list_train0, f)
        
if(opt['r']==1):
    #plt.plot(acc_list_train1,label='Train Accuracy on thread 1')
    #plt.savefig('Accuracy'+str(opt['r'])+'.png')
    with open('train1.pkl', 'wb') as f:
        pickle.dump(acc_list_train1, f)
        
    with open('loss1.pkl', 'wb') as f:
        pickle.dump(loss_list_train1, f)
        
if(opt['r']==2):
    #plt.plot(acc_list_train2,label='Train Accuracy on thread 2')
    #plt.savefig('Accuracy'+str(opt['r'])+'.png')
    with open('train2.pkl', 'wb') as f:
        pickle.dump(acc_list_train2, f)
        
    with open('loss2.pkl', 'wb') as f:
        pickle.dump(loss_list_train2, f)
        
if(opt['r']==3):
    #plt.plot(acc_list_train3,label='Train Accuracy on thread 3') 
    #plt.savefig('Accuracy'+str(opt['r'])+'.png')
    with open('train3.pkl', 'wb') as f:
        pickle.dump(acc_list_train3, f)
        
    with open('loss3.pkl', 'wb') as f:
        pickle.dump(loss_list_train3, f)
        
    
    
#plt.title('Accuracy vs Number of iterations')
#plt.legend(loc = 'lower right')
#plt.xlabel('No. of iterations')
#plt.ylabel('Accuracy')
#plt.grid(True)
