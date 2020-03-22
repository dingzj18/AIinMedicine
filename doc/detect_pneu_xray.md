# Pneunomia detection

# ChestX-ray14 dataset

We can download the data here(link).

To know the characteristics about this data, we should refer to the paper [@RajpurkarCheXNet]

Try an equation$E = m \dot C^2 $

# CheXNet

## An initial implementation

This imlementation follows [here](https://github.com/arnoweng/CheXNet).

### Define the dataset
    import os
    import numpy as np
    import torch
    from torch.utils.data import Dataset
    from PIL import Image
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    class ChestXrayDataset(Dataset):
    def __init__(self, data_dir, image_list_file, transform):
        '''
        :param data_dir: path to image directory
        :param image_list_file: path to the file containig images with corresponding labels
        :param transform: optinmal transform to be applied on a sample
        '''
        image_names = []
        labels = []
        with open(image_list_file, 'r') as f:
            for line in f:
                items = line.split()
                image_name = items[0]
                label = items[1:]
                label = [int(ix) for ix in label]
                image_name = os.path.join(data_dir, image_name)
                image_names.append( image_name )
                labels.append( label )
        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        '''
        :param index: the index of item
        :return: image and its labels
        '''
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)


Note that there are truncated images, we should use "ImageFile.LOAD_TRUNCATED_IMAGES = True".  As a begginer, we know that in the data class, we should define '__getitem__' and '__len__'.

After defining the data class, we can use the DataLoader, to load data via the data class.

    from torch.utils.data import DataLoader
    train_txt_path = 'to_your_train'
    valid_txt_path = 'to_your_valid'

    # define the transforms for each image
    normMean = [0.485, 0.456, 0.406]
    normStd = [0.229, 0.224, 0.225]
    normTransform = transforms.Normalize(normMean, normStd)

    trainTransform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(256, padding=4),
        transforms.ToTensor(),
        normTransform
    ])

    validTransform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        normTransform
    ])


    train_data = ChestXrayDataset(data_dir='iamge_path',
                                  image_list_file= train_txt_path,
                                  transform = trainTransform)
    valid_data = ChestXrayDataset(data_dir='image_path',
                                  image_list_file=valid_txt_path,
                                  transform=validTransform)

Then we can use the DataLoader.

    train_loader = DataLoader(dataset=train_data, batch_size=train_bs, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=valid_bs)

### Define the network
Before we define the model, some parameters should be specified.

    from datetime import datetime
    import torch.nn as nn
    from torch.autograd import Variable
    import torch.optim as optim
    import torch.backends.cudnn as cudnn
    import torchvision
    import torchvision.transforms as transforms
    from tensorboardX import SummaryWriter

    N_CLASSES = 14
    CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
                   'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
                   'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

    # log
    result_dir = '/home/zijianding/CheXNet/results'

    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    log_dir = os.path.join('/home/zijianding/CheXNet/log', time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)               

    writer = SummaryWriter(log_dir = log_dir)



Now we define the network structure.

    class DenseNet121(nn.Module):
        '''
        Define DenseNet121
        '''
        def __init__(self, out_size):
            super(DenseNet121, self).__init__()
            self.densenet121 = torchvision.models.densenet121(pretrained=True)
            num_ftrs = self.densenet121.classifier.in_features
            self.densenet121.classifier = nn.Linear(num_ftrs, out_size)
            # self.densenet121.classifier = nn.Sequential(
            #     nn.Linear(num_ftrs, out_size),
            #     nn.Sigmoid()
            # )
        def forward(self, x):
            return self.densenet121(x)


As a begginer, we should know that the backward function is done by Pytorch with the forward function.
### Train the network
Before we begin to describe the training process, we should know that there are three necessary steps in a trainig process: (1) initialize all parameters from scratch or from pretrained model;  (2) in each epoch and each batch, make all graduates as zero in the first place; (3) run the forward function; (4) run the backward function, namely calcualte the loss; (5) determine whether stop training, otherwise update parameters;

    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    cuda_gpu = torch.cuda.is_available()
    cudnn.benchmark = True   

    train_bs = 32
    valid_bs = 32
    lr_init = 0.001
    max_epoch = 1

    # initialize the network with pretrained parameters
    net = DenseNet121(N_CLASSES).cuda()
    net = torch.nn.DataParallel(net).cuda()

    # define loss function, optimizing algorithm and learning rate schedular
    criteria = nn.BCEWithLogitsLoss( pos_weight=torch.ones(N_CLASSES) ).cuda()
    optimizer = optim.SGD(net.parameters(), lr=lr_init, momentum=0.9, dampening=0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    # the training process
    for epoch in range(max_epoch):

        loss_sigma = 0.0 # sum loss
        correct = 0.0
        total = 0.0
        scheduler.step() # schedule the learning rate

        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = torch.autograd.Variable( inputs.cuda() ), torch.autograd.Variable( labels.cuda() )

            # forward, backward, update weights
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criteria(outputs, labels)
            loss.backward()
            optimizer.step() # ???

            # 统计预测结果

            loss_sigma += loss.item()

            # print loss
            if i % 10 == 9:
                loss_avg = loss_sigma / 10
                loss_sigma = 0.0
                print( "Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f}".format(
                    epoch + 1, max_epoch, i + 1, len(train_loader), loss_avg) )

                # 记录训练loss
                writer.add_scalars('Loss_group', {'train_loss': loss_avg}, epoch)
                # 记录learning rate
                writer.add_scalar('learning_rate', scheduler.get_lr()[0], epoch)
                # 记录Accuracy
                # NULL


        # 每个epoch, 记录梯度，权值
        for name, layer in net.named_parameters():
            writer.add_histogram(name + '_grad', layer.grad.cpu().data.numpy(), epoch)
            writer.add_histogram(name + '_data', layer.cpu().data.numpy(), epoch)

        #---------观察在验证集上的表现-------------#
        if epoch % 2 == 0:
            loss_sigma_2 = 0.0
            conf_mat = np.zeros([N_CLASSES, N_CLASSES])
            net.eval()
            for i, data in enumerate(valid_loader):
                 images, labels = data
                 images, labels = torch.autograd.Variable( images.cuda() ), torch.autograd.Variable( labels.cuda() )

                 outputs = net(images)
                 outputs.detach_() # block the backward

                 loss = criteria(outputs, labels)
                 loss_sigma_2 += loss.item()

      print('Finish Training')
      # save the model
      net_save_path = os.path.join(result_dir, 'net_params.pkl')
      torch.save(net.state_dict(), net_save_path)



### Test the network

Using the test data, we want to check the performance of the trained network.

    from sklearn.metrics import roc_auc_score
    CKPT_PATH = '/home/zijianding/CheXNet/results/net_params.pkl'
    DATA_DIR = '/home/zijianding/Datasets/ChestX-ray14/images'
    TEST_IMAGE_LIST = '/home/zijianding/Datasets/ChestX-ray14/labels/test_list_merge.txt'
    BATCH_SIZE = 32

    # first define the evaluating metrics
    def Compute_AUCs(gt, pred):
        '''
        compute AUC based on prediction scores
        :param gt: groud truth
        :param pred: predictions
        :return: list of AUC for all classes
        '''
        AUROCs = []
        gt_np = gt.cpu().numpy() # why?
        pred_np = pred.cpu().numpy()
        for i in range(N_CLASSES):
            AUROCs.append(roc_auc_score(gt_np[:,i], pred_np[:,i]))
        return AUROCs

    # load the model
    cudnn.benchmark = True
    # use_cuda = torch.cuda.is_available()
    # device = torch.device('cuda:4,5' if use_cuda else 'cpu')

    # load the model
    model = DenseNet121(N_CLASSES).cuda()
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(CKPT_PATH))

    # load the test data
    test_dataset = ChestXrayDataset(data_dir=DATA_DIR,
                                    image_list_file=TEST_IMAGE_LIST,
                                    transform=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.ToTensor(),
                                        normTransform
                                    ]))

    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()

    # switch to evaluate mode
    model.eval()
    # make predictions
    with torch.no_grad():
       for i, data in enumerate(test_loader):
           inp, target = data
           target = torch.autograd.Variable(target.cuda())
           gt = torch.cat((gt, target), 0)
           # bs, n_crops, c, h, w = inp.size()
           # input_var = torch.autograd.Variable(inp.view(-1, c, h, w).cuda(), volatile=True)
           output = model(torch.autograd.Variable(inp.cuda()))
           func = nn.Sigmoid() # to probabilities
           output = func(output)
           # output_mean = output.view(bs, n_crops, -1).mean(1)
           pred = torch.cat((pred, output), 0)

    # calculate the metrics
    AUROCs = Compute_AUCs(gt, pred)
    AUROC_avg = np.array(AUROCs).mean()
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(N_CLASSES):
        print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs[i]))
