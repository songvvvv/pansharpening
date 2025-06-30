import os
from datetime import datetime
import torch
import torchvision
import torch.backends.cudnn as cudnn
import pdb
from torch.nn import functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data.dataloader
import numpy as np
import scipy.io as scio




from data import Datain
from mylib import to_var, loss_func29_h, loss_func_RB  # ,  loss_func32 # loss_func29_h
from quality_assessment import calc_psnr, calc_rmse, calc_ergas, calc_sam, calc_cc, calc_ssim
from args_parser import args_parser
from models.model import DEANet


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# __________________ set params   __________________________________
args = args_parser()
print(args)
with open('train.txt', 'a') as f:
    f.write('\n' + '\n' + '\n' + 'time:' + str(datetime.now()) + '\n')
    f.write('args:' + str(args) + '\n' + '\n')
f.close()   # 这段代码将当前时间、空行和参数信息写入到 train.txt 文件中，以记录训练过程中的一些信息


def main():
    #  ------------------ make dirs if they are not exit -----------------------------------------------
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)
    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)

    # -----------------------             ---------------------------------------------------------------
    #writer = SummaryWriter() # 用于记录训练过程中的数据，例如损失和准确率等
    # best_test_loss = np.inf  # infinity  # 无限大的初始值
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('==> gpu or cpu:', device, ', how many gpus available:', torch.cuda.device_count())
    # ----------------------- load data   ---------------------------------------------------------------

    ''' train part'''
    data_train = Datain(args.data_path_mat_train, args.scale_ratio,train=None)# 创建一个数据集对象
    train_loader = torch.utils.data.DataLoader(data_train,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=1)# 创建一个训练数据加载器
    '''val part'''
    data_val = Datain(args.data_path_mat_val, args.scale_ratio,train=None)# 创建一个数据集对象
    val_loader = torch.utils.data.DataLoader(data_val,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=1)# 创建一个验证数据加载器

    # -----------------------  load model   ---------------------------------------------------------------

    my_model = DEANet()

    if use_cuda:
        my_model.cuda()
        print("use cuda")

    start_epoch = 0

        # -----------------------  Loss and Optimizer  ------------------------------------------------------------

    criterion = nn.L1Loss().cuda()

    optimizer = torch.optim.Adam(my_model.parameters(), lr=args.learning_rate)  # gai 优化器

    '''some parameters needed'''
    # best_psnr = 0
    # best_rmse = np.inf
    best_rmse = 1
    # print('psnr: ', best_psnr)
    total_loss = 0
    # Epochs
    print('Start Training: ')
    my_model.train() # 训练模式
    #loss = np.inf   # 损失初始值
    loss_train_list = []
    accuracy_train_list = []
    loss_valid_list = []
    accuracy_valid_list = []
    '''train start'''
    for epoch in range(start_epoch, args.epochs):
        print(
            '_________________________________________________________________________________________________________'
            '_______________________________________________________________Train_Epoch_{}:__'.format(epoch))
        count = 0
        t1 = datetime.now()

        for batch_id, (ref, pan, ms) in enumerate(train_loader):
            count += 1
            print('Train_Epoch_{}:  ______________Train_iter_{}: '.format(epoch, batch_id))
            '''ms upsample'''
            # ms = F.interpolate(ms, scale_factor=args.scale_ratio, mode="bicubic")  # ms上采样

            bms = F.interpolate(ms, scale_factor=4, mode="bicubic")
            print(ms.shape)
            print(bms.shape)
            print(pan.shape)
            #pbms = torch.cat((pan, bms), dim=1)

            print("change size of ms")
            print(ms.shape)

            if use_cuda:
                pan = Variable(pan).cuda()
                ms = Variable(ms).cuda()
                ref = Variable(ref).cuda()

                bms = Variable(bms).cuda()

            out = my_model(bms, pan)


            loss = criterion(ref, out)

            print("__________________________________loss________________________________________________")
            print(loss)
            print("__________________________________loss________________________________________________")
            total_loss += loss.item()

            optimizer.zero_grad()  # set gradient=0，以免影响其他batch
            loss.backward()  # 后向传播，计算梯度
            print('后向传播，计算梯度')
            optimizer.step()  # 利用梯度更新w b 参数

        t2 = datetime.now()
        run_time = t2 - t1
        print('____________________________one epoch run_time_{}: '.format(run_time))

        # One epoch's validation
        print('Val_Epoch_{}: '.format(epoch))
        recent_ergas, rmse, psnr_mean, sam_mean = val(my_model, epoch, val_loader, args)  # , ssim
        print('recent_psnr: {}'.format(psnr_mean))

        # # save model


        if best_rmse==1:
            best_rmse = rmse
        else:
            is_best = rmse < best_rmse
            best_rmse = min(rmse, best_rmse)
            if is_best and best_rmse > 0:
                torch.save(my_model.state_dict(), args.model_path + '/model_epoch%d.pth' % epoch)
                print('Saved!')
                print('')

        print('best_rmse: ', best_rmse)

        # print("__________________________________loss.item()________________________________________________")

        total_loss /= count
        print('total iters: ', count)
        print('train epoch [%d/%d] average_loss %.5f' % (epoch, count, total_loss))


        # model save
        if epoch == args.epochs - 1:
            torch.save(my_model.state_dict(),
                       args.model_path + '/model_epoch%d.pth' % epoch)  # save model belongs to the last epoch

        #writer.close()


def val(model, epoch, val_loader, args):

    rmse_list = []
    psnr_list = []
    ergas_list = []
    sam_list = []

    for batch_id, (ref_val, pan_val, ms_val) in enumerate(val_loader):
        # ref_val_down = F.interpolate(ref_val, scale_factor=0.5, mode="bicubic")
        # ref_val_down2 = F.interpolate(ref_val_down, scale_factor=0.5, mode="bicubic")
        # ms_val = F.interpolate(ms_val, scale_factor=args.scale_ratio, mode="bicubic")
        model.eval()

        # psnr = 0
        with torch.no_grad():
            '''Set mini-batch dataset'''
            ref = to_var(ref_val).detach()
            ms = to_var(ms_val).detach()
            pan = to_var(pan_val).detach()
            bms = F.interpolate(ms, scale_factor=args.scale_ratio, mode="bicubic")

            out = model(bms, pan)


            ref = ref.detach().cpu().numpy()
            out = out.detach().cpu().numpy()

            rmse = calc_rmse(ref, out)
            psnr = calc_psnr(ref, out)
            ergas = calc_ergas(ref, out)
            sam = calc_sam(ref, out)
            # ssim = calc_ssim(ref, out)

            rmse_list.append(rmse)
            psnr_list.append(psnr)
            ergas_list.append(ergas)
            sam_list.append(sam)

            with open('train.txt', 'a') as f:
                f.write('epoch:' + str(epoch) + '  ,  ' + 'rmse:' + str(rmse) + '  ,  ' + 'psnr:' + str(psnr) + '  ,  '
                        + 'ergas:' + str(ergas) + '  ,  ' + 'sam:' + str(
                    sam) + '\n')  # + str(ssim) + ','
            f.close()
    rmse_mean = np.mean(rmse_list)
    psnr_mean = np.mean(psnr_list)
    ergas_mean = np.mean(ergas_list)
    sam_mean = np.mean(sam_list)

    rmse_var = np.var(rmse_list)
    psnr_var = np.var(psnr_list)
    ergas_var = np.var(ergas_list)
    sam_var = np.var(sam_list)
    print('rmse_mean:{}          rmse_var:{} '.format(rmse_mean, rmse_var))
    print('psnr_mean:{}          psnr_var:{} '.format(psnr_mean, psnr_var))
    print('ergas_mean:{}         ergas_var:{} '.format(ergas_mean, ergas_var))
    print('sam_mean:{}           sam_var:{} '.format(sam_mean, sam_var))

    with open('train.txt', 'a') as f:
        f.write('time:' + str(datetime.now()) + '\n')

        f.write(
            'rmse_mean:' + str(rmse_mean) + '  ,  ' + 'rmse_var:' + str(rmse_var) + '\n' + 'psnr_mean:' + str(psnr_mean)
            + '  ,  ' + 'psnr_var:' + str(psnr_var) + '\n' + 'ergas_mean:' + str(ergas_mean) + '  ,  ' + 'ergas_var:' +
            str(ergas_var) + '\n' + 'sam_mean:' + str(sam_mean) + '  ,  ' + 'sam_var:' + str(sam_var) + '\n' )
    f.close()
    return ergas_mean, rmse_mean, psnr_mean, sam_mean  # ,ssim


if __name__ == '__main__':
    main()
