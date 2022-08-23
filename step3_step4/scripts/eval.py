import numpy as np
import torch
from tqdm import tqdm
from utils.loss import CrossEntropy2d
from models.utils import load_model
import torch.nn.functional as F
import torch.cuda.amp as amp
from utils.utils import reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu


def val(args, model, dataloader):
    print('start val!')
    # label_info = get_label_info(csv_path)
    with torch.no_grad():
        model.eval()
        precision_record = []
        hist = np.zeros((args.num_classes, args.num_classes))
        for index, batch in tqdm(enumerate(dataloader), disable=False):
            data, label, _, name = batch
            label = label.type(torch.LongTensor)
            data = data.cuda()
            label = label.long().cuda()
            # label = torch.squeeze(label).long().cuda()

            # get RGB predict image
            predict = model(data).squeeze()
            predict = reverse_one_hot(predict)
            predict = np.array(predict.cpu())

            # get RGB label image
            label = label.squeeze()
            if args.loss == 'dice':
                label = reverse_one_hot(label)
            label = np.array(label.cpu())

            # compute per pixel accuracy
            precision = compute_global_accuracy(predict, label)
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)

            # there is no need to transform the one-hot array to visual RGB array
            # predict = colour_code_segmentation(np.array(predict), label_info)
            # label = colour_code_segmentation(np.array(label), label_info)
            precision_record.append(precision)
        
        precision = np.mean(precision_record)
        miou_list = per_class_iu(hist)
        miou = np.mean(miou_list)
        print('precision per pixel for test: %.3f' % precision)
        print('mIoU for validation: %.3f' % miou)
        print(f'mIoU per class: {miou_list}')

        return precision, miou


def validate(args, model, val_loader, metrics, visualizer, val_iter):
    model.eval()
    metrics.reset()
    criterion_seg = CrossEntropy2d(ignore_label=args.ignore_index)
    scaler = amp.GradScaler()

    results = {}
    val_loss = 0.0
    with torch.no_grad():
        for index, batch in tqdm(enumerate(val_loader), disable=False):
            image, label, _, name = batch
            with amp.autocast():
                pred_high = model(image.to(args.gpu_ids[0], dtype=torch.float32))

            interp = torch.nn.Upsample(size=(label.shape[1], label.shape[2]), mode='bilinear', align_corners=True).to(args.gpu_ids[0])
            output = interp(pred_high)

            loss_seg = criterion_seg(output, label.to(args.gpu_ids[0], dtype=torch.long))
            val_loss += loss_seg.item()

            for img, lbl, out, n in zip(interp(image), label, output, name):
                visualizer.display_current_results([(img, lbl, out)], val_iter, 'Val', n)

            with amp.autocast():
                _, output = output.max(dim=1)
            output = output.cpu().numpy()
            label = label.cpu().numpy()
            metrics.update(label, output)

        visualizer.info(f'Validation loss at iter {val_iter}: {val_loss/len(val_loader)}')
        visualizer.add_scalar('Validation_Loss', val_loss/len(val_loader), val_iter)
        score = metrics.get_results()
        # visualizer.add_figure("Val_Confusion_Matrix_Recall", score['Confusion Matrix'], step=val_iter)
        # visualizer.add_figure("Val_Confusion_Matrix_Precision", score["Confusion Matrix Pred"], step=val_iter)
        # results["Val_IoU"] = score['Class IoU']
        visualizer.add_results(results)
        visualizer.add_scalar('Validation_mIoU', score['Mean IoU'], val_iter)
        visualizer.info(metrics.to_str_print(score))
        print(score)
    metrics.reset()


def test(args, model, test_loader, metrics, visualizer):

    # Resume model and set to eval
    model, _, _, test_iter = load_model(args, model)
    model.eval()

    # Reset Metric and define variables for logging
    metrics.reset()
    results = {}

    # Start testing
    with torch.no_grad():
        for index, batch in tqdm(enumerate(test_loader), disable=False):
            images, labels, _, names = batch
            _, pred_high = model(images.cuda())
            interp = torch.nn.Upsample(size=(labels.shape[1], labels.shape[2]), mode='bilinear', align_corners=True)
            outputs = interp(pred_high)

            for img, lbl, out in zip(interp(images), labels, outputs):
                visualizer.display_current_results([(img, lbl, out)], test_iter, 'Test')

            _, outputs = outputs.max(dim=1)
            outputs = outputs.cpu().numpy()
            labels = labels.cpu().numpy()

            metrics.update(labels, outputs)

        score = metrics.get_results()
        visualizer.add_figure("Test_Confusion_Matrix_Recall", score['Confusion Matrix'], step=test_iter)
        visualizer.add_figure("Test_Confusion_Matrix_Precision", score["Confusion Matrix Pred"], step=test_iter)
        results["Test_IoU"] = score['Class IoU']
        visualizer.add_results(results)
        visualizer.add_scalar('Test_mIoU', score['Mean IoU'], test_iter)
        visualizer.info(f'Test')
        visualizer.info(metrics.to_str_print(score))

    metrics.reset()

