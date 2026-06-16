# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np
import sys
import time
pysot_path = os.path.abspath('/file01/zhengwei/SiameseRPN++/pysot')
sys.path.insert(0, pysot_path)

toolkit_path = os.path.abspath('/file01/zhengwei/SiameseRPN++/toolkit')
sys.path.insert(0, toolkit_path)

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str
from get_boxes_on_img import  draw_boxes_on_image_opencv, get_ious, transform_box_to_search_region, WRITE_PSR,draw_bbox_on_template, save_adv_box,get_fig_2
from get_heatmap import get_heatmap
#from get_PSR import  calculate_psr

import torch
from torchsummary import summary
from thop import profile
from thop import clever_format
import torchvision.models as models


parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--dataset', default='OTB', type=str,
        help='datasets')
parser.add_argument('--config', default='/file01/zhengwei/SiameseRPN++/experiments/siamrpn_r50_1234_dwxcorr/config.yaml', type=str,
        help='config file')
parser.add_argument('--snapshot', default='/file01/zhengwei/SiameseRPN++/experiments/siamrpn_r50_1234_dwxcorr/model.pth', type=str,
        help='snapshot of models to eval')
parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--vis', action='store_true',
        help='whether visualzie result')
args = parser.parse_args()

torch.set_num_threads(1)

def main():
    # load config
    cfg.merge_from_file(args.config)

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    #dataset_root = os.path.join(cur_dir, '../testing_dataset', args.dataset)
    dataset_root = '/file01/dataset/dataset_OTB'
    #dataset_root = '/file01/dataset/got_10k_data/test'

    # create model
    torch.cuda.set_device(1)
    model = ModelBuilder()
    model = model.cuda()
    '''
    # 使用thop分析模型的运算量和参数量
    input = {}
    input['template'] = torch.randn(1, 3, 128, 128)
    input['search'] = torch.randn(1, 3, 256, 256)
    input['label_cls'] = torch.randn(1, 25, 25)
    input['label_loc'] = torch.randn(1, 4, 5, 25, 25)
    input['label_loc_weight'] = torch.randn(1, 5, 25, 25)


    flops, params = profile(model, inputs=(input,))

    # 将结果转换为更易于阅读的格式
    #MACs, params = clever_format([MACs, params], '%.3f')
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
    #print(f"运算量：{MACs}, 参数量：{params}")
    '''

    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()


    # build tracker
    tracker = build_tracker(model)

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    model_name = args.snapshot.split('/')[-1].split('.')[0]
    total_lost = 0
    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:
        # restart tracking
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            frame_counter = 0
            lost_number = 0
            toc = 0
            pred_bboxes = []
            for idx, (img, gt_bbox) in enumerate(video):
                if len(gt_bbox) == 4:
                    gt_bbox = [gt_bbox[0], gt_bbox[1],
                       gt_bbox[0], gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]]
                tic = cv2.getTickCount()
                if idx == frame_counter:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    tracker.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    pred_bboxes.append(1)
                elif idx > frame_counter:
                    outputs = tracker.track(img)
                    pred_bbox = outputs['bbox']
                    if cfg.MASK.MASK:
                        pred_bbox = outputs['polygon']
                    overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
                    if overlap > 0:
                        # not lost
                        pred_bboxes.append(pred_bbox)
                    else:
                        # lost object
                        pred_bboxes.append(2)
                        frame_counter = idx + 5 # skip 5 frames
                        lost_number += 1
                else:
                    pred_bboxes.append(0)
                toc += cv2.getTickCount() - tic
                if idx == 0:
                    cv2.destroyAllWindows()
                if args.vis and idx > frame_counter:
                    cv2.polylines(img, [np.array(gt_bbox, np.int).reshape((-1, 1, 2))],
                            True, (0, 255, 0), 3)
                    if cfg.MASK.MASK:
                        cv2.polylines(img, [np.array(pred_bbox, np.int).reshape((-1, 1, 2))],
                                True, (0, 255, 255), 3)
                    else:
                        bbox = list(map(int, pred_bbox))
                        cv2.rectangle(img, (bbox[0], bbox[1]),
                                      (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(img, str(lost_number), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
            toc /= cv2.getTickFrequency()
            # save results
            video_path = os.path.join('results', args.dataset, model_name,
                    'baseline', video.name)
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    if isinstance(x, int):
                        f.write("{:d}\n".format(x))
                    else:
                        f.write(','.join([vot_float2str("%.4f", i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
                    v_idx+1, video.name, toc, idx / toc, lost_number))
            total_lost += lost_number
        print("{:s} total lost: {:d}".format(model_name, total_lost))
    else:
        # OPE tracking
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            toc = 0

            pred_bboxes = []
            pred_bboxes_adv = []
            iteration_iou_adv = []
            scores = []
            track_times = []
            bboxes = []
            '''
            for idx, (img, gt_bbox) in enumerate(video):
                if idx == 0:
                    with torch.no_grad():
                        first_img = img
                        cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                        gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                        break
            if video.name == 'Car1':
                sec_nd = round(len(video.img_names) / 4)
                thi_rd = round(len(video.img_names) / 2)
                four_th = round(len(video.img_names) / 4 * 3)
                sec_nd_bboxes = []
                thi_rd_bboxes = []
                four_th_bboxes = []
                bboxes.append(gt_bbox_)
                sampled_boxes = sample_gaussian_boxes(
                    gt_box=gt_bbox_,
                    num_samples=32,
                    iou_threshold=0.7
                )

                for time in range(20):
                    sampled_boxes = process_perturbed_boxes(sampled_boxes, gt_bbox_, num_boxes=32,
                                                            iou_threshold=0.7)
                    for i in range(32):
                        sampled_box = sampled_boxes[i]
                        bbox_on_template, template = tracker.init(first_img, sampled_box, sampled_box)
                        # tracker.init(first_img, gt_bbox_, sampled_box)
                        outputs = tracker.track(first_img)
                        pred_bbox_adv = outputs['bbox']
                        if 'VOT2018-LT' == args.dataset:
                            pred_bboxes_adv.append([1])
                        else:
                            pred_bboxes_adv.append(pred_bbox_adv)
                    if time < 19:
                        ious = np.array([calculate_iou(box, gt_bbox_) for box in pred_bboxes_adv])
                        min = np.min(ious)
                        min_PSR_index = np.argmin(ious)
                        bbox_on_template, template = tracker.init(first_img,
                                                                  sampled_boxes[min_PSR_index],
                                                                  sampled_boxes[min_PSR_index])
                        outputs = tracker.track(first_img)
                        iteration_iou_adv.append(min)
                        sampled_boxes, attact_weights = resample_boxes(sampled_boxes, pred_bboxes_adv, gt_bbox_,
                                                                       num_samples=32,
                                                                       iou_threshold=0.7)
                        pred_bboxes_adv = []
                        if str(sampled_boxes.__class__.__name__) == 'Tensor':
                            sampled_boxes = sampled_boxes.numpy()

                    else:
                        ious = np.array([calculate_iou(box, gt_bbox_) for box in pred_bboxes_adv])
                        unique_ious, unique_indices = np.unique(ious, return_index=True)
                        #min_indices = unique_indices[np.argsort(unique_ious)[:5]]
                        #min_indices = np.random.choice(unique_indices, size=5, replace=False)
                        min_indices = np.argsort(ious)[0]
                        sampled_boxes = sampled_boxes[min_indices]
                #for box in sampled_boxes:
                    #bboxes.append(box)
                bboxes.append(sampled_boxes)
                last_pred_bbox = []
                i = 0
                for box in bboxes:
                    pred_bboxes = []
                    for idx, (img, gt_bbox) in enumerate(video):
                        tic = cv2.getTickCount()
                        if idx == 0:
                            with torch.no_grad():
                                first_img = img
                                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                                gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                                _, _ = tracker.init(img, box, box)
                        else:
                            scores.append(None)
                            outputs = tracker.track(img)
                            pred_bbox = outputs['bbox']
                            pred_bboxes.append(pred_bbox)
                            if idx == sec_nd:
                                sec_nd_img = img
                                sec_nd_bboxes.append(pred_bbox)
                            if idx == thi_rd:
                                thi_rd_img = img
                                thi_rd_bboxes.append(pred_bbox)
                            if idx == four_th:
                                four_th_img = img
                                four_th_bboxes.append(pred_bbox)
                            if idx == len(video.img_names)-1:
                                last_img = img
                                last_pred_bbox.append(pred_bbox)
                            scores.append(outputs['best_score'])
                    if i == 1:
                        i = i + 1
                    model_path = os.path.join('/file01/zhengwei/fig2/', video.name + '_' + str(i) + '.txt')
                    if not os.path.isdir('/file01/zhengwei/fig2/'):
                        os.makedirs('/file01/zhengwei/fig2/')

                    with open(model_path, 'w') as f:
                        for x in pred_bboxes:
                            f.write(','.join([str(i) for i in x])+'\n')

                    i = i + 1

                get_fig_2(bboxes, first_img,video_name=video.name,file = '1')
                get_fig_2(sec_nd_bboxes, sec_nd_img,video_name=video.name,file = '2')
                get_fig_2(thi_rd_bboxes,thi_rd_img,video_name=video.name,file = '3')
                get_fig_2(four_th_bboxes,four_th_img,video_name=video.name,file = '4')
                get_fig_2(last_pred_bbox,last_img,video_name=video.name, file = '5')



            '''
            for idx, (img, gt_bbox) in enumerate(video):
                tic = cv2.getTickCount()
                if idx == 0:
                    with torch.no_grad():
                        first_img = img
                        cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                        gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                        
                        tracker.init(img, gt_bbox_, gt_bbox_)
                        pred_bbox = gt_bbox_
                        scores.append(None)
                        if 'VOT2018-LT' == args.dataset:
                            pred_bboxes.append([1])
                        else:
                            pred_bboxes.append(pred_bbox)
                            #pred_bboxes.append(sampled_boxes[min_iou_index])
                        
                else:
                    if idx == 1:

                        PSR_STORE = []
                        NUM_SAMPLES = 32
                        IOU_THRESHOLD = 0.6
                        EPOCH = 20
                        sampled_boxes = sample_gaussian_boxes(
                            gt_box=gt_bbox_,
                            num_samples=NUM_SAMPLES,
                            iou_threshold=IOU_THRESHOLD
                        )
                        min_iou_index = 0
                        min_iou = 1
                        global_min = 1
                        bbox_on_template, template = tracker.init(first_img, gt_bbox_, gt_bbox_)
                        #outputs = tracker.track(first_img)
                        #ori_x_feature = outputs['x_feature']
                        for time_epoch in range(EPOCH):
                            sampled_boxes = process_perturbed_boxes(sampled_boxes, gt_bbox_, num_boxes=NUM_SAMPLES, iou_threshold=IOU_THRESHOLD)
                            start_time = time.perf_counter()
                            for i in range(NUM_SAMPLES):
                                sampled_box = sampled_boxes[i]
                                _, _ = tracker.init(first_img, sampled_box, sampled_box)
                                #tracker.init(first_img, gt_bbox_, sampled_box)
                                outputs = tracker.track(first_img)
                                pred_bbox_adv = outputs['bbox']
                                if 'VOT2018-LT' == args.dataset:
                                    pred_bboxes_adv.append([1])
                                else:
                                    pred_bboxes_adv.append(pred_bbox_adv)
                            end_time = time.perf_counter()
                            print(f"运行耗时: {end_time - start_time:.6f} 秒")
                            if time_epoch == 51:
                                end_time = time.perf_counter()
                                print(f"运行耗时: {end_time - start_time:.6f} 秒")

                            if time_epoch < EPOCH-1:
                                ious = np.array([calculate_iou(box, gt_bbox_) for box in pred_bboxes_adv])
                                min = np.min(ious)
                                '''
                                min_PSR_index = np.argmin(ious)
                                bbox_on_template, template = tracker.init(first_img,
                                  sampled_boxes[min_PSR_index], sampled_boxes[min_PSR_index])
                                outputs = tracker.track(first_img)
                                adv_x_feature = outputs['x_feature']
                                
                                distance = calculate_l2_distance(adv_x_feature,ori_x_feature)
                                distance_dir = '/file01/zhengwei/SiameseRPN++/tools/ALL_DISTANCE'
                                os.makedirs(distance_dir, exist_ok=True)
                                distance_path = os.path.join(distance_dir,str(time)+'.txt')
                                with open(distance_path, 'a') as f:
                                    f.write(f"{distance}\n")
                                '''
                                iteration_iou_adv.append(min)
                                sampled_boxes, attact_weights = resample_boxes(sampled_boxes, pred_bboxes_adv, gt_bbox_, num_samples=NUM_SAMPLES, iou_threshold=IOU_THRESHOLD)
                                pred_bboxes_adv = []
                                if str(sampled_boxes.__class__.__name__) == 'Tensor':
                                    sampled_boxes = sampled_boxes.numpy()
                            else:
                                index = -1
                                for j in pred_bboxes_adv:
                                    index = index + 1
                                    iou = calculate_iou(j, gt_bbox_)
                                    if iou < min_iou:
                                        min_iou = iou
                                        min_iou_index = index

                                final_iou = calculate_iou(sampled_boxes[min_iou_index], gt_bbox_)
                                #iteration_iou_adv.append(min)
                                bbox_on_template, template = tracker.init(first_img,
                                                                          sampled_boxes[min_iou_index], sampled_boxes[min_iou_index])
                                outputs = tracker.track(first_img)
                                adv_x_feature = outputs['x_feature']
                                '''
                                distance = calculate_l2_distance(adv_x_feature, ori_x_feature)
                                distance_dir = '/file01/zhengwei/SiameseRPN++/tools/ALL_DISTANCE'
                                os.makedirs(distance_dir, exist_ok=True)
                                distance_path = os.path.join(distance_dir,str(time)+'.txt')
                                with open(distance_path, 'a') as f:
                                    f.write(f"{distance}\n")
                                '''
                                score_map = outputs['score_map']
                                '''
                                PSR = calculate_psr(score_map)
                                PSR = [PSR]
                                WRITE_PSR(PSR, ty = str(time))
                                '''
                                iteration_iou_adv.append(min_iou)
                        

                        #get_ious(iteration_iou_adv,"adv_pred_iteration_iou.txt")
                        #final_iou = [final_iou]
                        #get_ious(final_iou,"final_adv_iou.txt")
                        

                            
                        #gt_bbox_on_template, gt_template = tracker.init(first_img, gt_bbox_, gt_bbox_)
                        #draw_bbox_on_template(gt_template, gt_bbox_on_template, Type='original2', video_name=video.name)
                        #adv_bbox_on_template, adv_template = tracker.init(first_img, sampled_boxes[min_iou_index], sampled_boxes[min_iou_index])
                        #draw_bbox_on_template(adv_template,adv_bbox_on_template,Type = 'adv2',video_name = video.name)
                        #outputs = tracker.track(first_img)
                        #score_map = outputs['score_map']
                        #PSR = calculate_psr(score_map)
                        #PSR = [PSR]
                        #WRITE_PSR(PSR, ty = 'adv2')
                            
                        #tracker.init(first_img, gt_bbox_, gt_bbox_)


                        
                        #x, y, w, h = gt_bbox_
                        # 定义高斯分布的标准差，使其与gt_box的尺寸相关
                        # 这使得采样对于不同大小的目标都具有适应性
                        '''
                        std_devs = np.array([
                            0.1 * w,  # x_center 的标准差
                            0.1 * h,  # y_center 的标准差
                            0.1 * w,  # width 的标准差
                            0.1 * h  # height 的标准差
                        ])
                        '''
                        #noise = np.random.normal(0, std_devs, 4)

                        # 生成候选采样框
                        #random_adv_box = gt_bbox_ + noise
                        #tracker.init(first_img, random_adv_box, random_adv_box)
                        
                        #tracker.init(first_img, gt_bbox_, gt_bbox_)
                        


                        #adv_box = [sampled_boxes[min_iou_index]]
                        #save_adv_box(adv_box,video.name)



                        #outputs = tracker.track(first_img)
                        #score_map = outputs['score_map']
                        #PSR = calculate_psr(score_map)
                        #PSR = [PSR]
                        #WRITE_PSR(PSR, ty='ori')

                        
                        scores.append(None)

                        if 'VOT2018-LT' == args.dataset:
                            pred_bboxes.append([1])
                        else:
                            #pred_bboxes.append(pred_bbox)
                            #pred_bboxes.append(worst_bbox)
                            pred_bboxes.append(gt_bbox_)
                            #pred_bboxes.append(sampled_boxes[min_iou_index])
                            #pred_bboxes.append(random_adv_box)

                        '''
                        original_adv_bbox_path = "/file01/zhengwei/SiameseRPN++/tools/All_ADV_Box"
                        seq_name = video.name + ".txt"
                        original_adv_bbox_path = os.path.join(original_adv_bbox_path, seq_name)

                        with open(original_adv_bbox_path, 'r') as file:
                            for line_num, line in enumerate(file, 1):
                                if line_num == 1:  # 只处理第一行
                                    line = line.strip()
                                    sampled_box = list(map(float, line.split()))
                                    break  # 读取完第一行就退出循环
                            pred_bboxes.append(sampled_box)
                        '''
                        



                        #_, _ = tracker.init(first_img, sampled_boxes[min_iou_index], sampled_boxes[min_iou_index])

                        #scores.append(None)



                    outputs = tracker.track(img)
                    #prediction_box = outputs['bbox']
                    #score_map = outputs['score_map']

                    '''
                    if video.name == "Woman":
                        output_bboxes = []
                        if idx == 10 or idx == 11:
                            search_region = outputs['search_region']
                            bbox_on_search_region = outputs['bbox_on_search_region']
                            gt_bbox_on_search_region = outputs['gt_bbox_on_search_region']
                            gt_bbox_on_search_region = np.array(gt_bbox_on_search_region)
                            output_bboxes.append(bbox_on_search_region)
                            output_bboxes.append(gt_bbox_on_search_region)
                            search_region_np = search_region.detach().cpu()
                            search_region_np = search_region_np.squeeze(0)
                            search_region_np = search_region_np.permute(1, 2, 0)
                            search_region_np = search_region_np.numpy()
                            draw_boxes_on_image_opencv(search_region_np, output_bboxes, idx, Type='ori',
                                                           video_name=video.name)
                            get_heatmap(score_map, search_region, idx, Type='ori', video_name=video.name)
                    '''
                    
                    pred_bbox = outputs['bbox']
                    pred_bboxes.append(pred_bbox)
                    scores.append(outputs['best_score'])
                    

                toc += cv2.getTickCount() - tic
                track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
                if idx == 0:
                    cv2.destroyAllWindows()
                if args.vis and idx > 0:
                    gt_bbox = list(map(int, gt_bbox))
                    pred_bbox = list(map(int, pred_bbox))
                    cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                  (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
                    cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                  (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)

            toc /= cv2.getTickFrequency()


            # save results
            if 'VOT2018-LT' == args.dataset:
                video_path = os.path.join('results', args.dataset, model_name,
                        'longterm', video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path,
                        '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
                result_path = os.path.join(video_path,
                        '{}_001_confidence.value'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in scores:
                        f.write('\n') if x is None else f.write("{:.6f}\n".format(x))
                result_path = os.path.join(video_path,
                        '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            elif 'GOT-10k' == args.dataset:
                video_path = os.path.join('results', args.dataset,'random', model_name, video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
                result_path = os.path.join(video_path,
                        '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            else:
                model_path = os.path.join('LaSOT_results/random', args.dataset, model_name)
                if not os.path.isdir(model_path):
                    os.makedirs(model_path)
                result_path = os.path.join(model_path, '{}.txt'.format(video.name))
                '''
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
                '''
            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                v_idx+1, video.name, toc, idx / toc))


def calculate_iou(boxA, boxB):
    """
       计算两个边界框的 IoU
       输入格式: (x, y, w, h) 其中 x,y 是左上角坐标
    """
    # 提取坐标和尺寸
    x1, y1, w1, h1 = boxA
    x2, y2, w2, h2 = boxB

    # 转换为 (x_min, y_min, x_max, y_max) 格式
    x1_min, y1_min = x1, y1
    x1_max, y1_max = x1 + w1, y1 + h1

    x2_min, y2_min = x2, y2
    x2_max, y2_max = x2 + w2, y2 + h2

    # 计算交集区域的坐标
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    # 检查是否有交集
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    # 计算交集面积
    inter_width = inter_x_max - inter_x_min
    inter_height = inter_y_max - inter_y_min
    inter_area = inter_width * inter_height

    # 计算两个边界框的面积
    area1 = w1 * h1
    area2 = w2 * h2

    # 计算并集面积
    union_area = area1 + area2 - inter_area

    # 计算 IoU
    iou = inter_area / union_area

    return iou


def sample_gaussian_boxes(gt_box, num_samples=32, iou_threshold=0.6):
    """
    在给定的真实框(gt_box)中心附近进行高斯采样，
    确保所有采样框与gt_box的IoU不低于阈值。

    参数:
    - gt_box (list or np.array): 真实框 [x_center, y_center, width, height]
    - num_samples (int): 需要采样的数量
    - iou_threshold (float): IoU的最低阈值

    返回:
    - np.array: 形状为 (num_samples, 4) 的采样框数组
    """
    x, y, w, h = gt_box
    gt_box = np.array(gt_box)

    # 定义高斯分布的标准差，使其与gt_box的尺寸相关
    # 这使得采样对于不同大小的目标都具有适应性

    std_devs = np.array([
        0.1 * w,  # x_center 的标准差
        0.1 * h,  # y_center 的标准差
        0.1 * w,  # width 的标准差
        0.1 * h  # height 的标准差
    ])
    '''
    std_devs = np.array([
        0.08 * w,  # x_center 的标准差
        0.08 * h,  # y_center 的标准差
        0.05 * w,  # width 的标准差
        0.05 * h  # height 的标准差
    ])
    '''
    valid_samples = []
    max_attempts = 5000  # 设置最大尝试次数以避免死循环
    attempts = 0

    #print(f"开始采样，目标数量: {num_samples}, IoU阈值: {iou_threshold}")

    while len(valid_samples) < num_samples and attempts < max_attempts:
        # 生成高斯扰动
        noise = np.random.normal(0, std_devs, 4)

        # 生成候选采样框
        candidate_box = gt_box + noise

        # 确保宽高为正数
        if candidate_box[2] < 0 or candidate_box[2] == 0:
                candidate_box[2] = max(gt_box[2], candidate_box[2])
        if candidate_box[3] < 0 or candidate_box[3] == 0:
                candidate_box[3] = max(gt_box[3], candidate_box[3])

        # 检验IoU
        if calculate_iou(candidate_box, gt_box) >= iou_threshold:
            valid_samples.append(candidate_box)

        attempts += 1

    # 如果尝试多次后仍不足，可以用gt_box自身填充
    if len(valid_samples) < num_samples:
        print(f"警告: 在{max_attempts}次尝试后只生成了{len(valid_samples)}个样本。")
        print("用真实框填充剩余位置。")
        remaining = num_samples - len(valid_samples)
        for _ in range(remaining):
            valid_samples.append(gt_box)

    #print(f"采样完成，共生成 {len(valid_samples)} 个采样框。")
    return np.array(valid_samples)


def process_perturbed_boxes(perturbed_boxes, true_box, iou_threshold=0.6, num_boxes=32, gaussian_std=0.1):
    """
    处理扰动框以满足IoU要求。

    Args:
        perturbed_boxes (np.ndarray): 初始的32个扰动框，形状 (32, 4)。
        true_box (np.ndarray): 真实框，形状 (4,)。
        iou_threshold (float): IoU阈值。
        num_boxes (int): 期望的扰动框数量。
        gaussian_std (float): 高斯扰动的标准差。

    Returns:
        np.ndarray: 满足IoU要求的32个扰动框。
    """
    if str(perturbed_boxes.__class__.__name__) == 'Tensor':
        perturbed_boxes = perturbed_boxes.detach().cpu().numpy()
    if str(true_box.__class__.__name__) == 'Tensor':
        true_box = true_box.detach().cpu().numpy()
    current_perturbed_boxes = perturbed_boxes.copy()

    while True:
        # 1. 检测32个扰动框是否均满足与真实框的iou不低于0.6
        ious = np.array([calculate_iou(box, true_box) for box in current_perturbed_boxes])

        if np.all(ious >= iou_threshold):
            #print("所有扰动框都满足IoU阈值。")
            return current_perturbed_boxes

        #print("存在IoU低于阈值的框，进入第二步。")

        # 2. 去除iou低于0.6的框，并从剩下的框中按照iou生成权重
        valid_indices = np.where(ious >= iou_threshold)[0]

        if len(valid_indices) == 0:
            #print("警告: 没有框满足IoU阈值。请检查输入或调整阈值。重新初始化扰动框。")
            # 重新初始化一个随机的扰动框集合作为起点，或者根据实际需求处理
            current_perturbed_boxes = true_box + np.random.randn(num_boxes, 4) * gaussian_std * 10  # 示例：更大扰动
            continue  # 重新开始循环

        valid_boxes = current_perturbed_boxes[valid_indices]
        valid_ious = ious[valid_indices]
        valid_ious = 1 - valid_ious
        # 根据IoU生成权重，IoU越高权重越low
        # 可以使用softmax或直接归一化
        weights = valid_ious / np.sum(valid_ious)
        # 确保权重和为1，防止浮点误差
        weights = weights / np.sum(weights)

        #print(f"筛选后剩余 {len(valid_boxes)} 个框满足IoU阈值。")

        # 3. 按照权重重采样32个扰动框，并附加随机的高斯扰动
        # 计算需要重采样的框的数量，即32个总数减去当前已满足条件的框数
        num_to_resample = num_boxes - len(valid_boxes)
        # 仅当还有空位需要填充时才进行重采样
        if num_to_resample > 0:
            resampled_indices = np.random.choice(len(valid_boxes), size=num_to_resample, p=weights, replace=True)
            # 将重采样的框和已满足条件的框合并
            new_perturbed_boxes = np.concatenate((valid_boxes, valid_boxes[resampled_indices]))
        else:
            # 如果已满足条件的框数量大于等于32个，则直接使用前32个
            new_perturbed_boxes = valid_boxes[:num_boxes]


        #resampled_indices = np.random.choice(len(valid_boxes), size=num_boxes, p=weights, replace=True)
        #new_perturbed_boxes = valid_boxes[resampled_indices].copy()

        # 附加随机的高斯扰动
        # 注意：这里假设框的坐标是浮点数。根据实际情况可能需要调整扰动的大小。
        #gaussian_noise = np.random.randn(num_boxes, 4) * gaussian_std
        #new_perturbed_boxes += gaussian_noise

        # 确保框的坐标仍然是有效的 (例如，x1 < x2, y1 < y2)
        # 这只是一个简单的处理，可能需要更复杂的逻辑来保证框的有效性
        #new_perturbed_boxes[:, 0] = np.minimum(new_perturbed_boxes[:, 0], new_perturbed_boxes[:, 2] - 1)
        #new_perturbed_boxes[:, 1] = np.minimum(new_perturbed_boxes[:, 1], new_perturbed_boxes[:, 3] - 1)

        current_perturbed_boxes = new_perturbed_boxes
        #print("完成重采样和高斯扰动，进入下一轮IoU检测。")

def resample_boxes(perturbed_boxes, predicted_boxes, ground_truth_box, num_samples=32, std_dev=0.01, iou_threshold = 0.6):
    """
    根据预测框与真实框的IoU，对扰动框进行重采样和高斯扰动。

    Args:
        perturbed_boxes (torch.Tensor): 初始的32个扰动框，形状为 [32, 4]。
        predicted_boxes (torch.Tensor): 32个预测框，形状为 [32, 4]。
        ground_truth_box (torch.Tensor): 单个真实框，形状为 [1, 4]。
        num_samples (int): 重采样后的样本数量，默认为32。
        std_dev (float): 高斯扰动的标准差，默认为0.1。

    Returns:
        torch.Tensor: 32个新的扰动框，形状为 [32, 4]。
    """
    weights = []
    # 1. 计算预测框与真实框的IoU
    for i in predicted_boxes:
        iou = calculate_iou(i, ground_truth_box)
        weight = 1-iou
        weights.append(weight)

    # 2. 根据1-IoU计算权重
    # 将权重归一化，使其和为1
    weights = np.array(weights)
    weights = weights / sum(weights)

    # 3. 按照权重从扰动框中进行重采样
    # torch.multinomial 是根据权重进行随机采样的理想选择
    weights = torch.from_numpy(weights)
    resampled_indices = torch.multinomial(weights, num_samples, replacement=True)
    resampled_boxes = perturbed_boxes[resampled_indices]
    resampled_boxes = torch.from_numpy(resampled_boxes)
    # 4. 对重采样后的框附加高斯扰动
    # 随机生成一个与框形状相同的高斯噪声
    noise = torch.randn_like(resampled_boxes) * std_dev
    # 将噪声添加到框的坐标上
    new_perturbed_boxes = resampled_boxes + noise
    new_perturbed_boxes = process_perturbed_boxes(new_perturbed_boxes, ground_truth_box, num_boxes=num_samples,iou_threshold=iou_threshold)
    #new_perturbed_boxes = resampled_boxes

    return new_perturbed_boxes, weights

def get_index_of_bboxes(bboxes,gt_bbox):
    index = -1
    min_iou = 1
    min_iou_index = 0
    for j in bboxes:
        index = index + 1
        iou = calculate_iou(j, gt_bbox)
        if iou < min_iou:
            min_iou = iou
            min_iou_index = index
    return  min_iou_index


def calculate_l2_distance(list1, list2):
    """
    计算两个列表中对应张量的L2范数距离

    参数:
    list1, list2: 包含3个张量的列表，形状分别为[1,512,31,31], [1,1024,31,31], [1,2048,31,31]

    返回:
    标量距离值
    """
    total_distance = 0.0

    for tensor1, tensor2 in zip(list1, list2):
        # 确保张量形状一致
        assert tensor1.shape == tensor2.shape, f"张量形状不匹配: {tensor1.shape} vs {tensor2.shape}"

        # 计算L2距离（欧几里得距离）
        distance = torch.norm(tensor1 - tensor2, p=2)

        # 将距离平方（可选，取决于你的需求）
        # 如果是求平均L2距离，可以不加平方
        total_distance += distance.item() ** 2

    # 计算平均距离
    avg_distance = total_distance / len(list1)

    # 如果你想得到欧几里得距离而不是平方距离
    final_distance = np.sqrt(avg_distance)

    return final_distance

if __name__ == '__main__':
    main()
