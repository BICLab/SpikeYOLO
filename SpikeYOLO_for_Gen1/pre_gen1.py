from prophesee.src.io.psee_loader import PSEELoader
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

import torch
import yaml
from tqdm import tqdm
from numpy.lib.recfunctions import structured_to_unstructured



def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]


def g1_img2labelpaths(img_paths):
    return ['/'.join(x.split('/')[:-1])+'/label'+str(x.split('/')[-1][3:])  for x in img_paths]



class LoadImagesAndLabels(Dataset):
    def __init__(self,path,outpath, sample_size,T,image_shape, mode="train"):
        self.mode = mode   
        self.outpath = outpath     
        self.sample_size =sample_size#defualt is 250000,需要后续去数据集核对
        self.quantization_size = [sample_size//T,1,1]#[50 000 ,1, 1]
        self.h, self.w = image_shape
        self.quantized_w = self.w // self.quantization_size[1]#304 quantization_size=1
        self.quantized_h = self.h // self.quantization_size[2]#240
        self.T=T
        #temp空
        labels=[]
        # self.labels = list(labels)#
            # data_dir = os.path.join(path, mode)
        save_file=' '
        build_path = path+'/'+ self.mode
        self.samples = self.build_dataset(build_path, save_file)
        #torch.save(self.samples, save_file)
        print(f"Done! File saved as {save_file}")
        self.shapes=np.zeros((len(self.samples),2))
        self.shapes[:,0]=image_shape[1]
        self.shapes[:,1]=image_shape[0]
        for i in range(len(self.samples)):
            labels.append(self.samples[i][1])
        self.labels=labels


            
    def __getitem__(self, index):
        img, target = self.samples[index]
        # sample=img.reshape([5,3,240,304])
        image=np.zeros((5,320,320,3))
        # sample = img.permute(0,3,1,2)      #[5,3,240,304];
        #需要对sample进行大小处理
        for i in range(img.shape[0]):
            # for j in range(sample.shape[1]):
                image[i]=cv2.resize(img[i],(320,320))

        image=image.reshape([5,3,320,320])#做了尺寸变换 这里是否好，是否有方法改进 304 *240 的大小是否适合YOLO

        labels_out = torch.zeros((target.shape[0], 6))
        labels_out[:, 1:] = torch.from_numpy(target)

        return torch.from_numpy(image), labels_out
    
    def __len__(self):
        return len(self.samples)#7501
        
    def build_dataset(self, path, save_file):
        # Remove duplicates (.npy and .dat)删去重复元素
        outpath= self.outpath
        task= self.mode
        files = [os.path.join(path, time_seq_name[:-9]) for time_seq_name in os.listdir(path)#路径下所有文件
                        if time_seq_name[-3:] == 'npy']
                        #读取对应集合文件名

        print('Building the Dataset')
        pbar = tqdm(total=len(files), unit='File', unit_scale=True)
        samples = []
        if os.path.exists(outpath):
            pass
        else:
            os.mkdir(outpath)
        savetxt = outpath+'/'+task+'.txt'
        svtxt = open(savetxt,'w')
        for file_name in files:
            p = 0
            #print(f"Processing {file_name}...")
            events_file = file_name + '_td.dat'
            video = PSEELoader(events_file)#video读取

            boxes_file = file_name + '_bbox.npy'
            boxes = np.load(boxes_file)
            # Rename 'ts' in 't' if needed (Prophesee GEN1)
            boxes.dtype.names = [dtype if dtype != "ts" else "t" for dtype in boxes.dtype.names]#npy文件中的name

            boxes_per_ts = np.split(boxes, np.unique(boxes['t'], return_index=True)[1][1:])#抽离出对应的box；记录为list格式
            #按照时间选定，独立划分时间
            #这里需要考虑下
            #this
            flname=file_name.split('/')[-1]
            if os.path.exists(outpath+'/'+task):
                pass 
            else:
                os.mkdir(outpath+'/'+task)
            for b in  boxes_per_ts:
                event_data,imglabels=self.create_sample(video,b)  #event_data:[T,H,W,C]
                svimgname = outpath+'/'+task+'/img_'+ flname + str(p)+'.npy'
                # svlbname  = outpath+'/'+task+'/label_'+ flname + str(p)+'.npy'
                svlbname = outpath + '/' + task + '/img_' + flname + str(p) + '.txt'
                np.save(svimgname,event_data)

                w = event_data.shape[1]
                h = event_data.shape[2]
                # np.save(svlbname,imglabels)
                scale = np.array([[1,0,0,0,0],[0,1/w,0,0,0],[0,0,1/h,0,0],[0,0,0,1/w,0],[0,0,0,0,1/h]],dtype = float)

                imglabels_txt =imglabels @ scale
                if (np.array(imglabels_txt) <= 1).all()== False:
                    print("error!") #判断归一化后是否有大于1的元素，保证归一化的维度正确

                np.savetxt(svlbname, imglabels_txt)
                if event_data is not None:
                    with open(savetxt,'a')as f:
                        f.write((str(svimgname)+'\n'))
                p+=1
            pbar.update(1)
        pbar.close()
        torch.save(samples, save_file)
        print(f"Done! File saved as {save_file}")
        return samples #空的
        
    def create_sample(self, video, boxes):

        ts = boxes['t'][0]
        video.seek_time(ts-self.sample_size)
        events=[]
        for i in range(self.T):
            events.append(video.load_delta_t(self.sample_size//self.T))

        targets,labels = self.create_targets(boxes)
        # self.labels=list(labels)
        #self.labes.extend(labels)
        if targets['boxes'].shape[0] == 0:
            print(f"No boxes at {ts}")
            return None
        elif len(events) == 0:
            print(f"No events at {ts}")
            return None
        else:
            return (self.create_data(events), labels)

    def create_targets(self, boxes):#解析包围盒和标签；只需要更改这个函数！！！！合并box和label(为什么会有负的？)
        torch_boxes = torch.from_numpy(structured_to_unstructured(boxes[['x', 'y', 'w', 'h']], dtype=np.float32))#抽出对应的4维

        # keep only last instance of every object per target
        _,unique_indices = np.unique(np.flip(boxes['track_id']), return_index=True) # keep last unique objects
        unique_indices = np.flip(-(unique_indices+1))
        torch_boxes = torch_boxes[[*unique_indices]]

        # torch_boxes[:, 2:] += torch_boxes[:, :2] # implicit conversion to xyxy
        torch_boxes[:, 0::2].clamp_(min=0, max=self.w) #第0和第2个元素
        torch_boxes[:, 1::2].clamp_(min=0, max=self.h)

        torch_boxes[:, 0] = torch_boxes[:, 0] + torch_boxes[:, 2] / 2
        torch_boxes[:, 1] = torch_boxes[:, 1] + torch_boxes[:, 3] / 2

        torch_boxes[:, 0::2].clamp_(min=0, max=self.w) #第0和第2个元素
        torch_boxes[:, 1::2].clamp_(min=0, max=self.h)

        # valid idx = width and height of GT bbox aren't 0
        valid_idx = (torch_boxes[:,2]!= 0) & (torch_boxes[:,3]!= 0)
        # valid_idx = (torch_boxes[:,2]-torch_boxes[:,0] != 0) & (torch_boxes[:,3]-torch_boxes[:,1] != 0)

        torch_boxes = torch_boxes[valid_idx, :]

        torch_labels = torch.from_numpy(boxes['class_id']).to(torch.long)
        torch_labels = torch_labels[[*unique_indices]]
        torch_labels = torch_labels[valid_idx]

        labels = structured_to_unstructured(boxes[['class_id','x', 'y', 'w', 'h']], dtype=np.float32)

        labels[:,1:]=torch_boxes.numpy()
        labels[:,0] =torch_labels.numpy()
        # #转换为center x,center y（已经在box那里转化过了）
        # labels[:,1]=labels[:,1]+labels[:,3]/2
        # labels[:,2]=labels[:,2]+labels[:,4]/2


        # labels[:,1]=labels[:,1]/304
        # labels[:,3]=labels[:,3]/304
        # labels[:,2]=labels[:,2]/240
        # labels[:,4]=labels[:,4]/240

        return {'boxes': torch_boxes, 'labels': torch_labels}, labels
    
    # def create_data(self, events,img=None):# 原版，输入shape是hw，后面改成wh 。新版中直接输入shape也改为wh
    #     height=240
    #     width=304
    #     #压缩到一张图片上；
    #     if img is None:
    #         img = 127 * np.ones((self.T,height, width, 3), dtype=np.uint8)#这是cv2维度顺序【h,w,c】
    #     else:
    #     # if an array was already allocated just paint it grey
    #         img[...] = 127
    #     if len(events):
    #         for i in range(self.T):
    #             if len(events[i]):
    #                 assert events[i]['x'].max() < width, "out of bound events: x = {}, w = {}".format(events[i]['x'].max(), width)
    #                 assert events[i]['y'].max() < height, "out of bound events: y = {}, h = {}".format(events[i]['y'].max(), height)
    #             img[i,events[i]['y'], events[i]['x'], :] = 255 * events[i]['p'][:, None]
    #     return img#numpy.ndarray[T,H,W，C]#255 -255 每个格子

    def create_data(self, events,img=None):
        width=304
        height=240
        #压缩到一张图片上；
        if img is None:
            img = 127 * np.ones((self.T,width,height, 3), dtype=np.uint8)#这是cv2维度顺序【h,w,c】
        else:
        # if an array was already allocated just paint it grey
            img[...] = 127
        if len(events):
            for i in range(self.T):
                if len(events[i]):
                    assert events[i]['x'].max() < width, "out of bound events: x = {}, w = {}".format(events[i]['x'].max(), width)
                    assert events[i]['y'].max() < height, "out of bound events: y = {}, h = {}".format(events[i]['y'].max(), height)
                img[i,events[i]['x'], events[i]['y'], :] = 255 * events[i]['p'][:, None]  #img默认为灰度127，有正事件则为255，无正事件则为0
        return img#numpy.ndarray[T,W,H，C]#255 -255 每个格子
    @staticmethod
    def collate_fn(batch):#定义数据拼接方式
        img,label=zip(*batch)
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()这个地方更换了label的第一维的值！！！
        return torch.stack(img, 0), torch.cat(label, 0)
        # return samples, targets#[B,T,C,H.W];list
# Ancillary functions ------------


#这份代码用于生成gen1的预处理数据，主要逻辑是根据ATIS的数据，考虑每次给出标签前2.5s数据，以0.5s进行分割形成5帧
#每帧数据则以最新事件进行替代，当没有发生事件输出为127，发生负事件为0，发生正事件为255，3通道是同一图案，因此您可以考虑将
#188行的通道3修正为1，后续可以考虑在模型的训练中修正通道数
#最终生成的npy数组为 5*240*304*3，label则以coco方式给出，，类别,左上角x，y坐标，长宽
#当然您也可以根据我们的代码进行你的数据预处理

#ATIS数据放在path路径下面，然后记得ATIS文件夹下的路径包含train,val,test,每个文件夹内部以 name_bbox.npy,name_td.dat组成
#输出的np文件在outpath内，将会生成对应的train,val,test3个子文件夹，随后内部生成文件名对应的txt文件，子文件夹内部以img_name.npy,label_name.npy组成

sample_size = 250000
image_shape = (240,304)
T           = 4
path = '/lxh/lxh_data/gen1_before' #use your path
outpath = '/lxh/lxh_data/gen1_coco_T4'

dataset_train = LoadImagesAndLabels(path,outpath, sample_size,T,image_shape, 'train')
# dataset_valid = LoadImagesAndLabels(path,outpath,sample_size,T,image_shape, 'val')
dataset_test  = LoadImagesAndLabels(path,outpath, sample_size,T,image_shape, 'test')


