import pickle
import os
import cv2
from matplotlib import pyplot as plt

label_class = ['Moving']

instance_color = ['yellow', 'lime', 'MediumVioletRed', 'Cyan', 'DarkOrange', 'Red', 'Navy', 'Indigo', 'RoyalBlue']


def takeLast(elem):
    return elem[-1]


def pkl_decode(opt):
    print('build finish, decode detection results', flush=True)
    with open(os.path.join(opt.inference_dir, 'tubes.pkl'), 'rb') as fid:
        pkl = pickle.load(fid)

    bbox_dict = {}
    tube_id = 0
    for label in pkl.keys():
        out = pkl[label]
        if len(out) > 0:
            out.sort(key=takeLast, reverse=True)
            for tube in out:
                tube_score = tube[1]
                if tube_score > opt.tube_vis_th:
                    label_name = label_class[label]
                    for frame in range(tube[0].shape[0]):
                        frame_score = tube[0][frame][5]
                        if frame_score > opt.frame_vis_th:
                            fid = tube[0][frame][0]
                            x1, y1, x2, y2 = tube[0][frame][1], tube[0][frame][2], tube[0][frame][3], tube[0][frame][4]
                            if fid not in bbox_dict:
                                bbox_dict[fid] = []
                                bbox_dict[fid].append([x1, y1, x2, y2, frame_score, label_name, tube_id])
                            else:
                                bbox_dict[fid].append([x1, y1, x2, y2, frame_score, label_name, tube_id])
                    tube_id += 1
    return bbox_dict


def vis_bbox(inference_dir, bbox_dict, instance_level=False):
    print('draw bboxes on each frame', flush=True)
    if not os.path.isdir(os.path.dirname('tmp')):
        os.system("mkdir -p tmp")
    
    im_list = os.listdir(inference_dir)
    im_list.sort()
    for pic in im_list:
        if pic.endswith('.jpg') or pic.endswith('.png'):
            
            im_data = cv2.imread(os.path.join(inference_dir, pic))
            
            fid = int(pic.split('.')[0])
            if fid not in bbox_dict:
                plt.savefig('tmp/' + pic)
                continue
            bbox_list = bbox_dict[fid]
            for bbox in bbox_list:
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                
                cv2.rectangle(im_data, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), thickness=3)    ## Red
                
                cv2.imwrite('tmp/' + pic, im_data)


def rgb2avi(inference_dir):
    print('convert .JPG to .AVI', flush=True)
    fps = 25
    
    height, width, _ = cv2.imread('tmp/000001.png').shape

    size = (width, height)

    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    video = cv2.VideoWriter(inference_dir + '/result_video.avi', fourcc, fps, size)

    filelist = os.listdir('tmp')
    filelist.sort()
    for pic in filelist:
        if pic.endswith('.jpg') or pic.endswith('.png'):
            video.write(cv2.imread(os.path.join('tmp', pic)))

    video.release()
    cv2.destroyAllWindows()
    
    
def rgb2gif(inference_dir):
    print('convert .JPG to .GIF', flush=True)
    import imageio
    GIF = []
    filelist = os.listdir('tmp')
    filelist.sort()
    for pic in filelist:
        if pic.endswith('.jpg') or pic.endswith('.png'):
            pic = cv2.imread(os.path.join('tmp', pic))[:, :, ::-1]
            GIF.append(pic)
    imageio.mimsave(inference_dir + '/result_video.gif', GIF, duration=0.04)  # the lower duration, the quicker gif speed
