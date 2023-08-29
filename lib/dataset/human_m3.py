
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os.path as osp
import numpy as np
import json_tricks as json
import pickle
import logging
import os
import copy

from dataset.JointsDataset import JointsDataset
from utils.transforms import projectPoints, get_affine_transform, affine_transform, get_scale
import cv2
import tqdm
import torch
import time
import open3d as o3d
logger = logging.getLogger(__name__)

collect_joints_def = {
    'pelvis':0,
    'left_hip':1,
    'right_hip':2,
    'spine1':3,
    'left_knee':4,
    'right_knee':5,
    'spine2':6,
    'left_ankle':7,
    'right_ankle':8,
    'spine3':9,
    'left_foot':10,
    'right_foot':11,
    'neck':12,
    'left_collar':13,
    'right_collar':14,
    'head':15,
    'left_shoulder':16,
    'right_shoulder':17,
    'left_elbow':18,
    'right_elbow':19,
    'left_wrist':20,
    'right_wrist':21
    }

valid_joint_index = np.array([0,1,2,4,5,7,8,12,15,16,17,18,19,20,21])

valid_bone_link = [[0,1],[0,2],[1,3],[2,4],[3,5],[4,6],[0,7],[7,8],[7,9],[7,10],[9,11],[10,12],[11,13],[12,14]]

valid_joints_def = {
    'pelvis':0,
    'left_hip':1,
    'right_hip':2,
    'left_knee':3,
    'right_knee':4,
    'left_ankle':5,
    'right_ankle':6,
    'neck':7,
    'head':8,
    'left_shoulder':9,
    'right_shoulder':10,
    'left_elbow':11,
    'right_elbow':12,
    'left_wrist':13,
    'right_wrist':14
    }

COCO_KEYPOINTS = [
  'nose',      # 1
  'left_eye',    # 2
  'right_eye',    # 3
  'left_ear',    # 4
  'right_ear',    # 5
  'left_shoulder',  # 6
  'right_shoulder', # 7
  'left_elbow',   # 8
  'right_elbow',   # 9
  'left_wrist',   # 10
  'right_wrist',   # 11
  'left_hip',    # 12
  'right_hip',    # 13
  'left_knee',    # 14
  'right_knee',   # 15
  'left_ankle',   # 16
  'right_ankle',   # 17
]
COLLECT_TO_COCO_12 = [16,17,18,19,20,21,1,2,4,5,7,8]
COCO_TO_COCO_12 = [5,6,7,8,9,10,11,12,13,14,15,16]
# COCO_TO_COLLECT = [[11,12], 11, 12, 13, 14, 15, 16, [5,6], [3,4], 5, 6, 7, 8, 9, 10]
COCO_TO_COLLECT = [0, 11, 12, 13, 14, 15, 16, 0, 0, 5, 6, 7, 8, 9, 10]
SCENE_LIST = ['basketball1', 'basketball2', 'basketball3', 'crossdata', 'multiperson']

class Human_M3(JointsDataset):
    def __init__(self, cfg, image_set, is_train, transform=None, db_folder = 'db/collection/'):
        super().__init__(cfg, image_set, is_train, transform)
        self.num_joints = len(COCO_TO_COLLECT)
        # self.num_joints_unified = 12
        # self.cam_list = [(0, 3), (0, 6), (0, 12), (0, 13), (0, 23)]
        self.num_views = [4,4,3,4,4]
        self.frames = [2000, 2000, 5000, 2000, 1200]
        self.is_train = is_train
        self.split = 'train' if self.is_train else 'test'
        self.ori_image_size = [[2048, 1536], [2048, 1536], [2048, 1536], [1024, 768], [1024, 768]]
        self.cfg = cfg
        self.image_size = cfg.NETWORK.IMAGE_SIZE
        this_dir = os.path.dirname(__file__)
        self.dataset_root = cfg.DATASET.ROOT
        self.image_set = image_set
        self.dataset_name = "collection"
        # self.image_width = 2048
        # self.image_height = 1536
        self._interval = 1
        self.max_num_persons = cfg.MULTI_PERSON.MAX_PEOPLE_NUM
        self.root_id = 0
        # self.use_pred_confidence = cfg.TEST.USE_PRED_CONFIDENCE
        # self.nms_threshold = cfg.TEST.NMS_THRESHOLD
        self.sequence_list = SCENE_LIST
        if self.image_set == "train":
            self.valid_frames = [np.arange(0, nf * 0.9) for nf in self.frames]
        elif self.image_set == "validation":
            self.valid_frames = [np.arange(nf * 0.9, nf) for nf in self.frames]
        #self.pred_pose2d = self._get_pred_pose2d(os.path.join(self.dataset_root, "keypoints_{}_results.json".format(self.image_set)))
        self.pred_pose2d_path = cfg.DATASET.ROOT_POSE2D
        # self.gt_num = 0
        self.with_pcd = cfg.NETWORK.USE_PCD
        self.cameras = self._get_cam()
        # self.db = self._get_db()
        # self.db_size = len(self.db)
        if not os.path.exists(db_folder):
            os.makedirs(db_folder)
        db_file = os.path.join(db_folder, 'train.pkl') if is_train else os.path.join(db_folder, 'test.pkl')
        if osp.exists(db_file):
            info = pickle.load(open(db_file, 'rb'))
            self.db = info['db']
        else:
            self.db = self._get_db()
            info = {
                'db': self.db
            }
            pickle.dump(info, open(db_file, 'wb'))
        self.db_size = len(self.db)
        # import pdb; pdb.set_trace()

    def _get_pred_pose2d(self, fp):
        with open(fp, "r") as f:
            logging.info("=> load {}".format(fp))
            preds = json.load(f)

        if self.is_train:
            image_to_preds = defaultdict(dict)
            for pred in preds:
                # === GT bounding boxes are used to obtain 2D pose estimation for training data so that we have the identity of each detected 2D pose
                #     Identity is needed in order to provide supervision on depths
                image_to_preds[pred["image_name"]][pred["id"]] = np.array(pred["pred"]).reshape([-1, 3])
        else:
            image_to_preds = defaultdict(list)
            for pred in preds:
                image_to_preds[pred["image_name"]].append(np.array(pred["pred"]).reshape([-1, 3]))
        logging.info("=> {} estimated 2D poses from {} images loaded".format(len(preds), len(image_to_preds)))

        return image_to_preds

    def _get_cam(self):
        cameras = dict()
        for ind, seq in enumerate(self.sequence_list):
            cameras[seq] = []
            for i in range(self.num_views[ind]):        
                cam_file = os.path.join(self.dataset_root, self.split, seq, 'camera_calibration', 'camera_'+str(i)+'.json')
                with open(cam_file, "r") as f:
                    calib = json.load(f)
                    for key in calib:
                        calib[key] = np.array(calib[key])
                    calib['R'] = calib['extrinsic'][:3,:3]
                    calib['t'] = calib['extrinsic'][:3,[3]]
                    calib['K'] = calib['intrinsic'][:3,:3]
                    calib['distCoef'] = np.zeros([5])
                cameras[seq].append(calib)
        return cameras

    def _get_db(self):
        db = []
        self.gt_num = 0
        show = False
        for ind,seq in tqdm.tqdm(enumerate(self.sequence_list)):
            cameras = self.cameras[seq]
            pred_pose2d_dir = sorted(os.listdir(os.path.join(self.pred_pose2d_path, seq)))
            curr_anno = os.path.join(self.dataset_root, self.split, seq, 'pose_calib')
            image_path_list = \
             [sorted(glob.glob(os.path.join(self.dataset_root, self.split, seq, "images", 'camera_'+str(iks), '*.jpg'))) + \
             sorted(glob.glob(os.path.join(self.dataset_root, self.split, seq, "images", 'camera_'+str(iks), '*.jpeg'))) \
             for iks, v in enumerate(cameras)]
            pred_cam_json = []
            for iks in range(len(pred_pose2d_dir)):
                pred_cam_folder = os.path.join(self.pred_pose2d_path, seq, pred_pose2d_dir[iks])
                pred_cam_json.append(sorted(glob.glob(os.path.join(pred_cam_folder, '*.json'))))

            for i, frame in enumerate(self.valid_frames[ind]):
                if i % self._interval != 0:
                    continue
                frame = int(frame)
                anno_file = os.path.join(curr_anno, str(frame).zfill(4)+'.json')
                with open(anno_file, "r") as f:
                    bodies = json.load(f)
                if len(bodies) == 0:
                    continue
                all_info = []
                if self.with_pcd:
                    pcd_file = os.path.join(self.dataset_root, self.split, seq, 'pointcloud', str(int(frame)).zfill(6)+'.pcd')
                    if show:
                        pcd = o3d.io.read_point_cloud(pcd_file)
                        pcd = np.array(pcd.points)
                        import matplotlib
                        matplotlib.use('TkAgg')
                        import matplotlib.pyplot as plt

                        # import pdb; pdb.set_trace()
                        fig = plt.figure()
                        ax = fig.add_subplot(111, projection = '3d')
                        ax.scatter(pcd[:,0], pcd[:,1], pcd[:,2], marker = '*', s = 1)
                        for key in bodies:
                            poses = np.array(bodies[key])
                            ax.scatter(poses[:,0], poses[:,1], poses[:,2], marker = '^', s = 5)
                        # import pdb; pdb.set_trace()
                        plt.show()
                else:
                    pcd_file = None
                # import pdb; pdb.set_trace()
                for iks, v in enumerate(cameras):
                    # print(frame - int(self.valid_frames[ind][0]))
                    image_path = image_path_list[iks][frame - int(self.valid_frames[ind][0])]
                    our_cam = dict()
                    our_cam['R'] = v['R']
                    our_cam['T'] = -1 * np.linalg.inv(v['R']) @ v['t'] #-np.dot(v['R'].T, v['t'])  # the order to handle rotation and translation is reversed
                    our_cam['fx'] = np.array(v['K'][0, 0])
                    our_cam['fy'] = np.array(v['K'][1, 1])
                    our_cam['cx'] = np.array(v['K'][0, 2])
                    our_cam['cy'] = np.array(v['K'][1, 2])
                    our_cam['k'] = np.zeros([3, 1])
                    our_cam['p'] = np.zeros([2, 1])
                    all_poses_3d = []
                    all_poses_3d_vis = []
                    all_poses = []
                    all_poses_vis = []
                    all_poses_2d_pred = []

                    for key in bodies.keys():
                        body = bodies[key]
                        pose3d = np.array(body)#[valid_joint_index, :]
                        all_poses_3d.append(pose3d)  # [Nj, 3]
                        joints_vis = np.ones([pose3d.shape[0],3])

                        all_poses_3d_vis.append(joints_vis)  # [Nj]
                        pose2d = np.zeros((pose3d.shape[0], 2))
                        pose2d[:, :2] = projectPoints(
                                pose3d[:, 0:3].transpose(), v['K'], v['R'],
                                v['t'], v['distCoef']).transpose()[:, :2]
                        x_check = np.bitwise_and(pose2d[:, 0] >= 0,
                                                    pose2d[:, 0] <= self.ori_image_size[ind][0] - 1)
                        y_check = np.bitwise_and(pose2d[:, 1] >= 0,
                                                    pose2d[:, 1] <= self.ori_image_size[ind][1] - 1)
                        check = np.bitwise_and(x_check, y_check)
                        joints_2d_vis = np.copy(joints_vis)
                        joints_2d_vis[np.logical_not(check)] = 0
                        all_poses.append(pose2d)
                        all_poses_vis.append(joints_2d_vis[:,:2])

                    with open(pred_cam_json[iks][frame], "r") as f:
                        pred_pose2d = json.load(f)
                        for pp2d in pred_pose2d:
                            pose_ori = np.array(pp2d['keypoints']).reshape([17,3])
                            kp2d = pose_ori[COCO_TO_COLLECT, :]#[5:,:] #[12,3]
                            if pose_ori[11,2] > 0.3 and pose_ori[12,2] > 0.3:
                                kp2d[0] = (pose_ori[11] + pose_ori[12]) / 2
                            if pose_ori[5,2] > 0.3 and pose_ori[6,2] > 0.3:
                                kp2d[7] = (pose_ori[5] + pose_ori[6]) / 2
                            if pose_ori[3,2] > 0.3 and pose_ori[4,2] > 0.3:
                                kp2d[8] = (pose_ori[3] + pose_ori[4]) / 2
                            all_poses_2d_pred.append(kp2d)

                    all_info.append({
                        "image_path": image_path,
                        "joints_2d": np.array(all_poses),
                        "joints_2d_vis": np.array(all_poses_vis),
                        "joints_3d": np.array(all_poses_3d),  # [Np, Nj, 3]
                        "joints_3d_vis": np.array(all_poses_3d_vis),  # [Np, Nj]
                        "pred_pose2d": np.array(all_poses_2d_pred),  # [Np_hrnet, Nj_unified, 2+1]
                        "camera": our_cam,
                    })
                self.gt_num += 1
                db.append({'info':all_info, 'pcd':pcd_file})

        logger.info("=> {} data from {} views loaded".format(len(db), self.num_views))
        return db

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        all_info = self.db[idx]['info']
        # print(all_info[0]['image_path'])
        # if all_info[0]['image_path'].split('/')[-1].split('.')[0] != '1656236109803480946':
        #     return None
        input, target, weight, target_3d, meta, input_heatmap = [], [], [], [], [], []
        for info in all_info:
            i, t, w, t3, m, ih = self._get_single_view_item(info)
            if i is None:
                continue
            input.append(i)
            target.append(t)
            weight.append(w)
            target_3d.append(t3)
            meta.append(m)
            input_heatmap.append(ih)
        if len(input) == 3: #self.is_train and 
            indx = np.random.randint(3)
            input.append(input[indx])
            target.append(target[indx])
            weight.append(weight[indx])
            target_3d.append(target_3d[indx])
            meta.append(meta[indx])
            input_heatmap.append(input_heatmap[indx])
        ret_dict ={}
        
        if self.cfg.NETWORK.USE_RGB:
            ret_dict['target_2d'] = target
            ret_dict['weight_2d'] = weight
        ret_dict['input'] = input

        ret_dict['target_3d'] = target_3d
        ret_dict['meta'] = meta
        if self.cfg.NETWORK.INPUT_HM:
            ret_dict['input_heatmap'] = input_heatmap

        if self.cfg.NETWORK.USE_PCD:
            pcd_file = self.db[idx]['pcd']
            pcd = o3d.io.read_point_cloud(pcd_file)
            pcd = torch.tensor(np.array(pcd.points))
            pcd_voxel = torch.zeros(self.cfg.MULTI_PERSON.INITIAL_CUBE_SIZE)
            pcd_index = pcd / torch.tensor(self.space_size).unsqueeze(0) * torch.tensor(self.initial_cube_size).unsqueeze(0)
            pcd_index = pcd_index.long()
            pcd_valid = (pcd_index[:,0] > 0) & (pcd_index[:,1] > 0) & \
                (pcd_index[:,2] > 0) & (pcd_index[:,0] < self.initial_cube_size[0]) & (pcd_index[:,1] < self.initial_cube_size[1]) & (pcd_index[:,2] < self.initial_cube_size[2])
            pcd_index = pcd_index[pcd_valid,:]
            pcd_voxel[pcd_index[:,0], pcd_index[:,1], pcd_index[:,2]] = 1
            ret_dict['pcd_voxel'] = pcd_voxel.unsqueeze(0)
            ret_dict['pcd_file'] = pcd_file
            # print(meta[0]['joints_3d'])
            # print(all_info[0]['image_path'])
            # from medpy.io import save
            # save(pcd_voxel, 'save_mha/'+all_info[0]['image_path'].split('/')[-1].split('.')[0]+'.mha')
        # import pdb; pdb.set_trace()
        # for key in ret_dict:
        #     if key == 'meta':
        #         continue
        #     print(key)
        #     if isinstance(ret_dict[key], list):
        #         for rd in ret_dict[key]:
        #             print(rd.shape)
        #     elif isinstance(ret_dict[key], dict):
        #         for rd in ret_dict[key]:
        #             print(ret_dict[key][rd].shape)
        #     else:
        #         print(ret_dict[key].shape)
        return ret_dict

    def _get_single_view_item(self, info):
        image_file = info['image_path']
        data_numpy = cv2.imread(
            image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        # scene = info['scene']
        # if data_numpy is None:
        #     # logger.error('=> fail to read {}'.format(image_file))
        #     # raise ValueError('Fail to read {}'.format(image_file))
        #     return None, None, None, None, None, None

        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        joints = info['joints_2d']
        joints_vis = info['joints_2d_vis']
        joints_3d = info['joints_3d']
        joints_3d_vis = info['joints_3d_vis']

        nposes = len(joints)
        assert nposes <= self.maximum_person, 'too many persons'

        height, width, _ = data_numpy.shape
        c = np.array([width / 2.0, height / 2.0])
        s = get_scale((width, height), self.cfg.NETWORK.IMAGE_SIZE)
        r = 0

        trans = get_affine_transform(c, s, r, self.cfg.NETWORK.IMAGE_SIZE)
        input = cv2.warpAffine(
            data_numpy,
            trans, (int(self.cfg.NETWORK.IMAGE_SIZE[0]), int(self.cfg.NETWORK.IMAGE_SIZE[1])),
            flags=cv2.INTER_LINEAR)

        if self.transform:
            input = self.transform(input)
        
        if self.cfg.NETWORK.INPUT_HM:
            joints = joints @ trans[:,:2].swapaxes(0,1)
            if 'pred_pose2d' in info and info['pred_pose2d'] is not None:

                pred_pose2d = info['pred_pose2d']
                if pred_pose2d.shape[0] > 0:
                    pred_pose2d[...,:2] = pred_pose2d[...,:2] @ trans[:,:2].swapaxes(0,1) #尺度变化

                input_heatmap = self.generate_input_heatmap(pred_pose2d)
                input_heatmap = torch.from_numpy(input_heatmap)
            else:
                input_heatmap = torch.zeros(self.cfg.NETWORK.NUM_JOINTS, self.heatmap_size[1], self.heatmap_size[0])

            target_heatmap, target_weight = self.generate_target_heatmap(
                joints, joints_vis)

            target_heatmap = torch.from_numpy(target_heatmap)
            target_weight = torch.from_numpy(target_weight)

            # make joints and joints_vis having same shape
            joints_u = np.zeros((self.maximum_person, self.num_joints, 2))
            joints_vis_u = np.zeros((self.maximum_person, self.num_joints, 2))
            for i in range(nposes):
                joints_u[i] = joints[i]
                joints_vis_u[i] = joints_vis[i]
        else:
            input_heatmap = np.zeros([1])
            target_heatmap = np.zeros([1])
            target_weight = np.zeros([1])
        joints_3d_u = np.zeros((self.maximum_person, self.num_joints, 3))
        joints_3d_vis_u = np.zeros((self.maximum_person, self.num_joints, 3))

        for i in range(nposes):
            joints_3d_u[i] = joints_3d[i][:, 0:3]
            joints_3d_vis_u[i] = joints_3d_vis[i][:, 0:3]

        target_3d = self.generate_3d_target(joints_3d, 0.2)
        target_3d = torch.from_numpy(target_3d)

        if isinstance(self.root_id, int):
            roots_3d = joints_3d_u[:, self.root_id]
        elif isinstance(self.root_id, list):
            roots_3d = np.mean([joints_3d_u[:, j] for j in self.root_id], axis=0)
        meta = {
            'image': image_file,
            'num_person': nposes,
            'joints_3d': joints_3d_u,
            'joints_3d_vis': joints_3d_vis_u,
            'roots_3d': roots_3d,
            'center': c,
            'scale': s,
            'rotation': r,
            'camera': info['camera'],
            'image_size': [width, height], #与相机参数绑定的图片尺寸
        }
        if self.cfg.NETWORK.INPUT_HM:
            meta.update({
            'joints': joints_u,
            'joints_vis': joints_vis_u,})
        return input, target_heatmap, target_weight, target_3d, meta, input_heatmap

    def evaluate(self, preds):
        eval_list = []
        gt_num = self.db_size
        # assert len(preds) == gt_num, 'number mismatch'

        total_gt = 0
        for i in range(gt_num):
            # index = self.num_views * i
            db_rec = copy.deepcopy(self.db[i]['info'][0])
            joints_3d = db_rec['joints_3d']
            joints_3d_vis = db_rec['joints_3d_vis']

            if len(joints_3d) == 0:
                continue

            pred = preds[i].copy()
            pred = pred[pred[:, 0, 3] >= 0]
            # import pdb; pdb.set_trace()
            for pose in pred:
                mpjpes = []
                for (gt, gt_vis) in zip(joints_3d, joints_3d_vis):
                    vis = gt_vis[:, 0] > 0
                    mpjpe = np.mean(np.sqrt(np.sum((pose[vis, 0:3] - gt[vis]) ** 2, axis=-1)))
                    mpjpes.append(mpjpe)
                min_gt = np.argmin(mpjpes)
                gt_this = joints_3d[min_gt]
                min_mpjpe = np.min(mpjpes)
                score = pose[0, 4]
                eval_list.append({
                    "mpjpe": float(min_mpjpe),
                    "score": float(score),
                    "gt_id": int(total_gt + min_gt),
                    "gt": gt_this,
                    "db": [{'image_path': info['image_path'], 'camera':info['camera']} for info in self.db[i]['info']],
                    "pcd_file": self.db[i]['pcd']
                })
            total_gt += len(joints_3d)
        # np.save('data_statistic/preds.npy', np.array(preds))
        # np.save('data_statistic/eval_list.npy', np.array(eval_list))
        mpjpe_threshold = np.arange(0.025, 0.155, 0.025)
        aps = []
        recs = []
        for t in mpjpe_threshold:
            ap, rec = self._eval_list_to_ap(eval_list, total_gt, t)
            aps.append(ap)
            recs.append(rec)
        return aps, recs, self._eval_list_to_mpjpe(eval_list, 0.5), self._eval_list_to_recall(eval_list, total_gt, 0.5)

    @staticmethod
    def _eval_list_to_ap(eval_list, total_gt, threshold):
        eval_list.sort(key=lambda k: k["score"], reverse=True)
        total_num = len(eval_list)

        tp = np.zeros(total_num)
        fp = np.zeros(total_num)
        gt_det = []
        for i, item in enumerate(eval_list):
            if item["mpjpe"] < threshold and item["gt_id"] not in gt_det:
                tp[i] = 1
                gt_det.append(item["gt_id"])
            else:
                fp[i] = 1
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        recall = tp / (total_gt + 1e-5)
        precise = tp / (tp + fp + 1e-5)
        for n in range(total_num - 2, -1, -1):
            precise[n] = max(precise[n], precise[n + 1])

        precise = np.concatenate(([0], precise, [0]))
        recall = np.concatenate(([0], recall, [1]))
        index = np.where(recall[1:] != recall[:-1])[0]
        ap = np.sum((recall[index + 1] - recall[index]) * precise[index + 1])

        return ap, recall[-2]

    @staticmethod
    def _eval_list_to_mpjpe(eval_list, threshold=500):
        eval_list.sort(key=lambda k: k["score"], reverse=True)
        gt_det = []

        mpjpes = []
        for i, item in enumerate(eval_list):
            if item["mpjpe"] < threshold and item["gt_id"] not in gt_det:
                mpjpes.append(item["mpjpe"])
                gt_det.append(item["gt_id"])

        return np.mean(mpjpes) if len(mpjpes) > 0 else np.inf

    @staticmethod
    def _eval_list_to_recall(eval_list, total_gt, threshold=500):
        gt_ids = [e["gt_id"] for e in eval_list if e["mpjpe"] < threshold]
        # import pdb; pdb.set_trace()
        return len(np.unique(gt_ids)) / total_gt

if __name__ == '__main__':
    collect = Collection()
    for data in collect:
        import pdb; pdb.set_trace()