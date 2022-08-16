import copy
import argparse, sys, multiprocessing as mp
from time import time, sleep

import torch
import cv2
import numpy as np
import operator
import threading

from deep_sort.detection import Detection
from tools import generate_detections as gdet

from deep_sort.tracker import Tracker
from deep_sort import nn_matching

#For Pose Estimation
'''from src import util
from src.body import Body'''

model_filename = 'model_data/models/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename,batch_size=1)

#Definition of the parameters
max_cosine_distance = 0.2
nn_budget = None
nms_max_overlap = 0.4


from reid import REID

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color

def cv2_addBox(track_id, frame, x1, y1, x2, y2, line_thickness, text_thickness,text_scale):
    color = get_color(abs(track_id))
    cv2.rectangle(frame, (x1, y1), (x2, y2),color=color, thickness=line_thickness)
    cv2.putText(frame, str(track_id),(x1, y1+30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0,0,255),thickness=text_thickness)

def get_FrameLabels(frame):
    text_scale = max(1, frame.shape[1] / 1600.)
    text_thickness = 1 if text_scale > 1.1 else 1
    line_thickness = max(1, int(frame.shape[1] / 500.))
    return text_scale, text_thickness, line_thickness

class ObjectDetection:
    """
    Class implements Yolo5 model to make inferences and re-identify objects on live streams.
    """

    def __init__(self, feats_dict_shared, images_queue_shared, feat_dict_lock):
        """
        Initializes the class with shared dictionaries to communicate between stream threads and helper subprocess
        :param feats_dict_shared: A `multiprocessing.Manager().dict()` which is shared by this class's instances and `extract_feature` subrprocess
        This dictionary is used to exchange object features between the processes.
        :param images_queue_shared: A `multiprocessing.Queue()` which is shared by this class's instances and `extract_feature` subrprocess.
        This queue is used to exchange images between the processes.
        :param feat_dict_lock: A `multiprocessing.Lock()` which is shared by this class's instances and `extract_feature` subrprocess.
        This lock prevents race condition during `feats_dict_shared` I/O.
        """
        self.feats_dict_shared = feats_dict_shared
        self.images_queue_shared = images_queue_shared
        self.feat_dict_lock = feat_dict_lock

        #Stores the main ids along wiith their sub-ids
        self.final_fuse_id = dict()
        #Stores images of the detected objects under their ids for each streaming device
        self.images_by_id = dict()
        #Stores all the ids encountered during a session
        self.exist_ids = set()
        #Controls the threshold for matching features
        self.threshold = 300
        self.reid = REID()

        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric, max_age=100)

        self.model = self._load_model()
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        #Pose estimation initialization
        #self.body_estimation = Body('model_data/body_pose_model.pth')
        print("Using Device: ", self.device)

    def _load_model(self):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='.\crowdhuman_yolov5m.pt', force_reload=True)
        model.classes = [0]
        return model

    def _score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def _box_transform(self, box: list, shape: tuple):
        """
        :param box: Takes the normalized coordinates of the bounding box output by YOLOv5.
        :param shape: Takes the width and height of frame(frame.shape[1], frame.shape[0])
        Return box with [top_x, top_y, w, h]
        """
        x = int(box[0]*shape[0])
        y = int(box[1]*shape[1])
        w = int((box[2]-box[0]) * shape[0])
        h = int((box[3]-box[1]) * shape[1])

        if x < 0 :
            w = w + x
            x = 0
        if y < 0 :
            h = h + y
            y = 0

        return [x, y, w, h]

    def inference(self, video_id: int, frame: any, h: int, w: int, frame_cnt: int=0) -> int:
        """
        :param video_id: Takes an integer to differentiate between streams
        :param frame: OpeCV frame
        :param h: OpeCV frame height
        :param w: OpeCV frame width
        :param frame_cnt: Integer indicating the number of current frame
        Return incremented frame_cnt
        """
        results = self._score_frame(frame)
            
        boxs = [self._box_transform(cords, (frame.shape[1], frame.shape[0])) for cords in results[1]] #[minx, miny, w, h]

        features = encoder(frame, boxs) # n * 128
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)] # length = n
        text_scale, text_thickness, line_thickness = get_FrameLabels(frame)

        self.tracker.predict()
        self.tracker.update(detections)
        tmp_ids = []
        ids_per_frame = []
        id_prefix = str(video_id) + "_"

        track_cnt = dict()
        frame_cnt += 1
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
                    
            bbox = track.to_tlbr()
            area = (int(bbox[2]) - int(bbox[0])) * (int(bbox[3]) - int(bbox[1]))
            ids = str(id_prefix + str(track.track_id))

            if bbox[0] >= 0 and bbox[1] >= 0 and bbox[3] < h and bbox[2] < w:
                tmp_ids.append(ids)
                if ids not in self.images_by_id[video_id]:
                    track_cnt[ids] = [[frame_cnt, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), area]]
                    self.images_by_id[video_id][ids] = [frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]]
                else:
                    track_cnt[ids] = [[frame_cnt, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), area]]
                    self.images_by_id[video_id][ids].append(frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])])
                idx = int(ids[ids.find('_') + 1: :])
            if len(tmp_ids) > 0:
                ids_per_frame.append(set(tmp_ids))
            print("IDs per frame: ", ids_per_frame)
            sleep(1)
        
        for i in self.images_by_id[video_id]:
            if len(self.images_by_id[video_id][i]) > 100:
                #To reduce memory consumption
                del self.images_by_id[video_id][i][:20:]

            self.images_queue_shared.put([i, frame_cnt, self.images_by_id[video_id][i]])

        self.feat_dict_lock.acquire()
        local_feats_dict = {}
        for key, value in self.feats_dict_shared.items():
            local_feats_dict[key] = copy.deepcopy(value)
        self.feat_dict_lock.release()

        min_num_of_features = 5
        for f in ids_per_frame:
            if f:
                if len(self.exist_ids) == 0:
                    for i in f:
                        self.final_fuse_id[i] = [i]
                        self.exist_ids = self.exist_ids | f
                else:
                    new_ids = f - self.exist_ids
                    for nid in new_ids:
                        dis = []
                        print("Started collecting with NEW ids")
                        t = time()
                        if not nid in local_feats_dict.keys() or local_feats_dict.shape[0] < min_num_of_features:
                            self.exist_ids.add(nid)
                            if nid in local_feats_dict.keys():
                                print("Not enough feats: {}, ID: {}".format(local_feats_dict[nid].shape[0], nid))
                            else:
                                print("New ID to be extracted: {}".format(nid))
                            continue
                        else:
                            pass
                            print("finished collecting with NEW ids: ", time() - t)

                    unpickable = []
                    for i in f:
                        for key,item in self.final_fuse_id.items():
                            if i in item:
                                unpickable += self.final_fuse_id[key]

                    for left_out_id in f & (self.exist_ids - set(unpickable)):
                        dis = []
                        t = time()
                        if not left_out_id in local_feats_dict.keys() or local_feats_dict[left_out_id].shape[0] < min_num_of_features:
                            continue
                        for main_id in self.final_fuse_id.keys():
                            tmp = np.mean(self.reid.compute_distance(local_feats_dict[left_out_id], local_feats_dict[main_id]))
                            print('Left out {}, Main ID {}, tmp {}'.format(left_out_id, main_id, tmp))
                            dis.append([main_id, tmp])
                        print("Finished reiding with old ids: ", time() - t)
                        if dis:
                            dis.sort(key=operator.itemgetter(1))
                            print("Closest match found b/w: ", dis[0][0], left_out_id, dis[0][1])
                            for i in range(0, len(dis)):
                                if dis[i][1] < self.threshold:
                                    print("Creating subIDs: ", dis[i][0], left_out_id, dis[i][1])
                                    combined_id = dis[i][0]
                                    self.images_by_id[int(combined_id[0:combined_id.find('_'):])][combined_id] += self.images_by_id[int(left_out_id[0:left_out_id.find('_'):])][left_out_id]
                                    self.final_fuse_id[combined_id].append(left_out_id)
                                else:
                                    print("New ID added: ", left_out_id)
                                    self.final_fuse_id[left_out_id] = [left_out_id]
                                    break
                        else:
                            print("New ID added: ", left_out_id)
                            self.final_fuse_id[left_out_id] = [left_out_id]

        for idx in self.final_fuse_id:
                for i in self.final_fuse_id[idx]:
                    for current_ids in ids_per_frame:
                        for f in current_ids:
                            if str(i) == str(f) or str(idx) == str(f):
                                #Only drawing the bounding box and detecting pose when match between subID and mainID is found
                                text_scale, text_thickness, line_thickness = get_FrameLabels(frame)
                                _idx = int(idx[idx.find('_') + 1: :])
                                detection_track = track_cnt[f][0]
                                cv2_addBox(_idx, frame, detection_track[1], detection_track[2], detection_track[3], detection_track[4], line_thickness, text_thickness, text_scale)

        return frame_cnt

    def _reid_on_streams(self, device: int=0, url: str=""):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and displays the output frame with ids and poses.
        :param device: An integer indicating the device number/id
        :param url: Takes a string containing the url along with the port to receive the stream.
        Example: url = "192.168.256.23:8080"
        This will be embedded into an http endpoint: "http://192.168.256.23:8080/video"

        :return: None
        """
        if url == "0":
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture()
            cap.open("http://{}/video".format(url))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        assert cap.isOpened()

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.images_by_id[device] = {}
        frame_cnt = 0

        if device > 0:
            #Gives time to the first camera to gather features
            sleep(30)

        print("Starting inference on device ", device)
        while True:
            ret, frame = cap.read()
            assert ret
            
            #Reidentification inference running on frames of current device. Shares data with other device's inference thread for re-identification
            frame_cnt = self.inference(device, frame, h, w, frame_cnt)

            cv2.imshow('Device: {}'.format(device), frame)
 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()

    def __call__(self, p_urls: list=[]):
        """
        This function is called when class is executed, it runs the loop to read the video streams frame by frame,
        and displays the frames with reidentified ids and bounding boxes.
        :param p_urls:

        :return: None
        """
        threads = []
        for id, url in enumerate(p_urls):
            t = threading.Thread(target=self._reid_on_streams, args=(id, url,))
            threads.append(t)
            t.start()
        
        for thread in threads:
            thread.join()

def extract_features(feats, q, f_lock) -> None:
    '''
    Receives images from the threads of detected persons and extracts features and adds to shared dictionary
    '''
    from reid import REID
    reid = REID()
    print("Feature extraction subprocess has started")
    l_dict = dict()
    while True:
        t = time()
        if not q.empty():
            id, cnt, img = q.get()
            if id in l_dict.keys():
                if l_dict[id][0] < cnt:
                    l_dict[id] = [cnt, img]
                else:
                    continue
            else:
                l_dict[id] = [cnt, img]

            f = reid._features(l_dict[id][1])
            f_lock.acquire()
            feats[id] = f
            f_lock.release()
            print("Succesfully extracted features of images with ID: ", id)


import warnings
warnings.filterwarnings('ignore')

class ExtendAction(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest) or []
        items.extend(values)
        setattr(namespace, self.dest, items)

if __name__ == "__main__":
    #Instantiate the parser
    parser = argparse.ArgumentParser()
    parser.register('action', 'extend', ExtendAction)
    parser.add_argument('-u', '--urls', action="extend", nargs="+", type=str, help='add urls made of host and port')
    urls = parser.parse_args(args = sys.argv[1:]).urls

    #Using a queue, shared memory dictionary with lock between extraction subprocess and inference subprocess
    FeatsLock = mp.Lock()
    shared_feats_dict = mp.Manager().dict()
    shared_images_queue = mp.Queue()

    extract_p = mp.Process(target=extract_features, args=(shared_feats_dict, shared_images_queue, FeatsLock,))
    extract_p.start()
    
    try:
        detector = ObjectDetection(shared_feats_dict, shared_images_queue, FeatsLock)
        detector(p_urls=urls)
    except Exception as e:
        print("Error occured: ", e)
        raise
    finally:
        extract_p.terminate()
        extract_p.join()
        shared_images_queue.close()