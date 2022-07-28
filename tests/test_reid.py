import unittest

from reidentifier import ObjectDetection, extract_features
from reid import REID

import multiprocessing as mp

import cv2

images_by_id = dict()
#Controls the threshold for matching features
threshold = 380
reid = REID()
#All the ids ever tracked are stored here
exist_ids = set()
#Contains sub-ids mapped to their main ids
final_fuse_id = dict()
#Shared dictionary for features
feature_dict = mp.Manager().dict()
#Lock for shared feature dictionary
FeatsLock = mp.Lock()
#Queue to exchanges images between processes
queue = mp.Queue()

class Testing(unittest.TestCase):
    def _validate_id_dict(self, p_id_dict):
        for ids in p_id_dict.values():
            print(ids)
            if isinstance(ids, list) and len(ids) > 0:
                return True
        
        return False

    def test_run_single_video_thread(self):
        from deep_sort.tracker import Tracker
        from deep_sort import nn_matching

        device = 0

        #Definition of the parameters
        max_cosine_distance = 0.2
        nn_budget = None
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        tracker = Tracker(metric, max_age=100)

        global images_by_id
        images_by_id[device] = {}

        global threshold
        global exist_ids
        global final_fuse_id
        global reid
        global FeatsLock
        global queue
        global feature_dict

        extract_prcoess = mp.Process(target=extract_features, args=(feature_dict, queue, FeatsLock,))
        extract_prcoess.start()

        cap = cv2.VideoCapture('./tests/videos/Double1.mp4')
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frame_cnt = 0
        try:
            detector = ObjectDetection(0, feature_dict, queue)
            
            while cap.isOpened() and frame_cnt <= 15:
                ret, frame = cap.read()
                assert ret

                frame_cnt = detector.inference(device, frame, h, w, frame_cnt, images_by_id, threshold, exist_ids, 
                    final_fuse_id, reid, FeatsLock, tracker)

            cap.release()

            print("IDs: ", final_fuse_id.keys())
        except Exception as e:
            print("Error: ", e)
            raise
        finally:
            extract_prcoess.terminate()
            queue.close()
            self.assertTrue(self._validate_id_dict(final_fuse_id))

if __name__ == "__main__":
    unittest.main(warnings='ignore')