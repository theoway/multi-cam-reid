import unittest

from reidentifier import ObjectDetection, extract_features
import multiprocessing as mp
import cv2

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
        device = 0

        extract_prcoess = mp.Process(target=extract_features, args=(feature_dict, queue, FeatsLock,))
        extract_prcoess.start()

        cap = cv2.VideoCapture('./tests/videos/Double1.mp4')
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frame_cnt = 0
        try:
            detector = ObjectDetection(feature_dict, queue, FeatsLock)
            detector.images_by_id[device] = {}

            while cap.isOpened() and frame_cnt <= 15:
                ret, frame = cap.read()
                assert ret

                frame_cnt = detector.inference(device, frame, h, w, frame_cnt)

            cap.release()

            print("IDs: ", detector.final_fuse_id)
        except Exception as e:
            print("Error: ", e)
            raise
        finally:
            extract_prcoess.terminate()
            queue.close()
            self.assertTrue(self._validate_id_dict(detector.final_fuse_id))

if __name__ == "__main__":
    unittest.main(warnings='ignore')