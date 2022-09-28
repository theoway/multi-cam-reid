import argparse, sys, multiprocessing as mp

from reidentifier import ObjectDetection, extract_features

import warnings
warnings.filterwarnings('ignore')

class ExtendAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest) or []
        items.extend(values)
        setattr(namespace, self.dest, items)

if __name__ == '__main__':
    #Instantiate the parser
    parser = argparse.ArgumentParser()
    parser.register('action', 'extend', ExtendAction)
    parser.add_argument('-p', '--pose', action="store_true", help='pass this to enable pose estimation on detected persons')
    parser.add_argument('-u', '--urls', action="extend", nargs="+", type=str, help='add urls made of host and port')
    urls = parser.parse_args(args = sys.argv[1:]).pose
    run_pose_estimation = parser.parse_args(args = sys.argv[1:]).urls

    print(urls, run_pose_estimation)
    #Using a queue, shared memory dictionary with lock between extraction subprocess and inference subprocess
    FeatsLock = mp.Lock()
    shared_feats_dict = mp.Manager().dict()
    shared_images_queue = mp.Queue()

    extract_p = mp.Process(target=extract_features, args=(shared_feats_dict, shared_images_queue, FeatsLock,))
    extract_p.start()
    
    try:
        detector = ObjectDetection(shared_feats_dict, shared_images_queue, FeatsLock)
        detector(p_urls=urls, estimate_pose=run_pose_estimation)
    except Exception as e:
        print("Error occured: ", e)
        raise
    finally:
        extract_p.terminate()
        extract_p.join()
        shared_images_queue.close()