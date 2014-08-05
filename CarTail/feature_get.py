import numpy as np
import common
import cv2
import os
from collections import namedtuple

FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH    = 6
flann_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
MIN_MATCH_COUNT = 10

PlanarTarget = namedtuple('PlaneTarget', 'image, rect, keypoints, descrs, data')
TrackedTarget = namedtuple('TrackedTarget', 'target, p0, p1, H, quad')


#videowriter = cv2.VideoWriter('123.avi',cv2.cv.CV_FOURCC('I', '4', '2', '0'),10,(640,480))


class Mainfunc:
    def __init__(self,data_input):
        self.paused = False
        #self.cap = cv2.VideoCapture(data_input)
        self.cap = cv2.VideoCapture('133.avi')
        self.frame = None
        self.detector = cv2.ORB(1000)
        self.matcher = cv2.FlannBasedMatcher(flann_params, {})
        self.taregets = []
        cv2.namedWindow('src')
        self.rect_selector = common.RectSelector('src',self.on_rect_selector)
    def on_rect_selector(self,rect):
        #self.taregets = []
        #self.matcher.clear()
        self.add_target(self.frame, rect)
        print rect
    def add_target(self,img,rect):
        x0,y0,x1,y1 = rect
        keypoints,descs_ = self.detect_features(img)
        points =[]
        descs= []
        for kp,desc in zip(keypoints,descs_):
            x,y = kp.pt
            if x0 <= x <= x1 and y0 <= y <= y1:
                points.append(kp)
                descs.append(desc)
        descs = np.uint8(descs)
        #self.matcher.clear()
        self.matcher.add([descs])
        target = PlanarTarget(image = img,rect=rect,keypoints=points,descrs=descs,data=None)
        self.taregets.append(target)         
    def detect_features(self,frame):
        keypoints,descrs = self.detector.detectAndCompute(frame,None)
        if descrs is None:
            descrs = []
        return keypoints,descrs
    def track(self,frame):
        self.frame_points,self.frame_descrs = self.detect_features(frame)
        if len(self.frame_points)< MIN_MATCH_COUNT:
            return []
        matches = self.matcher.knnMatch(self.frame_descrs, k = 2)
        matches = [m[0] for m in matches if len(m) == 2 and m[0].distance<m[1].distance*0.75]
        if len(matches) < MIN_MATCH_COUNT:
            return []
        matches_by_id = [[] for _ in xrange(len(self.taregets))]
        for m in matches:
            matches_by_id[m.imgIdx].append(m)
        tracked = []
        
        for imgIdx, matches in enumerate(matches_by_id):
            if len(matches)< MIN_MATCH_COUNT:
                continue
            target = self.taregets[imgIdx]
            p0 = [target.keypoints[m.trainIdx].pt for m in matches]
            p1 = [self.frame_points[m.queryIdx].pt for m in matches]
            p0, p1 = np.float32((p0, p1))
            H, status = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)
            status = status.ravel() != 0
            if status.sum() < MIN_MATCH_COUNT:
                continue
            p0, p1 = p0[status], p1[status]

            x0, y0, x1, y1 = target.rect
            quad = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            quad = cv2.perspectiveTransform(quad.reshape(1, -1, 2), H).reshape(-1, 2)

            track = TrackedTarget(target=target, p0=p0, p1=p1, H=H, quad=quad)
            tracked.append(track)
        tracked.sort(key = lambda t: len(t.p0), reverse=True)
        return tracked
        
    def run(self):
        num = 0
        while True:     
            if not self.paused:
                state,frame = self.cap.read()
                if not state:
                    print 'video end'
                    continue
                self.frame = frame.copy()
            show_img = self.frame.copy()
          
            if not self.paused and not  self.rect_selector.dragging:
                tracked = self.track(self.frame)
                if len(tracked)>0:
                    for tr in tracked:
                        if tr >0:
                            cv2.polylines(show_img, [np.int32(tr.quad)], True, (255, 255, 255), 2)
                if not self.paused:
                    keypoints,_ = self.detect_features(show_img)
                    common.draw_keypoints(show_img, keypoints)
                '''
                if len(tracked)>0:
                    tracked = tracked[0]
                    cv2.polylines(show_img, [np.int32(tracked.quad)], True, (255, 255, 255), 2)
                    '''
            self.rect_selector.draw(show_img)
            cv2.imshow('src',show_img)
            #videowriter.write(show_img)
            key = cv2.waitKey(10)
            if key == -1:
                continue
            elif key == 27:
                break
            elif key == ord(' '):
                self.paused = not self.paused
                print self.paused
            elif key == ord('c'):
                self.targets = []
                self.matcher.clear()

if __name__ == '__main__':
    Mainfunc('133.avi').run()
