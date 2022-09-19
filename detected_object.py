class DetectedObject():
        
    def __init__(self, bbox, object_class, score, isout = False):
        """
        :param bbox: in format [xmin, ymin, xmax, ymax]
        :param object_class:
        :param score:
        :param isout:
        """
        self.bbox = bbox
        self.object_class = object_class
        self.score = score
        self.isout = isout
        
    def get_xcycwh(self):
        return [int(self.bbox[0]+(self.bbox[2]-self.bbox[0])/2), int(self.bbox[1]+(self.bbox[3]-self.bbox[1])/2), self.bbox[2]-self.bbox[0], self.bbox[3]-self.bbox[1]]
    
    def get_xtlytlwh(self):
        return [self.bbox[0], self.bbox[1], self.bbox[2]-self.bbox[0], self.bbox[3]-self.bbox[1]]
