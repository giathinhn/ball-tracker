from collections import defaultdict, Counter
import numpy as np
import supervision as sv

def associate_tracker_ids(original: sv.Detections, tracked: sv.Detections) -> np.ndarray:
    """
    Associates tracker_ids from tracked detections back to original detections
    based on bounding box IoU (Intersection over Union).
    Returns an array of tracker_ids matching the length and order of original.
    """
    if len(original) == 0 or len(tracked) == 0:
        return np.array([None] * len(original))
        
    tracker_ids = np.array([None] * len(original))
    
    for i, box_orig in enumerate(original.xyxy):
        best_iou = 0.0
        best_tid = None
        for j, box_track in enumerate(tracked.xyxy):
            # Calculate IoU
            x_min = max(box_orig[0], box_track[0])
            y_min = max(box_orig[1], box_track[1])
            x_max = min(box_orig[2], box_track[2])
            y_max = min(box_orig[3], box_track[3])
            
            if x_max > x_min and y_max > y_min:
                inter_area = (x_max - x_min) * (y_max - y_min)
                orig_area = (box_orig[2] - box_orig[0]) * (box_orig[3] - box_orig[1])
                track_area = (box_track[2] - box_track[0]) * (box_track[3] - box_track[1])
                union_area = orig_area + track_area - inter_area
                iou = inter_area / union_area if union_area > 0 else 0.0
                
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tracked.tracker_id[j] if tracked.tracker_id is not None else None
                    
        if best_iou > 0.5:
            tracker_ids[i] = best_tid
            
    return tracker_ids

class TeamIdSmoother:
    """
    Smooths team predictions over time for each tracked player ID
    using a historical majority vote.
    """
    def __init__(self):
        self.history = defaultdict(list)

    def smooth(self, tracker_ids: np.ndarray, predicted_team_ids: np.ndarray) -> np.ndarray:
        if tracker_ids is None or len(tracker_ids) == 0:
            return predicted_team_ids

        smoothed_team_ids = np.copy(predicted_team_ids)
        for i, tid in enumerate(tracker_ids):
            if tid is not None and not (isinstance(tid, float) and np.isnan(tid)):
                try:
                    tid_int = int(tid)
                    self.history[tid_int].append(predicted_team_ids[i])
                    
                    # Retrieve the most common team ID assigned to this tracker ID
                    votes = Counter(self.history[tid_int])
                    smoothed_team_ids[i] = votes.most_common(1)[0][0]
                except (ValueError, TypeError):
                    pass
        return smoothed_team_ids
