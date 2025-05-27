import SimpleITK as sitk
import numpy as np
import seg_metrics.seg_metrics as sm
import pandas as pd

from collections.abc import Iterable

def calculateVolumeRange(image : sitk.Image, lowest_label_value : int, highest_label_value : int) -> int:
    mask = sitk.BinaryThreshold(image, lowerThreshold=lowest_label_value, upperThreshold=highest_label_value)

    return sitk.GetArrayFromImage(mask).sum()

def calculateVolume(image : sitk.Image, label_value : int | Iterable[int]) -> int:
    if type(label_value) == int:
        return calculateVolumeRange(image, label_value, label_value)
    else:
        return sum([calculateVolumeRange(image, value, value) for value in label_value])

class LabelMetrics:
    def __init__(self, volume_weight : int, dice : float, hd95 : float, msd : float) -> None:    
        self.volume_weight = volume_weight
        self.dice = dice
        self.hd95 = hd95
        self.msd = msd

    def __str__(self) -> str:
        return f"Dice: {self.dice} hd95:{self.hd95} msd:{self.msd}"

class ImageMetrics:
    def __init__(self) -> None:
        self.label_metrics = {}

    def addLabel(self, label : str, metrics : LabelMetrics) -> None:
        self.label_metrics[label] = metrics

    def getLabelMetrics(self, label : str) -> LabelMetrics:
        return self.label_metrics.get(label)
    
    def getAvgMetrics(self) -> LabelMetrics:
        dice = 0.0
        hd95 = 0.0
        msd = 0.0

        volume = 0.0

        for label_metric in self.label_metrics.values():
            dice += label_metric.volume_weight * label_metric.dice
            hd95 += label_metric.volume_weight * label_metric.hd95
            msd += label_metric.volume_weight * label_metric.msd
            
            volume += label_metric.volume_weight

        return LabelMetrics(volume, dice, hd95, msd)

def evaluateImage(image : sitk.Image, ground_truth : sitk.Image, labels_dict : dict) -> ImageMetrics:
    if image.GetSize() != ground_truth.GetSize():
        raise AttributeError("image size is not consistent with ground truth")

    result = ImageMetrics()

    total_volume = calculateVolume(ground_truth, labels_dict.values())

    for label, label_value in labels_dict.items():
        metrics = sm.write_metrics(labels=[label_value],
                  gdth_img=ground_truth,
                  pred_img=image,
                  csv_file=None,
                  metrics=['dice', 'hd95', 'msd'])[0]
        
        relative_volume = calculateVolumeRange(ground_truth, label_value, label_value) / total_volume
        
        result.addLabel(label, LabelMetrics(relative_volume, metrics['dice'][0], metrics['hd95'][0], metrics['msd'][0]))

    return result

def createRecord(name : str, metrics : ImageMetrics) -> dict:
    result = {}
    result["id"] = name

    avg_metrics = metrics.getAvgMetrics()
    result['dice'] = avg_metrics.dice
    result['hd95'] = avg_metrics.hd95
    result['msd'] = avg_metrics.msd
    
    for label, label_metrics in metrics.label_metrics.items():
        result[f"{label}_volume"] = label_metrics.volume_weight
        result[f"{label}_dice"] = label_metrics.dice
        result[f"{label}_hd95"] = label_metrics.hd95
        result[f"{label}_msd"] = label_metrics.msd

    return result

def createDataFrame(records : list[tuple[str,ImageMetrics]]) -> None:
    data = [createRecord(*record) for record in records]

    return pd.DataFrame(data)
