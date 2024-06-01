from Detector import *
import os

def main():
    base_path = "C:/Users/iremb/Desktop/opencv/real_time_object_detection_cpu-main"
    #videoPath = os.path.join("C:/Users/iremb/Desktop/opencv", "street.mp4")
    videoPath = 0
    configPath = os.path.join(base_path, "model_data", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    modelPath = os.path.join(base_path, "model_data", "frozen_inference_graph.pb")
    classesPath = os.path.join(base_path, "model_data", "coco.names")

    # Dosya yollarını kontrol etme
    paths = [videoPath, configPath, modelPath, classesPath]
    for path in paths:
        if not os.path.exists(path):
            print(f"Dosya mevcut değil: {path}")
            return
        else:
            print(f"Dosya bulundu: {path}")

    detector = Detector(videoPath, configPath, modelPath, classesPath)
    detector.readClasses()
    detector.onVideo()

if __name__ == "__main__":
    main()
