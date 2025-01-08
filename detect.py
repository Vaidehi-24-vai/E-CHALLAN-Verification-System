import cv2
import argparse
import csv
import os
import platform
import sys
from pathlib import Path
from text_recognition import perform_ocr
import torch
import numpy as np
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  
try:
    from ultralytics.utils.plotting import Annotator, colors, save_one_box

    from models.common import DetectMultiBackend
    from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
    from utils.general import (
         LOGGER,
         Profile,
         check_file,
         check_img_size,
         check_imshow,
         check_requirements,
         colorstr,
         cv2,
         increment_path,
         non_max_suppression,
         print_args,
         scale_boxes,
         strip_optimizer,
         xyxy2xywh,
    )
    from utils.torch_utils import select_device, smart_inference_mode
    MODULES_IMPORTED = True
except ImportError as e:
    MODULES_IMPORTED = False
    print("Error importing necessary modules:", e)


@smart_inference_mode()
def run(
    weights="E:/YOLO-V5/yolov5-master/yolov5-master/runs/train/exp/weights/best.pt",  # model path or triton URL
    image_path="",  # path to the input image for verification
    number_plate=None,
    source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_csv=False,  # save results in CSV format
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
):
    detected_image_path, result = None, None 
    source = str(image_path) if image_path else str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    #weights = str(weights)  # Convert to string
    print("Loading weights from:", weights) 
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize)
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file
        csv_path = save_dir / "predictions.csv"

        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)



        def check_valid_parking1(v_bbox, p_spots):
            """
            Check if the entire vehicle is within the boundaries of any parking spot.

            Args:
                vehicle_bbox (Tuple): Bounding box coordinates (left, top, right, bottom) of the vehicle.
                parking_spots (List[Tuple[Tuple[int, int], Tuple[int, int]]]): List of tuples representing parking spots, each with ((left, top), (right, bottom)).


            Returns:
                bool: True if the entire vehicle is within the boundaries of any parking spot, False otherwise.
            """

            print("Vehicle BBox:", v_bbox)
            print("Parking Spots:", p_spots)

            if len(v_bbox) != 4:
                print("Invalid vehicle bounding box format. Expected 4 values (left, top, right, bottom).")
                return False

            v_left, v_top, v_right, v_bottom = v_bbox
            v_center = ((v_left + v_right) / 2, (v_top + v_bottom) / 2)



            for s in p_spots:
                if isinstance(s, tuple) and len(s) == 2 and all(isinstance(coord, tuple) and len(coord) == 2 for coord in s):
                    s_left, s_top = s[0]
                    s_right, s_bottom = s[1]
            
                    if s_left <= v_center[0] <= s_right and s_top <= v_center[1] <= s_bottom:
                       return True
    
            print("Invalid parking or no available spot for the vehicle.")
            #print("Invalid vehicle bounding box format:", vehicle_bbox)
                   
            return False

        """ 
        def is_point_within_line(point, line, tolerance=5):
            #print("Contents of line variable:", line)
            (x1, y1), (x2, y2) = line
            px, py = point
            dx = x2 - x1
            dy = y2 - y1
            distance = abs(dy * px - dx * py + x2 * y1 - y2 * x1) / ((dx ** 2 + dy ** 2) ** 0.5)
            return distance <= tolerance
        """
       
        def d_parking_lines(input_image):
             #Your code for detecting parking lines goes here
             g_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
             e = cv2.Canny(g_image, 50, 150, apertureSize=3)
             l = cv2.HoughLinesP(e, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
             d_lines = []
             if l is not None:
                for l1 in l:
                    x1, y1, x2, y2 = l1[0]
                    d_lines.append(((x1, y1), (x2, y2)))

             return d_lines
  # Example: list of tuples representing line coordinates
             #return detected_lines
        

        # Process predictions
        for i, det1 in enumerate(pred):  # per image
            seen += 1
            if webcam:  
               p, im0, f1 = path[i], im0s[i].copy(), dataset.count
               s += f"{i}: "
            else:
                p, im0, f1 = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  
            save_path = str(save_dir / p.name) 
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{f1}")  
            s += "%gx%g " % im.shape[2:]  
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  
            imc = im0.copy() if save_crop else im0  
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det1):
               
                det1[:, :4] = scale_boxes(im.shape[2:], det1[:, :4], im0.shape).round()

                for c in det1[:, 5].unique():
                    n = (det1[:, 5] == c).sum()  
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  

                # Write results
                for *xyxy, conf, cls in reversed(det1):
                    c = int(cls) 
                    label = names[c] if hide_conf else f"{names[c]}"
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"

                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)
                for *xyxy, conf, cls in reversed(det1):
                    if int(cls) == 2:  
        
                       p_img = im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]

        
                       cv2.imwrite(f"number_plate_{seen}.jpg", p_img)

        
                       o_result = perform_ocr(p_img)

        
                       print("OCR Result:", o_result)

                       #e_plate = input("Enter the number plate number: ")
                       e_plate = number_plate
                       print(e_plate)
                       d_plate = " ".join(o_result).strip()

                       if e_plate == d_plate:
                          print("Number plate matched. Checking parking validity...")
                          v_left = min(xyxy[0], xyxy[2])
                          v_top = min(xyxy[1], xyxy[3])
                          v_right = max(xyxy[0], xyxy[2])
                          v_bottom = max(xyxy[1], xyxy[3])

        # Create the bounding box tuple
                          v_bbox = (v_left, v_top, v_right, v_bottom)

                       
        # Detect parking lines
                          d_lines = d_parking_lines(im0)

        # Check if the parking is valid
                          if check_valid_parking1(v_bbox, d_lines):
                             print("Valid Parking")
                             result = "Valid Parking"
                          else:
                             print("Invalid Parking")
                             result = "Invalid Parking"
                       else:
                            print("Number plate does not match. Vehicle not found.")
            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                detected_image_path = save_path
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                    print("Saved image to:", save_path)
                    detected_image_path = save_path
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)
                
                #print("Detected Image Path:", detected_image_path)  # Debugging print
                #detected_image_path = save_path
                #print("Updated Detected Image Path:", detected_image_path)  # Debugging print
        if detected_image_path is None:
           detected_image_path = "path_to_default_image.jpg"  # Replace "path_to_default_image.jpg" with the path to your default image
         # Print detected_image_path for debugging
        print("Detected Image Path:", detected_image_path)  # Debugging print
        LOGGER.info(f"{s}{'' if len(det1) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
    detected_image_path= save_path
    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
    return detected_image_path, result
   

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=None, help="model path or triton URL")
    parser.add_argument("--image-path", type=str, default="", help="path to the input image for verification")
    parser.add_argument("--number-plate", type=str, default=None, help="number plate input")
    parser.add_argument("--source", type=str, default=ROOT / "E:/YOLO-V5/yolov5-master/yolov5-master/inference/images", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "E:/YOLO-V5/yolov5-master/yolov5-master/data/custom_data.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    opt = parser.parse_args()
    #opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    #print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    print("Using weights from:", opt.weights)
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
