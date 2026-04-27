import gphoto2 as gp
import os
from datetime import datetime
import check_exposure


# Stop service blocking camera access
os.system("systemctl --user stop gvfs-gphoto2-volume-monitor.service")

camera = gp.Camera()
camera.init()
cam_config = camera.get_config()


# # Store images in SD card
# if os.system("gphoto2 --set-config-index capturetarget=1") != 0:
#     print("No camera detected! Aborting operation")
#     os.exit(-1)


def get_shutterspeed():
    ok, shutterspeed_widget = gp.gp_widget_get_child_by_name(cam_config, 'shutterspeed')
    current_idx = None
    num_choices = None
    if ok >= gp.GP_OK:
        current_val = shutterspeed_widget.get_value()
        num_choices = shutterspeed_widget.count_choices()
        choices = [shutterspeed_widget.get_choice(i) for i in range(num_choices)]
        current_idx = choices.index(current_val)

    return current_idx, num_choices
        

def set_shutterspeed(ss):
    ok, shutterspeed_widget = gp.gp_widget_get_child_by_name(cam_config, 'shutterspeed')
    ss_val = shutterspeed_widget.get_choice(ss)
    print(f"Setting shuterspeed to {ss_val}")
    shutterspeed_widget.set_value(ss_val)
    camera.set_config(cam_config)


def flush_events():
    while True:
        event_type, event_data = camera.wait_for_event(1) # Minimum timeout
        if event_type == gp.GP_EVENT_TIMEOUT:
            break

def delete_file(folder, name):
    try:
        camera.file_delete(folder, name)
    except:
        print("Delete operation not supported in this camera")

def shoot():
    flush_events() # just in case...

    downloaded_files = []

    # Capture returns first image (RAW if enabled)
    file_path = camera.capture(gp.GP_CAPTURE_IMAGE)
    print(f"CAPTURE {file_path.folder, file_path.name}")
    target_path = os.path.basename(file_path.name)
    camera_file = camera.file_get(file_path.folder, file_path.name, gp.GP_FILE_TYPE_NORMAL)
    camera_file.save(target_path)
    delete_file(file_path.folder, file_path.name)
    downloaded_files.append(target_path)

    if not target_path.endswith(".JPG"):
        # Try to download the JPG version
        jpeg_filename = os.path.splitext(target_path)[0] + ".JPG"
        camera_file = camera.file_get(file_path.folder, jpeg_filename, gp.GP_FILE_TYPE_NORMAL)
        camera_file.save(jpeg_filename)
        delete_file(file_path.folder, jpeg_filename)
        downloaded_files.append(jpeg_filename)

    # timeout = 2000 # ms
    # while True:
    #     event_type, event_data = camera.wait_for_event(timeout)

    #     if event_type == gp.GP_EVENT_TIMEOUT:
    #         print("TIMEOUT!")
    #         break;
    
    #     if event_type == gp.GP_EVENT_FILE_ADDED:
    #         print("file added: ", event_data.folder, event_data.name)
    #         file_path = event_data
    #         target_path = os.path.basename(file_path.name)
    #         camera_file = camera.file_get(file_path.folder, file_path.name, gp.GP_FILE_TYPE_NORMAL)
    #         camera_file.save(target_path)
    #         camera.file_delete(file_path.folder, file_path.name)
    #         downloaded_files.append(target_path)
    #     else:
    #         print("Unknown event:", event_type, event_data)

    return downloaded_files

def get_jpg(file_list):
    for f in file_list:
        if f.endswith(".JPG"):
            return f        
    return None

def store_photos(folder_name, file_list):
    if not os.path.isdir(folder_name):
        os.mkdirs(folder_name)
    for f in file_list:
        ext = os.path.splitext(f)[1].split(".")[1]
        dest_dir = os.path.join(folder_name, ext)
        if not os.path.isdir(dest_dir):
            os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, f)
        print(f"Moving file {f} to {dest}")
        os.rename(f, dest)
        

# Create directory for the capture
timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
folder_name = f"capture_{timestamp}"
os.makedirs(folder_name, exist_ok=True)

center_ss, num_ss = get_shutterspeed()
center_shot = shoot()
center_exp = check_exposure.histogramCheck(get_jpg(center_shot))
current_ss = center_ss
current_shoot = center_shot
current_exp = center_exp
store_photos(folder_name, current_shoot)
# Exposure difference between shots - Canon steps = 1/3 EV (f-stop)
ss_step = 3

exposure_ok = True

# Crawl down the histogram
while True:
    if current_exp & 1 == 0 or current_ss == 0:
        # Not underexposed or shutter speed is at minimum
        break

    if current_exp & 1 == 0:
        # If not underexposed, we're done here
        break
    elif current_ss == 0:
        # If shutter speed is at minimum, we cannot reach the DR of the scene
        print("CATASTROPHIC ERROR!!! - cannot capture full dynamic range for this scene")
        print("The image is UNDEREXPOSED")
        print("Try increasing ISO value, remove any filters, or use another camera")
        exposure_ok = False
        break

    # Underexposed
    # increase shutter time / decrease shutter speed
    current_ss = max(0, current_ss - ss_step)
    set_shutterspeed(current_ss)
    current_shoot = shoot()
    current_exp = check_exposure.histogramCheck(get_jpg(current_shoot))
    store_photos(folder_name, current_shoot)

# Crawl up the histogram
current_ss = center_ss
current_exp = center_exp
while True:
    if current_exp & 2 == 0:
        # If not overexposed, we're done here
        break
    elif current_ss == num_ss-1:
        # If shutter speed is at maximum, we cannot reach the DR of the scene
        print("CATASTROPHIC ERROR!!! - cannot capture full dynamic range for this scene")
        print("The image is OVEREXPOSED")
        print("Try decreasing ISO value, add ND filters, or use another camera")
        exposure_ok = False
        break

    # OverexposeduRE
    # decrease shutter time / increase shutter speed
    current_ss = min(num_ss-1, current_ss + ss_step)
    set_shutterspeed(current_ss)
    current_shoot = shoot()
    current_exp = check_exposure.histogramCheck(get_jpg(current_shoot))
    store_photos(folder_name, current_shoot)

camera.exit()

if exposure_ok:

    # HDR generation
    formats = [
        "JPG",
        "CR2",
        "ARW"
    ]
    for f in formats:
        if os.path.exists(os.path.join(folder_name, f)):
            os.system(f"luminance-hdr-cli --align AIS --save {folder_name}_JPG.exr {folder_name}/{f}/*.{f}")
    #os.system(f"luminance-hdr-cli --align AIS --save {folder_name}_JPG.exr {folder_name}/JPG/*.JPG")

else:
    print("Cannot reach dynamic range of this scene")
    os.exit(-1)

print("Done!")
