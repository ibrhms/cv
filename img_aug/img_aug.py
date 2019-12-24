import sys
from img_dir_scanner import img_dir_scanner
from image_augmentor import Flip
from image_augmentor import Skew
from image_augmentor import RandomBrightness
from PIL import Image

if __name__ == "__main__":
    #img_dir = "D://Documents//programming//bram_playground//img_aug//test_images//"
    img_dir = str(sys.argv[1])
    imgSource = img_dir_scanner(img_dir)

    # flipper = Flip("LEFT_RIGHT")    
    # for i in range(0,imgSource.image_number):
    #     im = Image.open(imgSource.image_list[i])
    #     flipped = flipper.perform_operation(im)
    #     flipped.save(img_dir +"flipped_"+imgSource.basenames[i])

    magnitude = 0.075
    
    skewer_tilt_lr = Skew("TILT_LEFT_RIGHT", magnitude)
    imgSource.scan_directory()
    for i in range(0,imgSource.image_number):
        im = Image.open(imgSource.image_list[i])
        skewed_lr = skewer_tilt_lr.perform_operation(im)
        skewed_lr.save(img_dir +"skewed_lr_"+imgSource.basenames[i])


    skewer_tilt_tb = Skew("TILT_TOP_BOTTOM", magnitude)
    imgSource.scan_directory()
    for i in range(0,imgSource.image_number):
        im = Image.open(imgSource.image_list[i])
        skewed_tb = skewer_tilt_tb.perform_operation(im)
        skewed_tb.save(img_dir +"skewed_tb_"+imgSource.basenames[i])
   
    brighter = RandomBrightness()    
    imgSource.scan_directory()
    for i in range(0,imgSource.image_number):
        im = Image.open(imgSource.image_list[i])
        brighted = brighter.perform_operation(im)
        brighted.save(img_dir +"brighted"+imgSource.basenames[i])
    #print(img_list)
    #print(imgSource.image_list[0])


    #img_dir = str(sys.argv[1])
    #num_aug_img = int(sys.argv[2])
    # p = Augmentor.Pipeline(img_dir)
    # p.flip_left_right(probability= 0.75)
    # p.skew(probability = 0.75,magnitude=0.1)
    # p.sample(num_aug_img)