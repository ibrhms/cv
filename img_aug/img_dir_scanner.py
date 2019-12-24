import os
import glob

class img_dir_scanner(object):
    """
    The ImageSource class is used to search for and contain paths to images for augmentation.
    """
    def __init__(self, source_directory, recursive_scan=False):
        self.source_directory = os.path.abspath(source_directory)      

        self.scan_directory(recursive_scan)
        
        self.largest_file_dimensions = (800, 600)


    def scan_directory(self, recusrive_scan=False):
        # TODO: Make this a static member somewhere later
        file_types = ['*.jpg', '*.bmp', '*.jpeg', '*.gif', '*.img', '*.png']
        #file_types.extend([str.upper(x) for x in file_types])

        list_of_files = []
        for file_type in file_types:
            list_of_files.extend(glob.glob(os.path.join(os.path.abspath(self.source_directory), file_type)))

        self.image_list =  list_of_files
        self.basenames, self.basenames_wo_ext, self.basenames_ext = self.get_basename(self.image_list)
        self.image_number = len(self.image_list)

    
    def get_basename(self, list_of_filles):
        basenames = []
        basenames_wo_ext = []
        basenames_ext = []
        for filename in list_of_filles:
            basename = os.path.basename(filename)
            basename_wo_ext = os.path.splitext(basename)[0]
            basename_ext = os.path.splitext(basename)[1]
            
            basenames_wo_ext.append(basename_wo_ext)
            basenames_ext.append(basename_ext)
            basenames.append(basename)

        return basenames, basenames_wo_ext, basenames_ext

    
