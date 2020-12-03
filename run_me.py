# from skimage.filters import threshold_multiotsu       # it will work with python 3.6
import os
import glob
import sys
import math
import cv2
import time
import ntpath
import numpy as np
import scipy.io as scipy_io
from PIL import Image
import skimage.io as skimage_io
from skimage.color import rgb2hsv
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_fill_holes
from skimage.measure import label, regionprops
from matlab import engine
from keras.applications import imagenet_utils
import matplotlib.patches as patches

import set_parameters as param
from networkClass import networkClass

class WSI_Processor:
    def __init__(self):
        self.wsi_dir = param.wsi_dir
        self.all_wsi_path = glob.glob(self.wsi_dir + "/*.jp2")
        if self.all_wsi_path != []:
            try:
                self.matlabObj = engine.start_matlab()
                self.matlabObj.cd(os.path.join(os.getcwd(), 'matlab'), nargout=0)
                # self.matlabObj.cd(os.path.join(os.path.normpath(os.getcwd() + os.sep + os.pardir), 'matlab'), nargout=0)
            except:
                print("Matlab Engine not started...")
        else:
            print('Only jp2 images are accepted. Add code for reading images of type other than jp2')
            sys.exit(0)

        self.patch_size = param.network_patch_size
        self.batch_size = param.batch_size
        self.block_size = 128           # for blur check
        self.image_patch_size = 5120
        self.mask_level = 7
        self.max_pixels = 2000000  # level 7 for extracting cellular region from the WSI
        self.wsi_output_dir = []
        self.patch_data = { 'mask_row': [],
                            'mask_col': [],
                            'mask_level': [],
                            'image_row': [],
                            'image_col': [],
                            'image_level': []}
        self.preprocessFunction = imagenet_utils.preprocess_input

    def get_bbox_cell(self, cell_mask):
        label_mask = label(cell_mask)

        regions = regionprops(label_mask)
        all_area = [ele.area for ele in regions]
        all_bbox = [ele.bbox for ele in regions]
        all_area = np.array(all_area)
        all_bbox = np.array(all_bbox)

        # modifying the bbox to make it like [x y width height]
        temp = all_bbox[:, 3] - all_bbox[:, 1]
        all_bbox[:, 3] = all_bbox[:, 2] - all_bbox[:, 0]
        all_bbox[:, 2] = temp
        all_bbox[:,0:2] = np.concatenate((all_bbox[:,1].reshape(-1,1), all_bbox[:,0].reshape(-1,1)), axis=1)

        if all_area.size != 0:
            ind = np.where((all_area >= 1700000) | (all_area <= 441) | (all_bbox[:,2]<21) | (all_bbox[:,3]<21)
                                                                     | (all_bbox[:,2]>1000) | (all_bbox[:,3]>1000))
            all_bbox = np.delete(all_bbox, ind, axis=0)

        return all_bbox, cell_mask

    def get_cell_mask_without_watershed_test_final(self, img, show_image):
        # img = imread('D:\\Dropbox\\PhD_Work\\PROJECTS\\Cytology_Project\\Annotations\\Annotated_Cell_Patches\\v3_variable_size\\MERGED_7Classes\\valid_exemplars\\image_for_threshold_selection.png')
        # img = imread('D:\\Dropbox\\PhD_Work\\PROJECTS\\Cytology_Project\\Annotations\\Annotated_Cell_Patches\\v3_variable_size\\MERGED_7Classes\\valid_exemplars\\patch_rectangle_65.png')
        hsv_img = rgb2hsv(img)
        s_channel = hsv_img[:,:,1]

        # thresholds = threshold_multiotsu(s_channel, classes=5)
        thresholds = [0.0510, 0.1451, 0.2392, 0.7647]
        thresh_mask = np.digitize(s_channel, bins=thresholds)
        thresh_mask[thresh_mask > 0] = 1
        thresh_mask = binary_fill_holes(thresh_mask).astype(int)

        if np.sum(thresh_mask) < 900:
            seg_M = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        else:
            seg_M = thresh_mask

        if show_image:
            plt.figure(figsize=(15, 6))
            plt.subplot(1, 3, 1)
            plt.imshow(img)
            plt.subplot(1, 3, 2)
            plt.imshow(thresh_mask)
            plt.subplot(1, 3, 3)
            plt.imshow(seg_M)
            plt.show()
        return seg_M

    def get_data(self, patchM, file_path):
        patchI, bbox, cellMask = [], [], []
        if np.sum(patchM) > (mask_patch_size ** 2) * 50 / 100:
            patchI, cellMask = obj.matlabObj.read_image_region(file_path, self.patch_data['image_row'][0], self.patch_data['image_row'][1],
                                                                          self.patch_data['image_col'][0], self.patch_data['image_col'][1],
                                                                          self.patch_data['image_level'], self.block_size, nargout=2)

            patchI = np.array(patchI._data).reshape(patchI.size, order='F')
            cellMask = np.array(cellMask._data).reshape(cellMask.size, order='F')

            patchM = Image.fromarray(patchM)
            patchM = np.array(patchM.resize((5120, 5120), Image.NEAREST))
            cellMask = cellMask * patchM

            if np.sum(cellMask) > 0:
                bbox, cellMask = obj.get_bbox_cell(cellMask)
        return patchI, bbox, cellMask

    def save_intermediate_output(self, patchI, cell_pred, bbox, save_filename):
        pred_color = ['g', 'c', 'y', 'b', 'm', 'r', 'k']
        fig, ax = plt.subplots(1)
        ax.imshow(patchI)
        for i, current_bbox in enumerate(bbox):
            current_bbox = np.array(current_bbox, dtype=int)
            rect = patches.Rectangle((current_bbox[0], current_bbox[1]), current_bbox[2], current_bbox[3], linewidth=1,
                                     edgecolor=pred_color[cell_pred[i]], facecolor='none')
            ax.add_patch(rect)
        # plt.show()
        plt.savefig(os.path.join(self.wsi_output_dir, save_filename))
        plt.close()

    def overlay_bbox(self, patch_I, bbox_info):
        fig, ax = plt.subplots(1)
        ax.imshow(patch_I)
        for current_bbox in bbox_info:
            current_bbox = np.array(current_bbox, dtype=int)
            rect = patches.Rectangle((current_bbox[0],current_bbox[1]),current_bbox[2],current_bbox[3],linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
        plt.show()

    def predict(self, model, patch_I, bbox_info):
        patch_pred_prob = []
        patch_pred_label = []

        for i, iBatch in enumerate(range(0, len(bbox_info), self.batch_size)):
            if i == int(len(bbox_info) / self.batch_size):
                batch_bbox = bbox_info[iBatch:len(bbox_info)]
            else:
                batch_bbox = bbox_info[iBatch:iBatch + self.batch_size]

            test_patches = []
            for current_bbox in batch_bbox:
                current_bbox = np.array(current_bbox, dtype=int)
                cell_patch = patch_I[current_bbox[0]:current_bbox[0] + current_bbox[2],
                             current_bbox[1]:current_bbox[1] + current_bbox[3], :]
                # plt.imshow(cell_patch)
                # plt.show()
                cell_patch = cv2.resize(cell_patch,
                                        (self.patch_size, self.patch_size),
                                        interpolation=cv2.INTER_CUBIC)
                test_patches.append(cell_patch)

            test_patches = np.array(test_patches)
            test_patches = self.preprocessFunction(test_patches)
            # test_patches = test_patches/255
            pred_prob = model.predict(test_patches, batch_size=self.batch_size, verbose=0, steps=None)
            pred_label = np.argmax(pred_prob, axis=1)

            patch_pred_prob.extend(pred_prob)
            patch_pred_label.extend(pred_label)

        patch_pred_prob = np.array(patch_pred_prob)
        patch_pred_label = np.array(patch_pred_label)
        patch_absolute_bbox = bbox_info[:, 0:2] + [self.patch_data['image_col'][0], self.patch_data['image_row'][0]]
        patch_absolute_bbox = np.append(patch_absolute_bbox, bbox_info[:, 2:4], axis = 1)

        return patch_pred_label, patch_pred_prob, patch_absolute_bbox

if __name__ == "__main__":
    net_obj = networkClass(param)
    model = net_obj.get_model()
    net_obj.optimize(model)
    net_obj.load_checkpoint(model)

    obj = WSI_Processor()
    for file_path in obj.all_wsi_path:
        all_pred, all_prob, all_bbox = [], [], []
        _, file_name = ntpath.split(file_path)

        start_time = time.time()
        obj.patch_data['mask_level'] = obj.matlabObj.getMaskLevel(file_path, obj.max_pixels)
        obj.patch_data['image_level'] = 0 if obj.patch_data['mask_level'] == obj.mask_level else 1
        print('If the total number of levels in an image pyramid is not equal to 11 or 12 then adjust the code accordingly.')
        mask_patch_size = int(obj.image_patch_size / (2 ** (obj.patch_data['mask_level'] - float(obj.patch_data['image_level']))))

        ROI = obj.matlabObj.detect_cellular_region(file_path, obj.patch_data['mask_level'], 0)
        ROI = np.array(ROI._data).reshape(ROI.size, order='F')

        if param.save_intermediate_results:
            skimage_io.imsave(os.path.join(obj.wsi_dir, file_name[:-4] + '_level_' + str(int(obj.patch_data['mask_level'])) + '.tif'), np.uint8(ROI) * 255)
            obj.wsi_output_dir = os.path.join(param.output_path, file_name[:-4])
            os.makedirs(obj.wsi_output_dir, exist_ok=True)

        no_rows = math.floor(ROI.shape[0]/mask_patch_size)
        no_cols = math.floor(ROI.shape[1] / mask_patch_size)
        print('%d Rows and %d Columns of a mask at level %d.'%(no_rows, no_cols, obj.patch_data['mask_level']))

        incr = 1
        for iRow in range(0, no_rows):
            obj.patch_data['mask_row'] = [iRow * mask_patch_size, (iRow * mask_patch_size) + mask_patch_size]
            obj.patch_data['image_row'] = [iRow * obj.image_patch_size, (iRow * obj.image_patch_size) + obj.image_patch_size]
            print('\n %2d'%(iRow+1), end='')

            for iCol in range(0, no_cols):
                # print(str(iRow) + ':' + str(iCol))
                obj.patch_data['mask_col'] = [iCol * mask_patch_size, (iCol * mask_patch_size) + mask_patch_size]
                obj.patch_data['image_col'] = [iCol * obj.image_patch_size, (iCol * obj.image_patch_size) + obj.image_patch_size]

                patchM = ROI[obj.patch_data['mask_row'][0]: obj.patch_data['mask_row'][1],
                             obj.patch_data['mask_col'][0]: obj.patch_data['mask_col'][1]]

                patchI, bbox, cellMask = obj.get_data(patchM, file_path)
                if len(bbox) != 0:
                    print(' 0', end='')
                    # obj.overlay_bbox(patchI, bbox)
                    cell_pred, cell_prob, cell_bbox = obj.predict(model, patchI, bbox)
                    if param.save_intermediate_results:
                        save_filename = 'cyto_%d_row%d_col%d' % (incr, iRow, iCol)
                        obj.save_intermediate_output(patchI, cell_pred, bbox, save_filename + '.png')
                        skimage_io.imsave(os.path.join(obj.wsi_output_dir,  save_filename + '.tif'), np.uint8(cellMask) * 255)
                        incr = incr + 1
                    all_pred.extend(cell_pred)
                    all_prob.extend(cell_prob)
                    all_bbox.extend(cell_bbox)
                else:
                    print(' #', end='')

        all_pred = np.array(all_pred)
        all_prob = np.array(all_prob)
        all_bbox = np.array(all_bbox)
        end_time = time.time() - start_time
        print('%s [WSI Processing Time in seconds: %.2f ]' % (file_name, end_time))
        scipy_io.savemat(os.path.join(obj.wsi_dir, file_name[:-3] + 'mat'), {'pred':all_pred, 'prob': all_prob, 'bbox': all_bbox})
        print('Done')



