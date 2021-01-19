from .base_dataset import BaseDataset
import os


class CelebADataset(BaseDataset):
    """docstring for CelebADataset"""
    def __init__(self):
        super(CelebADataset, self).__init__()
        
    def initialize(self, opt):
        super(CelebADataset, self).initialize(opt)

    def get_aus_by_path(self, img_path):
        assert os.path.isfile(img_path), "Cannot find image file: %s" % img_path
        img_id = str(os.path.splitext(os.path.basename(img_path))[0])
        return self.aus_dict[img_id] / 5.0   # norm to [0, 1]

    def make_dataset(self):
        # return all image full path in a list
        return [self.opt.img_path]

    def __getitem__(self, index):
        img_path = self.imgs_path[index]

        # load source image
        src_img = self.get_img_by_path(img_path)
        src_img_tensor = self.img2tensor(src_img)

        # record paths for debug and test usage
        data_dict = {'src_img': src_img_tensor, 'src_path': img_path}
        return data_dict
