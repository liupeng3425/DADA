from advent.dataset.cityscapes import CityscapesDataSet

from dada.dataset.depth import get_depth_cityscapes


class CityscapesD(CityscapesDataSet):

    def __getitem__(self, index):
        image, label, shape, name = super().__getitem__(index)

        depth__name = name.replace("leftImg8bit", "disparity")
        depth_file = self.root / 'disparity' / self.set / depth__name

        depth = self.get_depth(depth_file)
        return image, label, shape, name, depth

    def get_depth(self, file):
        return get_depth_cityscapes(self, file)
