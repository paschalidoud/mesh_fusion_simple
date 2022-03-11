import numpy as np
try:
    from functools import lru_cache
except ImportError:
    from backports.functools_lru_cache import lru_cache
from collections import OrderedDict
import os
from PIL import Image

from .mesh import read_mesh_file


class BaseModel(object):
    """BaseModel class is wrapper for all models, independent of dataset. Every
       model has a unique model_tag, mesh_file and Mesh object. Optionally, it
       can also have a tsdf file.
    """
    def __init__(self, tag):
        self._tag = tag
        # Initialize the contents of this instance to empty so that they can be
        # lazy loaded
        self._gt_mesh = None
        self._images = []
        self._image_paths = None

    @property
    def tag(self):
        return self._tag

    @property
    def path_to_mesh_file(self):
        raise NotImplementedError()

    @property
    def images_dir(self):
        raise NotImplementedError()

    @property
    def groundtruth_mesh(self):
        if self._gt_mesh is None:
            self._gt_mesh = read_mesh_file(self.path_to_mesh_file)
        return self._gt_mesh

    @groundtruth_mesh.setter
    def groundtruth_mesh(self, mesh):
        if self._gt_mesh is not None:
            raise RuntimeError("Trying to overwrite a mesh")
        self._gt_mesh = mesh

    @property
    def image_paths(self):
        if self._image_paths is None:
            self._image_paths = [
                os.path.join(self.images_dir, p)
                for p in sorted(os.listdir(self.images_dir))
                if p.endswith(".jpg") or p.endswith(".png")
            ]
        return self._image_paths

    def get_image(self, idx):
        return np.array(Image.open(self.image_paths[idx]).convert("RGB"))


class ModelCollection(object):
    def __len__(self):
        raise NotImplementedError()

    def _get_model(self, i):
        raise NotImplementedError()

    def __getitem__(self, i):
        if i >= len(self):
            raise IndexError()
        return self._get_model(i)


class ModelSubset(ModelCollection):
    def __init__(self, collection, subset):
        self._collection = collection
        self._subset = subset

    def __len__(self):
        return len(self._subset)

    def _get_sample(self, i):
        return self._collection[self._subset[i]]

    def __getitem__(self, i):
        if i >= len(self):
            raise IndexError()
        return self._get_sample(i)


class TagSubset(ModelSubset):
    def __init__(self, collection, tags):
        tags = set(tags)
        subset = [i for (i, m) in enumerate(collection) if m.tag in tags]
        super(TagSubset, self).__init__(collection, subset)


class RandomSubset(ModelSubset):
    def __init__(self, collection, percentage):
        N = len(collection)
        subset = np.random.choice(N, int(N*percentage)).tolist()
        super(RandomSubset, self).__init__(collection, subset)


class CategorySubset(ModelSubset):
    def __init__(self, collection, category_tags):
        category_tags = set(category_tags)
        subset = [
            i
            for (i, m) in enumerate(collection)
            if m.category in category_tags
        ]
        super(CategorySubset, self).__init__(collection, subset)


class DynamicFaust(ModelCollection):
    class Model(BaseModel):
        def __init__(self, base_dir, tag):
            super().__init__(tag)
            self._base_dir = base_dir
            self._category, self._sequence = tag.split(":")

        @property
        def category(self):
            return self._category

        @property
        def path_to_mesh_file(self):
            return os.path.join(self._base_dir, self._category,
                                "mesh_seq", self._sequence+".obj")
        @property
        def image_paths(self):
            return [os.path.join(self._base_dir, self._category,
                                 self._renderings_folder,
                                 "{}.png".format(self._sequence))]

    def __init__(self, base_dir):
        self._base_dir = base_dir
        self._paths = sorted([
            d
            for d in os.listdir(self._base_dir)
            if os.path.isdir(os.path.join(self._base_dir, d))
        ])

        # Note that we filter out the first 20 meshes from the sequence to
        # "discard" the neutral pose that is used for calibration purposes.
        self._tags = sorted([
            "{}:{}".format(d, l[:-4]) for d in self._paths
            for l in sorted(os.listdir(os.path.join(self._base_dir, d, mesh_folder)))[20:]
            if l.endswith(".obj")
        ])

        print("Found {} Dynamic Faust models".format(len(self)))

    def __len__(self):
        return len(self._tags)

    def _get_model(self, i):
        return self.Model(self._base_dir, self._tags[i])


class FreiHand(ModelCollection):
    class Model(BaseModel):
        def __init__(self, base_dir, tag):
            super().__init__(tag)
            self._base_dir = base_dir

        @property
        def category(self):
            return ""

        @property
        def path_to_mesh_file(self):
            return os.path.join(self._base_dir, self.tag + ".obj")

        @property
        def image_paths(self):
            return [os.path.join(self._base_dir, self.tag + ".png")]

    def __init__(self, base_dir):
        self._base_dir = base_dir
        self._tags = sorted([
            f[:-4]
            for f in os.listdir(self._base_dir)
            if f.endswith(".png")
        ])

    def __len__(self):
        return len(self._tags)

    def _get_model(self, i):
        return self.Model(self._base_dir, self._tags[i])


class TurbosquidAnimal(ModelCollection):
    class Model(BaseModel):
        def __init__(self, base_dir, tag):
            super().__init__(tag)
            self._base_dir = base_dir
            self._tag = tag

        @property
        def path_to_mesh_file(self):
            return os.path.join(
                self._base_dir, self._tag, "model_watertight.off"
            )

        @property
        def image_paths(self):
            return [os.path.join(self._base_dir, self.tag, "image_00000.png")]

    def __init__(self, base_dir):
        self._base_dir = base_dir

        self._tags = sorted([
            fi for fi in os.listdir(self._base_dir)
            if os.path.isdir(os.path.join(self._base_dir, fi))
        ])

        print("Found {} TurbosquidAnimal models".format(len(self)))

    def __len__(self):
        return len(self._tags)

    def _get_model(self, i):
        return self.Model(self._base_dir, self._tags[i])


class MultiModelsShapeNetV1(ModelCollection):
    class Model(BaseModel):
        def __init__(self, base_dir, tag):
            super().__init__(tag)
            self._base_dir = base_dir
            self._category, self._model = tag.split(":")

        @property
        def category(self):
            return self._category

        @property
        def path_to_mesh_file(self):
            return os.path.join(self._base_dir, self._category, self._model,
                                "model_off")

        @property
        def images_dir(self):
            return os.path.join(self._base_dir, self._category, self._model,
                                "img_choy2016")

    def __init__(self, base_dir):
        self._base_dir = base_dir
        self._models = sorted([
            d
            for d in os.listdir(self._base_dir)
            if os.path.isdir(os.path.join(self._base_dir, d))
        ])

        self._tags = sorted([
            "{}:{}".format(d, l) for d in self._models
            for l in os.listdir(os.path.join(self._base_dir, d))
            if os.path.isdir(os.path.join(self._base_dir, d, l))
        ])

        print("Found {} MultiModelsShapeNetV1 models".format(len(self)))

    def __len__(self):
        return len(self._tags)

    def _get_model(self, i):
        return self.Model(self._base_dir, self._tags[i])


class MeshCache(ModelCollection):
    """Cache the meshes from a collection and give them to the model before
    returning it."""
    def __init__(self, collection):
        self._collection = collection
        self._meshes = [None]*len(collection)

    def __len__(self):
        return len(self._collection)

    def _get_model(self, i):
        model = self._collection._get_model(i)
        if self._meshes[i] is not None:
            model.groundtruth_mesh = self._meshes[i]
        else:
            self._meshes[i] = model.groundtruth_mesh

        return model


class LRUCache(ModelCollection):
    def __init__(self, collection, n=2000):
        self._collection = collection
        self._cache = OrderedDict([])
        self._maxsize = n

    def __len__(self):
        return len(self._collection)

    def _get_model(self, i):
        m = None
        if i in self._cache:
            m = self._cache.pop(i)
        else:
            m = self._collection._get_model(i)
            if len(self._cache) > self._maxsize:
                self._cache.popitem()
        self._cache[i] = m
        return m


def model_factory(dataset_type):
    return {
        "dynamic_faust": DynamicFaust,
        "shapenet_v1": MultiModelsShapeNetV1,
        "freihand": FreiHand,
        "turbosquid_animal": TurbosquidAnimal
    }[dataset_type]


class ModelCollectionBuilder(object):
    def __init__(self):
        self._dataset_class = None
        self._cache_meshes = False
        self._lru_cache = 0
        self._tags = []
        self._category_tags = []
        self._percentage = 1.0

    def with_dataset(self, dataset_type):
        self._dataset_class = model_factory(dataset_type)
        return self

    def with_cache_meshes(self):
        self._cache_meshes = True
        return self

    def without_cache_meshes(self):
        self._cache_meshes = False
        return self

    def lru_cache(self, n=2000):
        self._lru_cache = n
        return self

    def filter_tags(self, tags):
        self._tags = tags
        return self

    def filter_category_tags(self, tags):
        self._category_tags = tags
        return self

    def random_subset(self, percentage):
        self._percentage = percentage
        return self

    def build(self, base_dir):
        dataset = self._dataset_class(base_dir)))
        if self._cache_meshes:
            dataset = MeshCache(dataset)
        if self._lru_cache > 0:
            dataset = LRUCache(dataset, self._lru_cache)
        if len(self._tags) > 0:
            prev_len = len(dataset)
            dataset = TagSubset(dataset, self._tags)
            print("Keep {}/{} based on tags".format(len(dataset), prev_len))
        if len(self._category_tags) > 0:
            prev_len = len(dataset)
            dataset = CategorySubset(dataset, self._category_tags)
            print("Keep {}/{} based on category tags".format(
                len(dataset), prev_len)
            )
        if self._percentage < 1.0:
            dataset = RandomSubset(dataset, self._percentage)

        return dataset
