from dataclasses import dataclass, field
from typing import List, Optional
from typing import Optional


@dataclass
class Category:
    id: int
    name: str
    supercategory: Optional[str] = None


@dataclass
class Image:
    id: int
    file_name: str
    width: int
    height: int


@dataclass
class BBox:
    x: float
    y: float
    width: float
    height: float


@dataclass
class Annotation:
    id: int
    image_id: int
    category_id: int
    bbox: BBox
    iscrowd: int = 0
    segmentation: Optional[list[float]] = None
    area: Optional[float] = None
    keypoints: Optional[list[float]] = None
    num_keypoints: Optional[int] = None


@dataclass
class DatasetIR:
    images: List[Image] = field(default_factory=list)
    annotations: List[Annotation] = field(default_factory=list)
    categories: List[Category] = field(default_factory=list)

    def get_image(self, image_id: int) -> Image:
        for im in self.images:
            if im.id == image_id:
                return im
        raise KeyError(f"Image id={image_id} not found")

    def get_annotations_for_image(self, image_id: int) -> list[Annotation]:
        return [a for a in self.annotations if a.image_id == image_id]

    def get_category(self, category_id: int) -> Category:
        for cat in self.categories:
            if cat.id == category_id:
                return cat
        raise KeyError(f"Category id={category_id} not found")
