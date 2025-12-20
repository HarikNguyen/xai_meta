import os
from PIL import Image


# Resize toàn bộ ảnh trong thư mục con của train/, val/, test/
def resize_all_images(root_dirs, size=(84, 84)):
    for root_dir in root_dirs:
        print(f"Đang xử lý: {root_dir}")
        for label in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, label)
            if not os.path.isdir(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                try:
                    with Image.open(img_path) as im:
                        im = im.resize(size, Image.LANCZOS)
                        im.save(img_path)
                except Exception as e:
                    print(f"Lỗi với ảnh: {img_path} - {e}")
        print(f"Đã xong {root_dir}")


# Gọi hàm resize
resize_all_images(["miniImagenet/train", "miniImagenet/val", "miniImagenet/test"])
