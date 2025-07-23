import os
import numpy as np
from PIL import Image
from skimage.segmentation import felzenszwalb
from tqdm import tqdm


def generate_segmentation_files():
    source_path = r'C:\Users\Administrator\Desktop\CSINet\LEVIR-CD'
    data_list = ['/']
    use_list = ['A/', 'B/']
    generate_root = 'FELZEN_DATA_LEVIR_fe/'
    resize_size = (128, 128)
    segment_scale = 50  # 控制分割粒度（越大块越大）

    for name in data_list:
        try:
            if not os.path.exists(generate_root + name):
                os.makedirs(generate_root + name)
            for use in use_list:
                try:
                    if not os.path.exists(generate_root + name + use):
                        os.makedirs(generate_root + name + use)
                    target_path = generate_root + name + use
                    image_path = source_path + name + use
                    images = os.listdir(image_path)
                    for image_name in tqdm(images, desc=f"Processing {name}{use}"):
                        output_file = target_path + image_name.split('.')[0] + '.npy'
                        if not os.path.exists(output_file):
                            try:
                                # 读取并预处理图像
                                img = Image.open(os.path.join(image_path, image_name))
                                img = np.array(img)  # (H, W, C)

                                # Felzenszwalb分割（速度快，无需特征工程）
                                segments = felzenszwalb(
                                    img,
                                    scale=segment_scale,  # 控制分割块大小
                                    sigma=0.5,  # 高斯平滑参数
                                    min_size=20  # 最小区域尺寸
                                )

                                # 调整大小（保持标签不变，使用最近邻插值）
                                segments = Image.fromarray(segments.astype(np.uint8))
                                segments = segments.resize(resize_size, resample=Image.NEAREST)
                                segments = np.array(segments)

                                # 保存为npy文件
                                np.save(output_file, segments)

                            except Exception as e:
                                print(f"Error processing {image_name}: {e}")
                except Exception as e:
                    print(f"Error processing {name}{use}: {e}")
        except Exception as e:
            print(f"Error creating directory {generate_root + name}: {e}")

if __name__ == "__main__":
    generate_segmentation_files()