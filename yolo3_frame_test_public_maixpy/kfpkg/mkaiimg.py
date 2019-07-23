from skimage.io import imread, imsave
from skimage.transform import resize
import numpy as np
import sys

if __name__ == '__main__':
    img = imread(sys.argv[1])
    print(sys.argv[1])
    img = resize(img, (224, 320), preserve_range=True).astype('uint8')
    imsave('test.jpg', img)
    img = np.transpose(img, [2, 0, 1])
    with open('aiimg.h', 'w') as f:
        f.write('#ifndef _AIIMG_H_\n#define _AIIMG_H_\n#include <stdint.h>\nconst unsigned char ai_image[] __attribute__((aligned(128))) = {')
        f.write(', '.join([str(i) for i in img.flatten()]))
        f.write('};\n#endif')
