import os
import sys

if __name__ == '__main__':
    demo_file_path = r'D:\actProject\files\unpack\malicious\test' \
                     r'\0421df2f603ce755ccef222281eca5de0cf804e42576204033d30dbf1006049e '
    path = r'C:/users/shini/desktop/benign'

    # get_api_seq_batch(path)

    f = open('C:/users/shini/desktop/要删除的.txt')

    for file in f.readlines():
        file = file.strip()
        path = os.path.join('C:/users/shini/desktop/benign', file)
        try:
            os.remove(path)
        except:
            pass
