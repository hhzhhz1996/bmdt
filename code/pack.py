import os


def upx(src, dst):
    return f'upx -9 {src} -o {dst}'


def pecompact(src, dst):
    return f'PEC2 {src} /Nb'


def obsidium(src, dst):
    return f'obsi_cmd.exe ' \
           f'   --project C:/users/shini/desktop/demo.opf ' \
           f'   --input C:/users/shini/desktop/file_demo/*.exe ' \
           f'   --output C:/users/shini/desktop/file_demo --abort-on-error'


def telock(src, dst):
    return f'telock -S {src}'


def aspack(src, dst):
    return f'aspack {src} -o {dst}'


def themida(src, dst, project_path):
    return f'themida /protect {project_path} /inputfile {src} /outputfile {dst}'


def do_pack(src_dir, dst_dir, packer):
    ls = os.listdir(src_dir)
    sorted(ls)
    for each_file in ls:
        src = os.path.join(src_dir, each_file)
        dst = os.path.join(dst_dir, each_file)
        if packer == 'upx':
            command = upx(src, dst)
        elif packer == 'pecompact':
            command = pecompact(src, dst)
        elif packer == 'aspack':
            command = aspack(src, dst)
        elif packer == 'themida':
            command = themida(src, dst)
        elif packer == 'telock':
            command = telock(src, dst)
        else:
            command = 'None'
        if not command.__eq__('None'):
            os.system(command)
        else:
            return None
