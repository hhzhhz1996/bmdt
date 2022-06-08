import pefile
import os
import capstone
import shutil
from constant import BYTE_STREAM_LENGTH, api_mapping


def get_api_set_single(file_path, to_id=True):
    try:
        pe = pefile.PE(file_path)
    except Exception:
        raise TypeError("unsupported type or damaged file")

    if pe.DIRECTORY_ENTRY_IMPORT is None:
        raise ValueError("no imports")

    api_set = []
    for dll in pe.DIRECTORY_ENTRY_IMPORT:
        for imp in dll.imports:
            try:
                name = str(imp.name, 'utf-8')
                if name in api_mapping:
                    if to_id:
                        api_set.append(api_mapping[name])
                    else:
                        api_set.append(name)
            except Exception:
                continue
    return api_set


def get_api_set_batch(directory_path, to_id=True):
    file_ls = os.listdir(directory_path)
    sorted(file_ls)
    api_sets = []
    for index, file in enumerate(file_ls):
        file_path = os.path.join(directory_path, file)
        try:
            api_ls = get_api_set_single(file_path, to_id)
            api_sets.append(api_ls)
        except TypeError:
            continue
        except ValueError:
            api_sets.append([0])
    return api_sets


def get_api_seq_single(file_path, arch=capstone.CS_ARCH_X86, mode=capstone.CS_MODE_32, to_id=True):
    pe = pefile.PE(file_path)
    # eop = pe.OPTIONAL_HEADER.AddressOfEntryPoint
    # code_addr = pe.OPTIONAL_HEADER.ImageBase + code_section.VirtualAddress
    api_address = {}
    cnt = 0
    for dll in pe.DIRECTORY_ENTRY_IMPORT:
        for imp in dll.imports:
            try:
                addr, name = hex(imp.address), str(imp.name, encoding='utf-8')
                api_address[addr], api_address[name] = name, addr
                # print(name, addr)
                cnt += 1
            except Exception as e:
                pass
    # print(cnt)
    md = capstone.Cs(arch, mode)
    md.detail = True
    md.skipdata = True
    md.skipdata_setup = ("db", None, None)

    api_call_seq = []

    for section in pe.sections:
        for i in md.disasm(section.get_data(), section.VirtualAddress):
            # print("0x%x:\t%s\t%s" % (i.address, i.mnemonic, i.op_str))

            if i.mnemonic in ('call', 'jmp'):
                split = i.op_str.split(' ')
                if len(split) == 1:
                    address = split[0]
                elif len(split) == 3:
                    address = split[-1][1:-1]
                else:
                    address = ''
                if address in api_address:
                    name = api_address[address]
                    if name in api_mapping:
                        api_call_seq.append(api_mapping[name] if to_id else name)

    return api_call_seq


def get_api_seq_batch(directory_path):
    file_ls = os.listdir(directory_path)
    sorted(file_ls)
    api_call_seqs = []
    for index, file in enumerate(file_ls):
        try:
            api_call_seq = get_api_seq_single(os.path.join(directory_path, file), to_id=True)
            if len(api_call_seq) > 0:
                api_call_seqs.append(api_call_seq)
            else:
                print(file)
        except Exception as e:
            print(file)

    return api_call_seqs


def get_byte_stream_single(file_path):
    with open(file_path, 'rb') as f:
        s = f.read(BYTE_STREAM_LENGTH)
        ls = []
        for _ in s:
            ls.append(_ + 1)  # 0作为填充  所以+1

        if len(ls) < BYTE_STREAM_LENGTH:
            ls += [0 for i in range(BYTE_STREAM_LENGTH - len(ls))]
        return ls


def get_byte_stream_batch(directory_path):
    byte_streams = []
    for file in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file)
        byte_stream = get_byte_stream_single(file_path)
        byte_streams.append(byte_stream)

    return byte_streams


def get_pack_check_features(file_apth):




def dump_feature(lists, directory_path, file_name, append=False):
    action = 'a' if append else 'w'
    with open(os.path.join(directory_path, file_name), action) as wf:
        for line in lists:
            wf.write(','.join(line))
            wf.write('\n')


if __name__ == '__main__':
    # for test
    demo_file_path = r'D:\actProject\files\unpack\malicious\test' \
                r'\0421df2f603ce755ccef222281eca5de0cf804e42576204033d30dbf1006049e '
    path = r'C:/users/shini/desktop/malware'


    # single = get_api_seq_single(demo_file_path, to_id=False)
    # print('\n'.join(single))

    get_api_seq_batch(path)

    # f = open('C:/users/shini/desktop/result.txt', encoding='utf-8')
    # s = f.read()
    # f.close()
    # files = s.split('D:/actProject/files/unpack/malicious/train/')
    # files = files[1:]
    # for file in files:
    #     ms = file.split('\n')
    #     name = ms[0][:ms[0].find(' ')]
    #     _type = ms[1]
    #     if '打包工具' in file or '保护器' in file or _type != 'PE32':
    #         continue
    #
    #     src = os.path.join('D:/actProject/files/unpack/malicious/train/', name)
    #     dst = os.path.join('C:/users/shini/desktop/malware', name)
    #     shutil.copy(src, dst)